"""E11 figures and statistics (Grad-CAM, DCN offset maps, CAD-channel attribution) from trained checkpoints.

  python tools/make_gradcam.py operators                       # U-Net / DCNv2 / DCNv3 / DCNv4 (SEM+CAD) on simulated SEM
  python tools/make_gradcam.py cad                             # SEM+CAD DCNv3 vs SEM-only DCNv3
  python tools/make_gradcam.py carinthia                       # four operators on real Carinthia-S defects
  python tools/make_gradcam.py stages --runs A B C --labels a b c   # roadmap stages on the same simulated images
Outputs: results/gradcam/*.png and results/gradcam/*_stats.csv
"""
import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from skimage import measure  # noqa: E402

import evaluate as ev  # noqa: E402
import explain as ex  # noqa: E402
import registration as rg  # noqa: E402
import train as tr  # noqa: E402
from data_lithobench import load_config  # noqa: E402
from models import UNet, build_for_run  # noqa: E402

DEV = "cuda"
OPS = [("screen_conv_cad_s1", "U-Net"), ("screen_dcnv2_cad_s1", "DCNv2"), ("screen_dcnv3_cad_s1", "DCNv3"), ("screen_dcnv4_cad_s1", "DCNv4")]
LAYERS = ("mid", "dec_op1", "last")
LAYER_TITLE = {"mid": "Grad-CAM, deepest deformable block (1/16)", "dec_op1": "Grad-CAM, last deformable block (1/4)",
               "last": "Grad-CAM, final decoder layer (1/1)"}


def out_dir(cfg):
    d = os.path.join(cfg["paths"]["results"], "gradcam")
    os.makedirs(d, exist_ok=True)
    return d


def load_run(cfg, run):
    ck = torch.load(os.path.join(ROOT, "checkpoints", run, "best.pt"), map_location=DEV, weights_only=False)
    m = build_for_run(cfg["model"], ck["run"]).to(DEV)
    m.load_state_dict(ck["model"])
    return m.eval(), ck["run"]


def sim_item(cfg, split, set_name, tile, cond, dose):
    """Everything needed to explain one simulated image: noisy SEM, estimated registration, probes, true contour."""
    S, sc = cfg["lithobench"]["crop_px"], cfg["sim"]
    d = ev.load_cache(os.path.join(cfg["paths"]["results"], "raw", "cache", split, set_name, tile + ".npz"))
    r = np.load(os.path.join(cfg["paths"]["results"], "raw", ev.reg_dirname(cfg), split, set_name, tile + ".npz"))
    j = [tuple(x) for x in r["jobs"].tolist()].index((cond, dose, 0))
    img = ev.eval_noisy(d["clean"][cond], dose, int(d["seed"]) * 100 + cond, 0, sc)
    A_est, b_est = r["A"][j], r["b"][j]
    cad = tr.cad_in_sem(d["T"], A_est, b_est, S, DEV)
    Ft = rg._warp(torch.as_tensor(d["F"][cond].astype(np.float32))[None, None], torch.as_tensor(d["A"][cond], dtype=torch.float32)[None],
                  torch.as_tensor(d["b"][cond], dtype=torch.float32)[None], S)[0, 0].numpy()
    return dict(img=img, cad=cad, A=A_est, b=b_est, probes=d["probes"], truth=Ft, set=set_name, tile=tile)


def model_input(item, with_cad):
    x = torch.as_tensor(item["img"], device=DEV, dtype=torch.float32)[None, None]
    x = (x - x.mean()) / x.std().clamp(min=1e-6)
    if with_cad:
        x = torch.cat([x, torch.as_tensor(item["cad"], device=DEV, dtype=torch.float32)[None, None]], 1)
    return x


def explain_sim(cfg, model, run, item):
    x = model_input(item, run["cad"])
    cams, attr, _, score = ex.grad_cam(model, x, ex.metrology_score(cfg, item["probes"], item["A"], item["b"]))
    dist = ex.contour_distance(item["truth"] > 0)
    row = {}
    for k in LAYERS:
        row.update({f"{k}_{a}": v for a, v in ex.focus_stats(cams[k], dist, inside=item["truth"] > 0).items()})
    offs = {}
    for level in ("dec_op1", "mid"):
        mag = ex.offset_magnitude(model, x, level)
        if mag is not None:
            offs[level] = mag
            row.update({f"off_{level}_{a}": v for a, v in ex.offset_stats(mag, dist, ex.STRIDE[level]).items()})
    row["cad_attribution_share"] = float(attr[1]) if len(attr) > 1 else float("nan")
    return cams, offs, row


def show(ax, img, cam=None, contour=None, title=None, cmap="gray", alpha=0.55):
    ax.imshow(img, cmap=cmap)
    if cam is not None:
        ax.imshow(cam, cmap="jet", alpha=alpha, vmin=0, vmax=1)
    if contour is not None:
        for c in measure.find_contours(contour[0], contour[1]):
            ax.plot(c[:, 1], c[:, 0], c="white", lw=0.8)
    if title:
        ax.set_title(title, fontsize=8)
    ax.axis("off")


def pick_items(cfg, split="val", dose=8, per_set=1, offset=2):
    cond = ev.nominal_index(cfg)
    tiles = ev.eval_tiles(cfg, split)
    items = []
    for s in ("MetalSet", "ViaSet"):
        ids = [t for s2, t in tiles if s2 == s]
        items += [sim_item(cfg, split, s, ids[offset + i], cond, dose) for i in range(per_set)]
    return items


def stat_items(cfg, split="val", dose=8, n_per_set=15):
    cond = ev.nominal_index(cfg)
    tiles = ev.eval_tiles(cfg, split)
    return [sim_item(cfg, split, s, t, cond, dose) for s in ("MetalSet", "ViaSet") for t in [t for s2, t in tiles if s2 == s][:n_per_set]]


def operators_cmd(cfg, runs, tag):
    od = out_dir(cfg)
    models = [(lab, *load_run(cfg, r)) for r, lab in runs]
    ex_items = pick_items(cfg)
    for it in ex_items:                                        # figures
        fig, ax = plt.subplots(len(models), 5, figsize=(14, 2.9 * len(models)))
        for i, (lab, m, run) in enumerate(models):
            cams, offs, _ = explain_sim(cfg, m, run, it)
            show(ax[i, 0], it["img"], contour=(it["truth"], 0), title=f"{lab}: SEM (8 frames) + true contour" if i == 0 else lab)
            for c, k in enumerate(LAYERS):
                show(ax[i, 1 + c], it["img"], cams[k], (it["truth"], 0), LAYER_TITLE[k] if i == 0 else None)
            if "dec_op1" in offs:
                o = ax[i, 4].imshow(offs["dec_op1"] * ex.STRIDE["dec_op1"], cmap="magma", vmin=0)
                ax[i, 4].set_title("learned sampling offset, last DCN block (px)" if i == 0 else "", fontsize=8)
                plt.colorbar(o, ax=ax[i, 4], fraction=0.046)
            else:
                ax[i, 4].text(0.5, 0.5, "regular convolution\n(no offsets)", ha="center", va="center", fontsize=9)
            ax[i, 4].axis("off")
        fig.suptitle(f"Grad-CAM for the predicted CD (sum over CD gauges), {it['set']}/{it['tile']}; white = true printed contour", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(od, f"{tag}_{it['set']}.png"), dpi=110)
        plt.close(fig)
    rows = []
    for it in stat_items(cfg):                                 # statistics
        for lab, m, run in models:
            _, _, row = explain_sim(cfg, m, run, it)
            rows.append(dict(model=lab, set=it["set"], tile=it["tile"], **row))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(od, f"{tag}_stats.csv"), index=False)
    agg = df.groupby("model", sort=False).mean(numeric_only=True)
    agg.to_csv(os.path.join(od, f"{tag}_stats_mean.csv"))
    cols = [c for c in agg.columns if c.startswith(("mid_cam_far", "mid_area_far", "mid_cam_inside", "dec_op1_cam_inside", "dec_op1_area_inside", "off_dec_op1", "cad_att"))]
    print(agg[cols].round(3).T.to_string())


def carinthia_cmd(cfg):
    import carinthia as ca
    od = out_dir(cfg)
    split = __import__("json").load(open(ca.SPLIT_JSON))
    border = cfg["carinthia"]["right_border_px"]
    ck = os.path.join(ROOT, "checkpoints")
    per = pd.read_csv(os.path.join(ck, "carinthia_dcnv3_s1", "test_per_image.csv"))
    sel = {}
    for lab, name in ((4, "pit"), (3, "particle"), (1, "scratch")):
        g = per[per.label == lab]
        sel[name] = g.iloc[(g.dice - g.dice.median()).abs().argmin()].filename
    rng = np.random.default_rng(0)
    stat_files = []
    for lab in (1, 2, 3, 4, 5):
        f = per[per.label == lab].filename.to_numpy()
        stat_files += list(rng.choice(f, min(len(f), 20), replace=False))
    files = list(dict.fromkeys(list(sel.values()) + stat_files))
    X, Y, L = ca.load_arrays(cfg, files)
    valid = ca.valid_region(X.shape[-2:], border)
    V = torch.as_tensor(valid, device=DEV)[None, None].float()
    Xt = torch.as_tensor(X, device=DEV)[:, None]
    ops = [("conv", "U-Net"), ("dcnv2", "DCNv2"), ("dcnv3", "DCNv3"), ("dcnv4", "DCNv4")]
    models = []
    for op, lab in ops:
        m = UNet(in_ch=1, width=cfg["model"]["width"][op], op=op).to(DEV)
        m.load_state_dict(torch.load(os.path.join(ck, f"carinthia_{op}_s1", "best.pt"), map_location=DEV))
        models.append((lab, m.eval()))
    rows, cache = [], {}
    for fi, f in enumerate(files):
        x = ca._norm(Xt[fi:fi + 1], V)
        gt = Y[fi]
        dist = ex.contour_distance(gt)
        for lab, m in models:
            cams, _, out, _ = ex.grad_cam(m, x, ex.segmentation_score(gt))
            row = dict(model=lab, file=f, label=int(L[fi]))
            for k in ("dec_op1", "last"):
                w = cams[k] * valid / max((cams[k] * valid).sum(), 1e-12)
                row[f"{k}_cam_in_defect"] = float(w[gt].sum())
                row[f"{k}_area_in_defect"] = float(gt.mean())
                row[f"{k}_cam_within_8px"] = float(w[dist < 8].sum())
                row[f"{k}_area_within_8px"] = float((dist < 8).mean())
            mag = ex.offset_magnitude(m, x, "dec_op1")
            if mag is not None:
                row.update({f"off_{a}": v for a, v in ex.offset_stats(mag, dist, ex.STRIDE["dec_op1"], near=8).items()})
                cache[(lab, f)] = mag
            rows.append(row)
            if f in sel.values():
                cache[(lab, f, "cam")] = cams
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(od, "carinthia_stats.csv"), index=False)
    agg = df.groupby("model", sort=False).mean(numeric_only=True)
    agg.to_csv(os.path.join(od, "carinthia_stats_mean.csv"))
    print(agg.round(3).T.to_string())
    for name, f in sel.items():
        fi = files.index(f)
        fig, ax = plt.subplots(len(models), 4, figsize=(11, 2.8 * len(models)))
        for i, (lab, m) in enumerate(models):
            cams = cache[(lab, f, "cam")]
            show(ax[i, 0], X[fi], contour=(Y[fi].astype(float), 0.5), title=f"{lab}: SEM + annotation" if i == 0 else lab)
            show(ax[i, 1], X[fi], cams["dec_op1"], (Y[fi].astype(float), 0.5), LAYER_TITLE["dec_op1"] if i == 0 else None)
            show(ax[i, 2], X[fi], cams["last"], (Y[fi].astype(float), 0.5), LAYER_TITLE["last"] if i == 0 else None)
            mag = cache.get((lab, f))
            if mag is not None:
                o = ax[i, 3].imshow(mag * ex.STRIDE["dec_op1"], cmap="magma", vmin=0)
                ax[i, 3].set_title("learned sampling offset (px)" if i == 0 else "", fontsize=8)
                plt.colorbar(o, ax=ax[i, 3], fraction=0.046)
            else:
                ax[i, 3].text(0.5, 0.5, "regular convolution", ha="center", va="center", fontsize=9)
            ax[i, 3].axis("off")
        fig.suptitle(f"Seg-Grad-CAM on real SEM (Carinthia-S, {name}); white = annotated defect", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(od, f"carinthia_{name}.png"), dpi=110)
        plt.close(fig)


def stages_cmd(cfg, runs, labels, tag):
    od = out_dir(cfg)
    models = [(lab, *load_run(cfg, r)) for r, lab in zip(runs, labels)]
    items = pick_items(cfg)
    for it in items:
        fig, ax = plt.subplots(len(models), 4, figsize=(11, 2.8 * len(models)))
        base = None
        for i, (lab, m, run) in enumerate(models):
            cams, _, _ = explain_sim(cfg, m, run, it)
            show(ax[i, 0], it["img"], contour=(it["truth"], 0), title=f"{lab}: SEM + true contour" if i == 0 else lab)
            show(ax[i, 1], it["img"], cams["dec_op1"], (it["truth"], 0), LAYER_TITLE["dec_op1"] if i == 0 else None)
            show(ax[i, 2], it["img"], cams["last"], (it["truth"], 0), LAYER_TITLE["last"] if i == 0 else None)
            base = cams["last"] if base is None else base
            ax[i, 3].imshow(cams["last"] - base, cmap="coolwarm", vmin=-1, vmax=1)
            ax[i, 3].set_title("final-layer CAM minus first stage" if i == 0 else "", fontsize=8)
            ax[i, 3].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(od, f"{tag}_{it['set']}.png"), dpi=110)
        plt.close(fig)
    rows = []
    for it in stat_items(cfg, n_per_set=10):
        for lab, m, run in models:
            _, _, row = explain_sim(cfg, m, run, it)
            rows.append(dict(stage=lab, set=it["set"], tile=it["tile"], **row))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(od, f"{tag}_stats.csv"), index=False)
    print(df.groupby("stage", sort=False).mean(numeric_only=True)[["dec_op1_cam_within_4px", "last_cam_within_4px"]].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["operators", "cad", "carinthia", "stages"])
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    cfg = load_config()
    if a.cmd == "operators":
        operators_cmd(cfg, OPS, "operators_sim")
    elif a.cmd == "cad":
        operators_cmd(cfg, [("screen_dcnv3_cad_s1", "DCNv3 SEM+CAD"), ("cadtest_dcnv3_semonly_s1", "DCNv3 SEM only")], "cad_vs_sem")
    elif a.cmd == "carinthia":
        carinthia_cmd(cfg)
    else:
        stages_cmd(cfg, a.runs, a.labels, a.tag or "stages")
