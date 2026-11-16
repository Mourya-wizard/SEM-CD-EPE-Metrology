"""Grad-CAM v2 (E11, extended to E16 and E17): ONE fixed figure layout for every dataset and every model set.

  python tools/make_gradcam.py list                    # model sets and checkpoints (existing / missing)
  python tools/make_gradcam.py run <set> [<set> ...]   # figures + statistics, or `run all`
Output: results/gradcam/<set>/*.png and <set>_stats.csv / <set>_stats_mean.csv;  `teaser` -> results/figures/gradcam_overview.png

Every figure has the same layout. Rows = models in a FIXED order (conv-32, conv-36, DCNv2, DCNv3, DCNv4). Columns, left to right = the
network from input to output:   Input | Enc 1/4 | Enc 1/8 | Bottleneck 1/16 | Dec 1/8 | Dec 1/4 | Final 1/1 | Sampling offsets.
The five middle CAM columns are the five operator blocks of the U-Net (results/architecture); "Final" is the last decoder layer.
Every CAM panel is scaled to its own maximum (red = most influential for the model's decision); offsets share one scale per figure.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from skimage import measure  # noqa: E402

import explain as ex  # noqa: E402
import gradcam_inputs as g  # noqa: E402  (simulated-image loading, model inputs)
from data_lithobench import load_config  # noqa: E402
from models import UNet, n_params  # noqa: E402
import paperstyle  # noqa: E402

paperstyle.use()

DEV = "cuda"
# column key, title line 1, title line 2, group
COLS = [("enc_op2", "operator block", "1/4 resolution", "Encoder"), ("enc_op3", "operator block", "1/8 resolution", "Encoder"),
        ("mid", "operator block", "1/16 resolution", "Bottleneck"), ("dec_op0", "operator block", "1/8 resolution", "Decoder"),
        ("dec_op1", "operator block", "1/4 resolution", "Decoder"), ("last", "last decoder layer", "1/1 resolution", "Decoder")]
OFF_LEVELS = ("enc_op2", "enc_op3", "mid", "dec_op0", "dec_op1")
FIG_OFF = "dec_op1"

# model sets: key -> (title, dataset, [(label, checkpoint name)]);  label = what is printed on the row
S = {1: "s1", 2: "s2"}


def e16_sim(seed):
    return [(f"Conv, w=32", f"e16_conv_w32_semonly_s{seed}"), (f"Conv, w=36", f"e16_conv_w36_semonly_s{seed}"),
            (f"DCNv2, w=36", f"e16_dcnv2_w36_semonly_s{seed}"), (f"DCNv3, w=36", f"cadtest_dcnv3_semonly_s{seed}"),
            (f"DCNv4, w=36", f"e16_dcnv4_w36_semonly_s{seed}")]


def e16_car(seed):
    return [("Conv, w=32", f"carinthia_conv_s{seed}"), ("Conv, w=36", f"e16_carinthia_conv_w36_s{seed}"),
            ("DCNv2, w=36", f"e16_carinthia_dcnv2_w36_s{seed}"), ("DCNv3, w=36", f"carinthia_dcnv3_s{seed}"),
            ("DCNv4, w=36", f"carinthia_dcnv4_s{seed}")]


SETS = {
    "sim_e4_operators": ("Simulated SEM, parameter-matched operators, SEM + CAD input (E4)", "sim",
                         [("Conv (U-Net), w=32", "screen_conv_cad_s1"), ("DCNv2, w=32", "screen_dcnv2_cad_s1"),
                          ("DCNv3, w=36", "screen_dcnv3_cad_s1"), ("DCNv4, w=36", "screen_dcnv4_cad_s1")]),
    "sim_e6_cad": ("Simulated SEM, DCNv3 with and without CAD input (E6)", "sim",
                   [("DCNv3, SEM + CAD", "screen_dcnv3_cad_s1"), ("DCNv3, SEM only", "cadtest_dcnv3_semonly_s1")]),
    "sim_e13_e14_stages": ("Simulated SEM, DCNv3 training variants (E13, E14)", "sim",
                           [("Baseline", "cadtest_dcnv3_semonly_s1"), ("+ E13 edge-offset head", "e13_dcnv3_semonly_aux_s1"),
                            ("+ E14 spectrum loss", "e14_dcnv3_semonly_spec_s1")]),
    "sim_e16_seed1": ("Simulated SEM, equal-width operators (E16, seed 1)", "sim", e16_sim(1)),
    "sim_e16_seed2": ("Simulated SEM, equal-width operators (E16, seed 2)", "sim", e16_sim(2)),
    "carinthia_e9_operators": ("Carinthia-S, parameter-matched operators (E9, seed 1)", "carinthia",
                               [("Conv (U-Net), w=32", "carinthia_conv_s1"), ("DCNv2, w=32", "carinthia_dcnv2_s1"),
                                ("DCNv3, w=36", "carinthia_dcnv3_s1"), ("DCNv4, w=36", "carinthia_dcnv4_s1")]),
    "carinthia_e16_seed1": ("Carinthia-S, equal-width operators (E16, seed 1)", "carinthia", e16_car(1)),
    "maxplanck_e17_seed1": ("Max Planck IC metal-2, equal-width operators (E17, seed 1)", "maxplanck",
                            [(l, f"e17_maxplanck_{t}_s1") for l, t in (("Conv, w=32", "conv_w32"), ("Conv, w=36", "conv_w36"), ("DCNv2, w=36", "dcnv2_w36"),
                                                                       ("DCNv3, w=36", "dcnv3_w36"), ("DCNv4, w=36", "dcnv4_w36"))]),
}


# ------------------------------------------------------------------------------------------------------------ loading
def ckpt_path(name):
    return os.path.join(ROOT, "checkpoints", name, "best.pt")


def load_any(cfg, name):
    """Return (model, run dict) for any checkpoint: metrology runs store the run in the checkpoint, segmentation runs in metrics.json."""
    ck = torch.load(ckpt_path(name), map_location=DEV, weights_only=False)
    if isinstance(ck, dict) and "run" in ck and "model" in ck:
        m = g.build_for_run(cfg["model"], ck["run"]).to(DEV)
        m.load_state_dict(ck["model"])
        return m.eval(), ck["run"]
    run = json.load(open(os.path.join(ROOT, "checkpoints", name, "metrics.json")))["run"]
    m = UNet(in_ch=1, width=run.get("width") or cfg["model"]["width"][run["op"]], op=run["op"]).to(DEV)
    m.load_state_dict(ck)
    return m.eval(), dict(run, cad=False)


# --------------------------------------------------------------------------------------------------------- explanations
def stats_row(cams, offs, dist, inside, near=4):
    row = {}
    for k, *_ in COLS:
        fs = ex.focus_stats(cams[k], dist, near=(near,), inside=inside)
        row[f"{k}_edge_enrichment"] = fs[f"cam_within_{near}px"] / max(fs[f"area_within_{near}px"], 1e-9)   # 1 = chance
        row[f"{k}_cam_inside_features"] = fs["cam_inside_features"]
    for lv in OFF_LEVELS:
        if offs.get(lv) is not None:
            o = ex.offset_stats(offs[lv], dist, ex.STRIDE[lv], near=near)
            row[f"off_{lv}_near_px"], row[f"off_{lv}_far_px"] = o["offset_near_px"], o["offset_far_px"]
    return row


def explain_sim(cfg, model, run, item):
    x = g.model_input(item, run["cad"])
    cams, attr, _, _ = ex.grad_cam(model, x, ex.metrology_score(cfg, item["probes"], item["A"], item["b"]))
    offs = {lv: ex.offset_magnitude(model, x, lv) for lv in OFF_LEVELS}
    inside = item["truth"] > 0
    row = stats_row(cams, offs, ex.contour_distance(inside), inside)
    row["cad_attribution_share"] = float(attr[1]) if len(attr) > 1 else float("nan")
    return cams, offs, row


def explain_seg(cfg, model, x, gt, valid):
    cams, _, _, _ = ex.grad_cam(model, x, ex.segmentation_score(gt))
    offs = {lv: ex.offset_magnitude(model, x, lv) for lv in OFF_LEVELS}
    cams = {k: v * valid for k, v in cams.items()}
    return cams, offs, stats_row(cams, offs, ex.contour_distance(gt), gt)


# -------------------------------------------------------------------------------------------------------------- figure
def draw_figure(path, title, caption, rows, in_title, contour_note):
    """rows: one dict per model (label, params, img, gt field, gt_level, cams{layer}, offs{level}). All figures share this layout."""
    import textwrap
    n = len(rows)
    cw, rh, top_in, bot_in, left_in = 1.95, 2.0, 2.15, 2.6, 2.5
    W, H = cw * 8 + left_in + 0.3, rh * n + top_in + bot_in
    L, R_ = left_in / W, 1 - 0.3 / W
    fr = lambda y_in: y_in / H                                                      # noqa: E731  inches from the bottom -> figure fraction
    fig, axes = plt.subplots(n, 8, figsize=(W, H), squeeze=False)
    fig.subplots_adjust(left=L, right=R_, top=1 - fr(top_in), bottom=fr(bot_in), wspace=0.035, hspace=0.05)
    vmax = max([np.nanpercentile(p["offs"][FIG_OFF] * ex.STRIDE[FIG_OFF], 99) for p in rows if p["offs"].get(FIG_OFF) is not None] + [1e-6])
    for i, p in enumerate(rows):
        a = axes[i, 0]
        a.imshow(p["img"], cmap="gray")
        for c in measure.find_contours(p["gt"], p["gt_level"]):
            a.plot(c[:, 1], c[:, 0], c="white", lw=0.8)
        a.text(-0.06, 0.5, f"{p['label']}\n{p['params'] / 1e6:.2f} M params", transform=a.transAxes, ha="right", va="center", fontsize=18, weight="bold", linespacing=1.25)
        for j, (k, *_rest) in enumerate(COLS):
            a = axes[i, 1 + j]
            a.imshow(p["img"], cmap="gray")
            a.imshow(p["cams"][k], cmap="jet", alpha=0.55, vmin=0, vmax=1)
            for c in measure.find_contours(p["gt"], p["gt_level"]):
                a.plot(c[:, 1], c[:, 0], c="white", lw=0.5, alpha=0.9)
        a = axes[i, 7]
        o = p["offs"].get(FIG_OFF)
        if o is not None:
            a.imshow(o * ex.STRIDE[FIG_OFF], cmap="magma", vmin=0, vmax=vmax)
        else:
            a.set_facecolor("#f1f0ec")
            a.text(0.5, 0.5, "regular conv:\nno offsets", ha="center", va="center", fontsize=15, color="#555", transform=a.transAxes)
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
    heads = [("Input", in_title)] + [(t1, t2) for _, t1, t2, _ in COLS] + [("Sampling offsets", "decoder 1/4 block")]
    for j, (t1, t2) in enumerate(heads):
        axes[0, j].set_title(f"{t1}\n{t2}", fontsize=16, linespacing=1.2, pad=6)
    fig.canvas.draw()
    ytop = axes[0, 0].get_position().y1 + fr(0.8)                                  # group brackets above the CAM columns
    for name, js in (("ENCODER", (1, 2)), ("BOTTLENECK", (3,)), ("DECODER", (4, 5, 6))):
        x0, x1 = axes[0, js[0]].get_position().x0, axes[0, js[-1]].get_position().x1
        fig.add_artist(Line2D([x0 + 0.004, x1 - 0.004], [ytop, ytop], c="#333", lw=1.4))
        fig.text((x0 + x1) / 2, ytop + fr(0.04), name, ha="center", va="bottom", fontsize=17, weight="bold", color="#333")
    fig.text((L + R_) / 2, 1 - fr(0.12), title, ha="center", va="top", fontsize=22, weight="bold")
    y0 = axes[-1, 0].get_position().y0                                              # shared colour bars
    for ax_, cmap_, norm_, x0, x1, lab, ticks in (
            ("cam", "jet", plt.Normalize(0, 1), axes[-1, 1].get_position().x0, axes[-1, 6].get_position().x1 - 0.012,
             "Grad-CAM, relative influence (each panel scaled to its maximum)", [0, 0.25, 0.5, 0.75, 1]),
            ("off", "magma", plt.Normalize(0, vmax), axes[-1, 7].get_position().x0 + 0.004, axes[-1, 7].get_position().x1 - 0.004, "sampling offset (px)", None)):
        cax = fig.add_axes([x0, y0 - fr(0.45), x1 - x0, fr(0.2)])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_, cmap=cmap_), cax=cax, orientation="horizontal", ticks=ticks)
        cb.set_label(lab, fontsize=16, labelpad=4)
        cax.tick_params(labelsize=15)
    note = contour_note[0].upper() + contour_note[1:] + "."
    fig.text(L, y0 - fr(1.4), textwrap.fill(caption + " " + note, 130), ha="left", va="top", fontsize=15, color="#333", linespacing=1.35)
    fig.savefig(path, dpi=100, pil_kwargs={"optimize": True})
    plt.close(fig)


CAPTION_CAM = ("Columns follow the network from input to output: five operator blocks, then the last decoder layer. "
               "Each panel is scaled to its own maximum; compare where the map is high, not its absolute level.")


# ------------------------------------------------------------------------------------------------------------- datasets
def run_sim(cfg, key, title, models, od):
    cams_by = {}
    loaded = [(lab, *load_any(cfg, ck)) for lab, ck in models]
    items = g.pick_items(cfg)
    for it in items:
        rows = []
        for lab, m, run in loaded:
            cams, offs, _ = explain_sim(cfg, m, run, it)
            rows.append(dict(label=lab, params=n_params(m), img=it["img"], gt=it["truth"], gt_level=0, cams=cams, offs=offs))
        draw_figure(os.path.join(od, f"{key}_{it['set']}.png"), f"{title}: {it['set']} tile, 8 frames",
                    CAPTION_CAM + (" Explained output: summed predicted edge positions over the " + ("CD gauges." if it["set"] == "MetalSet" else "contact gauges.")) , rows, "SEM, 8 frames", "white line = exact printed contour (simulator truth)")
    allrows = []
    for it in g.stat_items(cfg, n_per_set=15):
        for lab, m, run in loaded:
            _, _, row = explain_sim(cfg, m, run, it)
            allrows.append(dict(model=lab, set=it["set"], tile=it["tile"], **row))
    return pd.DataFrame(allrows)


def car_names():
    return {1: "scratch", 3: "particle", 4: "pit"}


def run_carinthia(cfg, key, title, models, od):
    import carinthia as ca
    split = json.load(open(ca.SPLIT_JSON))
    border = cfg["carinthia"]["right_border_px"]
    per = pd.read_csv(os.path.join(ROOT, "checkpoints", "carinthia_dcnv3_s1", "test_per_image.csv"))
    sel = {car_names()[lab]: per[per.label == lab].iloc[(per[per.label == lab].dice - per[per.label == lab].dice.median()).abs().argmin()].filename for lab in car_names()}
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
    loaded = [(lab, *load_any(cfg, ck)) for lab, ck in models]
    allrows, shown = [], {}
    for fi, f in enumerate(files):
        x = ca._norm(Xt[fi:fi + 1], V)
        for lab, m, run in loaded:
            cams, offs, row = explain_seg(cfg, m, x, Y[fi], valid)
            allrows.append(dict(model=lab, file=f, label=int(L[fi]), **row))
            if f in sel.values():
                shown.setdefault(f, []).append(dict(label=lab, params=n_params(m), img=X[fi], gt=Y[fi].astype(float), gt_level=0.5, cams=cams, offs=offs))
    for name, f in sel.items():
        draw_figure(os.path.join(od, f"{key}_{name}.png"), f"{title}: {name} defect",
                    CAPTION_CAM + " Explained output: summed defect logits.", shown[f], "SEM image", "white line = annotated defect outline")
    return pd.DataFrame(allrows)


def run_maxplanck(cfg, key, title, models, od):
    """Max Planck IC metal-2 test tiles: a typical tile (median Dice of the reference model), a hard tile (10th percentile Dice), a via-rich tile."""
    import maxplanck as mp
    img, lab = mp.arrays()
    _, idx = mp.load_split()
    tl = mp.grid_tiles(idx["test"], img.shape[1:])
    ref = pd.read_csv(os.path.join(ROOT, "checkpoints", models[1][1], "test_tiles.csv")).dice.to_numpy()      # conv-36 per-tile Dice, same tile order
    order = np.argsort(ref)
    vias = np.array([((lab[k, y:y + mp.T, x:x + mp.T] >> 1) & 1).mean() for k, y, x in tl])
    sel = {"typical_tile": int(order[len(order) // 2]), "hard_tile": int(order[len(order) // 10]), "via_rich_tile": int(vias.argmax())}
    rng = np.random.default_rng(0)
    stat = sorted(set(rng.choice(len(tl), 150, replace=False).tolist()) | set(sel.values()))
    loaded = [(lab_, *load_any(cfg, ck)) for lab_, ck in models]
    ones = np.ones((mp.T, mp.T), np.float32)
    rows, shown = [], {}
    for j in stat:
        k, y, x = tl[j]
        tile, gt = np.array(img[k, y:y + mp.T, x:x + mp.T]), (np.array(lab[k, y:y + mp.T, x:x + mp.T]) & 3) > 0
        xin = mp.norm(torch.as_tensor(tile, device=DEV).float()[None, None])
        for lab_, m, run in loaded:
            cams, offs, row = explain_seg(cfg, m, xin, gt, ones)
            rows.append(dict(model=lab_, tile=j, **row))
            for name, jj in sel.items():
                if jj == j:
                    shown.setdefault(name, []).append(dict(label=lab_, params=n_params(m), img=tile, gt=gt.astype(float), gt_level=0.5, cams=cams, offs=offs))
    for name, panels in shown.items():
        draw_figure(os.path.join(od, f"{key}_{name}.png"), f"{title}: {name.replace('via_rich', 'via-rich').replace('_', ' ')}",
                    CAPTION_CAM + " Explained output: summed track and via logits.", panels, "SEM, 14.65 nm/px",
                    "white line = label outline (tracks and vias, labels are about 1-3 px from the visible edge)")
    return pd.DataFrame(rows)


def run_set(cfg, key):
    title, ds, models = SETS[key]
    missing = [ck for _, ck in models if not os.path.exists(ckpt_path(ck))]
    if missing:
        print(f"[{key}] SKIPPED, missing checkpoints: {missing}")
        return
    od = os.path.join(cfg["paths"]["results"], "gradcam", key)
    os.makedirs(od, exist_ok=True)
    df = {"sim": run_sim, "carinthia": run_carinthia, "maxplanck": run_maxplanck}[ds](cfg, key, title, models, od)
    df.to_csv(os.path.join(od, f"{key}_stats.csv"), index=False)
    agg = df.groupby("model", sort=False).mean(numeric_only=True)
    agg.to_csv(os.path.join(od, f"{key}_stats_mean.csv"))
    print(f"[{key}] done -> {od}")
    print(agg[[f"{k}_edge_enrichment" for k, *_ in COLS]].round(2).to_string())


def teaser(cfg):
    """README overview figure: Conv-36, DCNv3, DCNv4 (seed 1) on one tile of each dataset; two CAM columns per dataset (decoder 1/4 block,
    last decoder layer), on the same 6-column grid as the README banner. -> results/figures/gradcam_overview.png"""
    import carinthia as ca
    import maxplanck as mp
    keep = lambda ms: [ms[1], ms[3], ms[4]]                                          # noqa: E731  conv-36, DCNv3, DCNv4
    mp_models = [(l, f"e17_maxplanck_{t}_s1") for l, t in (("Conv, w=36", "conv_w36"), ("DCNv3, w=36", "dcnv3_w36"), ("DCNv4, w=36", "dcnv4_w36"))]
    cols = []                                                                         # (dataset title, img, gt, gt_level, [cams per model])
    it = g.pick_items(cfg, offset=5)[0]                                               # same tile as the README pipeline figure
    cams = [explain_sim(cfg, m, run, it)[0] for _, m, run in [(l, *load_any(cfg, c)) for l, c in keep(e16_sim(1))]]
    cols.append(("Simulated SEM (8 frames)", it["img"], it["truth"], 0, cams))
    per = pd.read_csv(os.path.join(ROOT, "checkpoints", "carinthia_dcnv3_s1", "test_per_image.csv"))
    pits = per[per.label == 4]
    f = pits.iloc[(pits.dice - pits.dice.median()).abs().argmin()].filename
    X, Y, _ = ca.load_arrays(cfg, [f])
    valid = ca.valid_region(X.shape[-2:], cfg["carinthia"]["right_border_px"])
    x = ca._norm(torch.as_tensor(X, device=DEV)[:, None], torch.as_tensor(valid, device=DEV)[None, None].float())
    cams = [explain_seg(cfg, m, x, Y[0], valid)[0] for _, m, _r in [(l, *load_any(cfg, c)) for l, c in keep(e16_car(1))]]
    cols.append(("Carinthia-S wafer defect", X[0], Y[0].astype(float), 0.5, cams))
    img, lab = mp.arrays()
    tl = mp.grid_tiles(mp.load_split()[1]["test"], img.shape[1:])
    k, y, x0 = tl[int(np.argmax([((lab[k_, y_:y_ + mp.T, x_:x_ + mp.T] >> 1) & 1).mean() for k_, y_, x_ in tl]))]
    tile, gt = np.array(img[k, y:y + mp.T, x0:x0 + mp.T]), (np.array(lab[k, y:y + mp.T, x0:x0 + mp.T]) & 3) > 0
    xin = mp.norm(torch.as_tensor(tile, device=DEV).float()[None, None])
    ones = np.ones((mp.T, mp.T), np.float32)
    cams = [explain_seg(cfg, m, xin, gt, ones)[0] for _, m, _r in [(l, *load_any(cfg, c)) for l, c in mp_models]]
    cols.append(("Max Planck IC wiring", tile, gt.astype(float), 0.5, cams))

    names = ["Conv", "DCNv3", "DCNv4"]
    fig = plt.figure(figsize=(18, 10.4))
    gs = fig.add_gridspec(3, 6, wspace=0.05, hspace=0.06, left=0.075, right=0.995, top=0.86, bottom=0.075)
    for d, (title, im, gt_, lvl, cams) in enumerate(cols):
        for r in range(3):
            for c, key in enumerate(("dec_op1", "last")):
                a = fig.add_subplot(gs[r, 2 * d + c])
                a.imshow(im, cmap="gray")
                a.imshow(cams[r][key], cmap="jet", alpha=0.55, vmin=0, vmax=1)
                for cc in measure.find_contours(gt_, lvl):
                    a.plot(cc[:, 1], cc[:, 0], c="white", lw=0.6, alpha=0.9)
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(False)
                if r == 0:
                    a.set_title(("decoder 1/4 block", "last decoder layer")[c], fontsize=15, pad=5)
                if d == 0 and c == 0:
                    a.set_ylabel(names[r], fontsize=18, weight="bold", labelpad=10)
        fig.canvas.draw()
        b0, b1 = fig.axes[-6].get_position(), fig.axes[-1].get_position()
        x0_, x1_ = b0.x0, b1.x1
        fig.add_artist(Line2D([x0_ + 0.004, x1_ - 0.004], [0.925, 0.925], c="#333", lw=1.3))
        fig.text((x0_ + x1_) / 2, 0.935, title, ha="center", va="bottom", fontsize=18, weight="bold")
    fig.text(0.5, 0.02, "Grad-CAM, each panel scaled to its own maximum (red = most influential).  White line = true contour or annotation.  "
             "All models at width 36, seed 1.", ha="center", fontsize=14.5, color="#333")
    out = os.path.join(cfg["paths"]["results"], "figures", "gradcam_overview.png")
    fig.savefig(out, dpi=110, facecolor="white"); plt.close(fig)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run", "teaser"])
    ap.add_argument("sets", nargs="*")
    a = ap.parse_args()
    cfg = load_config()
    if a.cmd == "list":
        for k, (t, ds, ms) in SETS.items():
            print(f"{k}  [{ds}]  {t}")
            for lab, ck in ms:
                print(f"    {lab:22s} {ck:42s} {'ok' if os.path.exists(ckpt_path(ck)) else 'MISSING'}")
    elif a.cmd == "teaser":
        teaser(cfg)
    else:
        for k in (list(SETS) if a.sets == ["all"] else a.sets):
            run_set(cfg, k)
