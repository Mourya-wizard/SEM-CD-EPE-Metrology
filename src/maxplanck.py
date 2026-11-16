"""E17: real-SEM evaluation on the Max Planck IC metal-2 set (SEM input only; labels = track polygons + via circles).

  python tools/maxplanck_prepare.py                                   # once: arrays + split (data/maxplanck/prepared, results/maxplanck)
  python src/maxplanck.py train --run configs/runs/maxplanck/<name>.yaml    # queued GPU job (one model, one seed)
  python src/maxplanck.py classical                                   # CPU: val-tuned global threshold baseline on val/test
  python src/maxplanck.py zeroshot                                    # GPU: simulated-SEM TSDF models applied as they are (x2 / x4 up-sampling)
  python src/maxplanck.py aggregate                                   # CPU: tables, gates -> results/maxplanck/e17-*

Same shell, loss, optimiser and schedule as the Carinthia-S runs (src/carinthia.py): models.UNet(in_ch=1, width, op), 1-channel logit,
BCE + soft Dice, 3000 iterations, batch 8, per-sample dihedral augmentation, per-crop standardisation, best-val-Dice checkpoint.
Training samples random 512x512 crops of the central 80% of the training images; evaluation uses the non-overlapping 512x512 grid of the
central region of val / test images (30 tiles per image). The test split is evaluated once, with the best-validation checkpoint.
Metrics (px and nm, 14.65 nm/px): Dice, boundary F1 at 2 px, symmetric boundary distance, per-component track width (2 x median distance-
transform value on the skeleton) error against the label, failure (component missed or |width error| > 3 px), and the 3-sigma spread of the
width over repeated added-noise draws (precision).
LIMIT: label edges sit about 1-3 px (15-45 nm) from the visible edge of this blurred 14.65 nm/px data, so absolute nm accuracy is NOT verifiable;
only differences between models on the same labels are meaningful.
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_lithobench import load_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREP = os.path.join(ROOT, "data", "maxplanck", "prepared")
SPLIT = os.path.join(ROOT, "results", "maxplanck", "maxplanck_split.json")
BASE_YAML = os.path.join(ROOT, "configs", "runs", "maxplanck", "base.yaml")
NM = 14.65
T = 512
WORKERS = 12                                                  # CPU cap on this machine is 15 processes
GROUPS = {"conv_w32": "e17_maxplanck_conv_w32", "conv_w36": "e17_maxplanck_conv_w36", "dcnv2_w36": "e17_maxplanck_dcnv2_w36",
          "dcnv3_w36": "e17_maxplanck_dcnv3_w36", "dcnv4_w36": "e17_maxplanck_dcnv4_w36"}


# ------------------------------------------------------------------------------------------------------------- data
def load_split():
    sp = json.load(open(SPLIT))
    order = {i: k for k, i in enumerate(sp["index_order"])}
    return sp, {s: [order[i] for i in sp[s]] for s in ("train", "val", "test")}


def arrays():
    return np.load(os.path.join(PREP, "img.npy"), mmap_mode="r"), np.load(os.path.join(PREP, "lab.npy"), mmap_mode="r")


def grid_tiles(idx, shape):
    """Non-overlapping 512 grid of the central region: list of (image index, y, x)."""
    H, W = shape
    return [(k, y, x) for k in idx for y in range(0, H - T + 1, T) for x in range(0, W - T + 1, T)]


def norm(x):
    """x float (B,1,H,W) -> per-sample standardised."""
    m = x.mean((2, 3), keepdim=True)
    s = x.std((2, 3), keepdim=True)
    return (x - m) / (s + 1e-6)


def predict(model, X, bs=16):
    """X uint8 tensor (N,512,512) on GPU -> foreground probability (N,512,512) on GPU."""
    import torch
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(norm(X[i:i + bs, None].float())).float())[:, 0])
    return torch.cat(out)


def gather(img, lab, tiles):
    return (np.stack([img[k, y:y + T, x:x + T] for k, y, x in tiles]), np.stack([(lab[k, y:y + T, x:x + T] & 3) > 0 for k, y, x in tiles]))


# ----------------------------------------------------------------------------------------------------------- metrics
def _skel(mask):
    return ndimage.distance_transform_edt(mask), skeletonize(mask)


def width_table(gt, pr, margin=10, min_area=300, min_pts=10):
    """Per GT component: label width and predicted width in px (nan if the component is missed)."""
    inner = np.zeros_like(gt)
    inner[margin:-margin, margin:-margin] = True
    e_g, s_g = _skel(gt)
    e_p, s_p = _skel(pr) if pr.any() else (np.zeros(gt.shape), np.zeros(gt.shape, bool))
    lab, n = ndimage.label(gt, structure=np.ones((3, 3)))
    rows = []
    for c in range(1, n + 1):
        reg = lab == c
        if reg.sum() < min_area:
            continue
        pg = s_g & reg & inner
        if pg.sum() < min_pts:
            continue
        near = ndimage.binary_dilation(reg, iterations=6)
        pp = s_p & near & inner
        rows.append((int(reg.sum()), 2 * float(np.median(e_g[pg])), 2 * float(np.median(e_p[pp])) if pp.sum() >= min_pts else float("nan")))
    return rows


def tile_metrics(args):
    gt, pr = args
    i, ps, gs = int((pr & gt).sum()), int(pr.sum()), int(gt.sum())
    r = {"dice": 1.0 if ps + gs == 0 else 2 * i / (ps + gs), "iou": 1.0 if ps + gs - i == 0 else i / (ps + gs - i)}
    bg, bp = gt ^ ndimage.binary_erosion(gt), pr ^ ndimage.binary_erosion(pr)
    if bg.any() and bp.any():
        d_pg = ndimage.distance_transform_edt(~bg)[bp]
        d_gp = ndimage.distance_transform_edt(~bp)[bg]
        p, q = float((d_pg <= 2).mean()), float((d_gp <= 2).mean())
        r.update(bf1_2px=0.0 if p + q == 0 else 2 * p * q / (p + q), edge_dist_px=float((d_pg.mean() + d_gp.mean()) / 2))
    else:
        r.update(bf1_2px=float("nan"), edge_dist_px=float("nan"))
    return r, width_table(gt, pr)


def evaluate_masks(G, P, pool=None):
    """G, P bool arrays (N,512,512) -> per-tile DataFrame, per-component DataFrame."""
    jobs = list(zip(G, P))
    res = list(pool.map(tile_metrics, jobs, chunksize=8)) if pool else [tile_metrics(j) for j in jobs]
    tiles = pd.DataFrame([r for r, _ in res])
    comps = pd.DataFrame([dict(tile=k, area=a, w_gt_px=wg, w_pred_px=wp) for k, (_, rows) in enumerate(res) for a, wg, wp in rows])
    return tiles, comps


def summarise(tiles, comps, fail_px=3.0):
    err = (comps.w_pred_px - comps.w_gt_px) * NM
    miss = comps.w_pred_px.isna()
    fail = miss | ((comps.w_pred_px - comps.w_gt_px).abs() > fail_px)
    ok = err[~miss]
    return {"dice_mean": float(tiles.dice.mean()), "bf1_2px": float(tiles.bf1_2px.mean()),
            "edge_dist_px": float(tiles.edge_dist_px.mean()), "edge_dist_nm": float(tiles.edge_dist_px.mean() * NM),
            "width_bias_nm": float(ok.mean()), "width_mae_nm": float(ok.abs().mean()), "width_3sigma_nm": float(3 * ok.std()),
            "width_fail_rate": float(fail.mean()), "n_components": int(len(comps))}


def noise_precision(model, img, lab, tiles, sigmas=(10, 20), draws=8, seed=0, pool=None):
    """3-sigma spread (nm) of the per-component width over `draws` independent Gaussian noise draws (sigma in grey levels)."""
    import torch
    X, G = gather(img, lab, tiles)
    out, dice = {}, {}
    rng = np.random.default_rng(seed)
    for s in sigmas:
        W, D = [], []
        for _ in range(draws):
            Xn = np.clip(X.astype(np.float32) + rng.normal(0, s, X.shape), 0, 255).astype(np.uint8)
            P = (predict(model, torch.as_tensor(Xn, device="cuda")) > 0.5).cpu().numpy()
            t, c = evaluate_masks(G, P, pool)
            W.append(c.set_index(["tile", "area"]).w_pred_px.rename(len(W)))
            D.append(float(t.dice.mean()))
        w = pd.concat(W, axis=1)
        ok = w.notna().sum(1) >= int(0.75 * draws)
        out[f"prec3s_nm_noise{s}"] = float((3 * w[ok].std(1, ddof=1) * NM).mean())
        dice[f"dice_noise{s}"] = float(np.mean(D))
    return {**out, **dice}


# ------------------------------------------------------------------------------------------------------------ train
def _dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def train(cfg, run_path):
    import torch
    import torch.nn.functional as Fn
    from models import UNet, n_params
    run = dict(yaml.safe_load(open(BASE_YAML)), **yaml.safe_load(open(run_path)))
    tc, op, seed = run["train"], run["op"], int(run["seed"])
    out = os.path.join(ROOT, "checkpoints", run["name"])
    os.makedirs(out, exist_ok=True)
    _dump({"state": "running", "run": run}, os.path.join(out, "status.json"))
    torch.cuda.set_per_process_memory_fraction(cfg["queue"]["mem_fraction"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev, t0 = "cuda", time.time()
    img, lab = arrays()
    _, idx = load_split()
    Xtr = torch.from_numpy(np.ascontiguousarray(img[idx["train"]])).to(dev)
    Ytr = torch.from_numpy(np.ascontiguousarray((lab[idx["train"]] & 3) > 0)).to(dev)
    Hh, Ww = Xtr.shape[-2:]
    vt = grid_tiles(idx["val"], (Hh, Ww))
    Xv, Gv = gather(img, lab, vt)
    Xv_t = torch.from_numpy(Xv).to(dev)
    print(f"loaded train {tuple(Xtr.shape)}, val tiles {len(vt)} in {time.time() - t0:.0f}s", flush=True)
    model = UNet(in_ch=1, width=run.get("width") or cfg["model"]["width"][op], op=op).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / tc["warmup"]) * 0.5 * (1 + np.cos(np.pi * min(i, tc["iters"]) / tc["iters"])))
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, hist, t_train = (-1.0, -1), [], time.time()
    Gv_t = torch.from_numpy(Gv).to(dev)
    for it in range(1, tc["iters"] + 1):
        model.train()
        xs, ys = [], []
        for _ in range(tc["batch"]):
            k = int(torch.randint(0, len(Xtr), (1,), generator=g))
            y0, x0 = int(torch.randint(0, Hh - T + 1, (1,), generator=g)), int(torch.randint(0, Ww - T + 1, (1,), generator=g))
            a, b = Xtr[k, y0:y0 + T, x0:x0 + T][None], Ytr[k, y0:y0 + T, x0:x0 + T][None].float()
            r, fl = int(torch.randint(0, 4, (1,), generator=g)), bool(torch.randint(0, 2, (1,), generator=g))
            a, b = torch.rot90(a, r, (1, 2)), torch.rot90(b, r, (1, 2))
            xs.append(a.flip(2) if fl else a)
            ys.append(b.flip(2) if fl else b)
        x, y = torch.stack(xs).float(), torch.stack(ys)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(norm(x)).float()
        bce = Fn.binary_cross_entropy_with_logits(logit, y)
        p = torch.sigmoid(logit)
        loss = bce + tc["dice_weight"] * (1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1))
        if not torch.isfinite(loss):
            _dump({"state": "failed", "reason": f"non-finite loss at iter {it}"}, os.path.join(out, "status.json"))
            raise SystemExit(3)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        sched.step()
        if it % tc["log_every"] == 0:
            print(f"it {it} loss {loss.item():.4f} bce {bce.item():.4f} {(time.time() - t_train) / it:.3f}s/it", flush=True)
        if it % tc["val_every"] == 0 or it == tc["iters"]:
            model.eval()
            pv = predict(model, Xv_t) > 0.5
            inter = (pv & Gv_t).sum((1, 2)).float()
            vd = float((2 * inter / (pv.sum((1, 2)) + Gv_t.sum((1, 2))).clamp(min=1)).mean())
            hist.append({"iter": it, "val_dice": vd, "loss": float(loss.item()), "t": time.time() - t_train})
            print(hist[-1], flush=True)
            if vd > best[0]:
                best = (vd, it)
                torch.save(model.state_dict(), os.path.join(out, "best.pt"))
    t_train = time.time() - t_train
    model.load_state_dict(torch.load(os.path.join(out, "best.pt")))
    model.eval()
    del Xtr, Ytr
    res = {}
    with ProcessPoolExecutor(WORKERS) as pool:
        for s in ("val", "test"):
            tl = grid_tiles(idx[s], (Hh, Ww))
            X, G = gather(img, lab, tl)
            P = (predict(model, torch.from_numpy(X).to(dev)) > 0.5).cpu().numpy()
            tiles, comps = evaluate_masks(G, P, pool)
            tiles.to_csv(os.path.join(out, f"{s}_tiles.csv"), index=False)
            comps.to_csv(os.path.join(out, f"{s}_components.csv"), index=False)
            res[s] = summarise(tiles, comps)
        rs = np.random.default_rng(0)
        sub = [grid_tiles(idx["test"], (Hh, Ww))[j] for j in sorted(rs.choice(len(grid_tiles(idx["test"], (Hh, Ww))), 120, replace=False))]
        res["test_noise"] = noise_precision(model, img, lab, sub, pool=pool)
    m = {"name": run["name"], "op": op, "seed": seed, "params": n_params(model), "train_s": t_train, "best_val_dice": best[0], "best_iter": best[1],
         "history": hist, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "total_s": time.time() - t0, "results": res, "run": run}
    _dump(m, os.path.join(out, "metrics.json"))
    _dump({"state": "done"}, os.path.join(out, "status.json"))
    print(json.dumps(res, indent=1), flush=True)


# --------------------------------------------------------------------------------------- classical / zero-shot / table
def classical(cfg):
    """Global threshold on the per-crop standardised, Gaussian-smoothed image; the threshold is tuned on VAL only, then applied to val and test."""
    img, lab = arrays()
    _, idx = load_split()
    H, W = img.shape[1:]
    out = {}
    with ProcessPoolExecutor(WORKERS) as pool:
        data = {}
        for s in ("val", "test"):
            X, G = gather(img, lab, grid_tiles(idx[s], (H, W)))
            Z = np.stack([ndimage.gaussian_filter(x.astype(np.float32), 1.5) for x in X])
            Z = (Z - Z.mean((1, 2), keepdims=True)) / (Z.std((1, 2), keepdims=True) + 1e-6)
            data[s] = (Z, G)
        sub = np.random.default_rng(0).choice(len(data["val"][0]), 300, replace=False)
        grid = np.arange(-0.5, 1.51, 0.25)
        dices = [float(np.mean([(2 * ((z > t) & g).sum()) / max(((z > t).sum() + g.sum()), 1) for z, g in zip(data["val"][0][sub], data["val"][1][sub])])) for t in grid]
        t = float(grid[int(np.argmax(dices))])
        for s in ("val", "test"):
            Z, G = data[s]
            tiles, comps = evaluate_masks(G, Z > t, pool)
            out[s] = summarise(tiles, comps)
        out["threshold_std_units"] = t
        out["threshold_grid_val_dice"] = dict(zip(map(float, grid), dices))
    os.makedirs(os.path.join(ROOT, "results", "maxplanck"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "maxplanck", "e17-classical.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "threshold_grid_val_dice"}, indent=1))


def zeroshot(cfg, seeds=(1, 2)):
    """Simulated-SEM models (SEM-only TSDF, trained at 1 nm/px) applied to the real tiles after nearest-free bilinear up-sampling by x2 / x4;
    foreground = TSDF > 0 (inside positive). No training on this data."""
    import torch
    import torch.nn.functional as Fn
    from models import UNet
    img, lab = arrays()
    _, idx = load_split()
    H, W = img.shape[1:]
    tl = grid_tiles(idx["test"], (H, W))
    tl = [tl[j] for j in sorted(np.random.default_rng(0).choice(len(tl), 300, replace=False))]
    X, G = gather(img, lab, tl)
    out = {}
    with ProcessPoolExecutor(WORKERS) as pool:
        for seed in seeds:
            runs = {"conv_w32": f"e16_conv_w32_semonly_s{seed}", "conv_w36": f"e16_conv_w36_semonly_s{seed}", "dcnv2_w36": f"e16_dcnv2_w36_semonly_s{seed}",
                    "dcnv3_w36": f"cadtest_dcnv3_semonly_s{seed}", "dcnv4_w36": f"e16_dcnv4_w36_semonly_s{seed}"}
            for grp, name in runs.items():
                ck = torch.load(os.path.join(ROOT, "checkpoints", name, "best.pt"), map_location="cuda", weights_only=False)
                from models import build_for_run
                m = build_for_run(cfg["model"], ck["run"]).cuda().eval()
                m.load_state_dict(ck["model"])
                for sc in (2, 4):
                    c = T // sc
                    Xc = torch.as_tensor(X[:, :c, :c], device="cuda").float()[:, None]
                    Xu = Fn.interpolate(Xc, scale_factor=sc, mode="bilinear", align_corners=False)
                    with torch.no_grad():
                        t = torch.cat([m(norm(Xu[i:i + 8])).float() for i in range(0, len(Xu), 8)])
                    P = Fn.interpolate((t > 0).float(), size=(c, c), mode="area")[:, 0] > 0.5
                    Gc = G[:, :c, :c]
                    Pn = P.cpu().numpy()
                    i = (Pn & Gc).sum((1, 2))
                    d = 2 * i / np.maximum(Pn.sum((1, 2)) + Gc.sum((1, 2)), 1)
                    out[f"{grp}_x{sc}_s{seed}"] = {"dice_mean": float(d.mean()), "pred_fg_frac": float(Pn.mean()), "gt_fg_frac": float(Gc.mean())}
    json.dump(out, open(os.path.join(ROOT, "results", "maxplanck", "e17-zeroshot.json"), "w"), indent=1)


def aggregate(cfg):
    rows = []
    for g, stem in GROUPS.items():
        for s in (1, 2, 3):
            p = os.path.join(ROOT, "checkpoints", f"{stem}_s{s}", "metrics.json")
            if os.path.exists(p):
                m = json.load(open(p))
                r = m["results"]
                rows.append(dict(group=g, seed=s, params=m["params"], best_iter=m["best_iter"], **{f"val_{k}": v for k, v in r["val"].items()},
                                 **{f"test_{k}": v for k, v in r["test"].items()}, **{f"noise_{k}": v for k, v in r["test_noise"].items()}))
    df = pd.DataFrame(rows)
    od = os.path.join(ROOT, "results", "maxplanck")
    df.to_csv(os.path.join(od, "e17-per-seed.csv"), index=False)
    cols = [c for c in df.columns if c not in ("group", "seed")]
    mean = df.groupby("group", sort=False)[cols].mean()
    sd = df.groupby("group", sort=False)[cols].std()
    mean.to_csv(os.path.join(od, "e17-summary-mean.csv"))
    sd.to_csv(os.path.join(od, "e17-summary-sd.csv"))
    gates = {}
    if "conv_w36" in set(df.group):
        c = df[df.group == "conv_w36"].set_index("seed")
        for g in ("dcnv2_w36", "dcnv3_w36", "dcnv4_w36"):
            if g not in set(df.group):
                continue
            d = df[df.group == g].set_index("seed")
            sh = sorted(set(d.index) & set(c.index))
            if len(sh) < 3:
                continue
            dd, cc = d.loc[sh], c.loc[sh]
            gates[g] = {"G1_val_dice_mean_ge_conv_plus_0.003_and_every_seed": bool(dd.val_dice_mean.mean() >= cc.val_dice_mean.mean() + 0.003 and (dd.val_dice_mean > cc.val_dice_mean).all()),
                        "G2_bf1_not_worse_0.003": bool(dd.val_bf1_2px.mean() >= cc.val_bf1_2px.mean() - 0.003),
                        "G3_width_mae_not_worse_0.5nm": bool(dd.val_width_mae_nm.mean() <= cc.val_width_mae_nm.mean() + 0.5),
                        "G4_fail_rate_not_worse_0.5pt": bool(dd.val_width_fail_rate.mean() <= cc.val_width_fail_rate.mean() + 0.005)}
            gates[g]["pass_all"] = bool(all(gates[g].values()))
    json.dump(gates, open(os.path.join(od, "e17-gates.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(mean.round(4).T.to_string())
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "classical", "zeroshot", "aggregate"])
    ap.add_argument("--run")
    a = ap.parse_args()
    cfg = load_config()
    {"train": lambda: train(cfg, a.run), "classical": lambda: classical(cfg), "zeroshot": lambda: zeroshot(cfg), "aggregate": lambda: aggregate(cfg)}[a.cmd]()
