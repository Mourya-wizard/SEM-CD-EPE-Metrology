"""Carinthia-S real SEM defect data (SEM input only, no CD/EPE truth).

  python src/carinthia.py --audit                       # M1: class counts, mask status, example grid
  python src/carinthia.py split                         # M7: stratified split -> results/carinthia/carinthia_split.json
  python src/carinthia.py train --run configs/runs/carinthia/carinthia_<op>_s<seed>.yaml   # queued GPU job
  python src/carinthia.py aggregate                     # CPU: metrics, CIs, threshold, curves, grid, summary
  python src/carinthia.py noise                         # CPU: flat-region noise vs simulator noise model

M7 model: models.UNet(in_ch=1, width=cfg.model.width[op], op) with the 1-channel output read as a binary
defect-mask logit (BCE + soft Dice). Same shell, loss, optimizer, schedule and data for every operator.
Border: the 3 px black right border is filled with the image mean and excluded from loss and metrics.
Capture score (fixed before test): max predicted foreground probability inside the valid region.
"""
import os

os.environ.setdefault("HF_HOME", "./.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "./.cache/torch")
os.environ.setdefault("XDG_CACHE_HOME", "./.cache")
os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")

import argparse
import glob
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_lithobench import load_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_JSON = os.path.join(ROOT, "results", "carinthia", "carinthia_split.json")
BASE_YAML = os.path.join(ROOT, "configs", "runs", "carinthia", "base.yaml")
DEFECT = (1, 2, 3, 4, 5)


def load_index(cfg):
    root = cfg["paths"]["carinthia"]
    df = pd.read_csv(os.path.join(root, cfg["carinthia"]["csv"]), sep=";")
    return root, df


def ckpt_root(cfg):
    return os.path.join(os.path.dirname(os.path.abspath(cfg["paths"]["data_root_abs"])), "checkpoints")


# ----------------------------------------------------------------------------------- data, split, metrics
def binarise(mask, thr=127):
    return np.asarray(mask) > thr


def valid_region(shape, border):
    v = np.ones(shape, bool)
    if border:
        v[..., -border:] = False
    return v


def fill_border(img, border):
    """Replace the black right border by the mean of the valid region (float32 copy)."""
    x = np.asarray(img, np.float32).copy()
    if border:
        x[..., -border:] = x[..., :-border].mean()
    return x


def make_split(df, seed, fracs=(0.70, 0.15, 0.15)):
    """Stratified by label. Classes with n >= 3 get >= 1 image in val and test; the rest goes to train."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for lab in sorted(df.label.unique()):
        f = df.filename[df.label == lab].sort_values().to_numpy()
        f = f[rng.permutation(len(f))]
        n = len(f)
        nv = max(1, int(round(n * fracs[1]))) if n >= 3 else 0
        nt = max(1, int(round(n * fracs[2]))) if n >= 3 else 0
        out["val"] += list(f[:nv]); out["test"] += list(f[nv:nv + nt]); out["train"] += list(f[nv + nt:])
    return out


def seg_stats(pred, gt, valid=None):
    """Per-image IoU / Dice of boolean masks (empty-empty = 1)."""
    if valid is not None:
        pred, gt = pred & valid, gt & valid
    i, ps, gs = int((pred & gt).sum()), int(pred.sum()), int(gt.sum())
    u = ps + gs - i
    return {"inter": i, "union": u, "pred_px": ps, "gt_px": gs,
            "iou": 1.0 if u == 0 else i / u, "dice": 1.0 if ps + gs == 0 else 2 * i / (ps + gs)}


def bootstrap_ci(x, n=2000, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 2
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))]


def pick_threshold(neg_scores, max_rate):
    """Smallest threshold t (flag iff score >= t) with flagged fraction of negatives <= max_rate."""
    s = np.sort(np.asarray(neg_scores, float))[::-1]
    k = int(np.floor(max_rate * len(s) + 1e-9))       # allowed number of flagged negatives
    return -np.inf if k >= len(s) else float(np.nextafter(s[k], np.inf))


def flag_rate(scores, t):
    return float(np.mean(np.asarray(scores) >= t)) if len(scores) else float("nan")


def curve(pos, neg):
    t = np.unique(np.concatenate([pos, neg, [np.inf]]))
    return t, np.array([flag_rate(pos, x) for x in t]), np.array([flag_rate(neg, x) for x in t])


def load_arrays(cfg, files, threads=3):
    root, df = load_index(cfg)
    d = df.set_index("filename")
    thr = cfg["carinthia"]["mask_threshold"]

    def rd(f):
        return (cv2.imread(os.path.join(root, d.image_path[f]), cv2.IMREAD_GRAYSCALE),
                binarise(cv2.imread(os.path.join(root, d.mask_path[f]), cv2.IMREAD_GRAYSCALE), thr))
    with ThreadPoolExecutor(threads) as ex:
        r = list(ex.map(rd, files))
    return np.stack([a for a, _ in r]), np.stack([b for _, b in r]), d.label[list(files)].to_numpy()


# ----------------------------------------------------------------------------------------------- training
def _norm(xb, v):
    """xb uint8 (B,1,H,W) -> per-image standardised float on the valid region, border set to 0."""
    x = xb.float()
    n = v.sum((2, 3), keepdim=True)
    m = (x * v).sum((2, 3), keepdim=True) / n
    s = (((x - m) * v) ** 2).sum((2, 3), keepdim=True).div(n).sqrt()
    return (x - m) / (s + 1e-6) * v


def predict(model, X, V, bs=16):
    import torch
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), bs):
            v = V.expand(len(X[i:i + bs]), -1, -1, -1)
            out.append(torch.sigmoid(model(_norm(X[i:i + bs], v)).float()) * v)
    return torch.cat(out)[:, 0]


def per_image(P, Y, labels, files, valid, thr):
    rows = []
    for k in range(len(P)):
        p = P[k].cpu().numpy()
        r = seg_stats(p > thr, Y[k, 0].cpu().numpy(), valid)
        rows.append(dict(filename=files[k], label=int(labels[k]), max_prob=float(p[valid].max()), **r))
    return pd.DataFrame(rows)


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
    out = os.path.join(ckpt_root(cfg), run["name"]); os.makedirs(out, exist_ok=True)
    _dump({"state": "running", "run": run}, os.path.join(out, "status.json"))
    torch.cuda.set_per_process_memory_fraction(cfg["queue"]["mem_fraction"])
    torch.manual_seed(seed); np.random.seed(seed)
    dev, border = "cuda", cfg["carinthia"]["right_border_px"]
    split, t0 = json.load(open(SPLIT_JSON)), time.time()
    D = {}
    for s in ("train", "val", "test"):
        X, Y, L = load_arrays(cfg, split[s])
        D[s] = (torch.from_numpy(X)[:, None].to(dev), torch.from_numpy(Y)[:, None].to(dev), L, split[s])
    print(f"loaded {[len(D[s][2]) for s in D]} images in {time.time() - t0:.0f}s", flush=True)
    H, W = D["train"][0].shape[-2:]
    valid = valid_region((H, W), border)
    V = torch.from_numpy(valid)[None, None].float().to(dev)
    model = UNet(in_ch=1, width=cfg["model"]["width"][op], op=op).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / tc["warmup"]) * 0.5 * (
        1 + np.cos(np.pi * min(i, tc["iters"]) / tc["iters"])))
    g = torch.Generator(device="cpu").manual_seed(seed)
    Xt, Yt = D["train"][:2]
    best, hist, t_train = (-1.0, -1), [], time.time()
    vdef = np.isin(D["val"][2], DEFECT)
    for it in range(1, tc["iters"] + 1):
        model.train()
        idx = torch.randint(0, len(Xt), (tc["batch"],), generator=g)
        xs, ys, vs = [], [], []
        for j, i in enumerate(idx.tolist()):                      # per-sample dihedral augmentation
            k, fl = int(torch.randint(0, 4, (1,), generator=g)), bool(torch.randint(0, 2, (1,), generator=g))
            a = [torch.rot90(t, k, (1, 2)) for t in (Xt[i], Yt[i].float(), V[0])]
            a = [t.flip(2) if fl else t for t in a]
            xs.append(a[0]); ys.append(a[1]); vs.append(a[2])
        x, y, v = torch.stack(xs), torch.stack(ys), torch.stack(vs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(_norm(x, v)).float()
        bce = (Fn.binary_cross_entropy_with_logits(logit, y, reduction="none") * v).sum() / v.sum()
        p = torch.sigmoid(logit) * v
        loss = bce + tc["dice_weight"] * (1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * v).sum() + 1))
        if not torch.isfinite(loss):
            _dump({"state": "failed", "reason": f"non-finite loss at iter {it}"}, os.path.join(out, "status.json"))
            raise SystemExit(f"NaN loss at iter {it}")
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"]); opt.step(); sched.step()
        if it % tc["log_every"] == 0:
            print(f"it {it} loss {loss.item():.4f} bce {bce.item():.4f} {(time.time() - t_train) / it:.3f}s/it", flush=True)
        if it % tc["val_every"] == 0 or it == tc["iters"]:
            model.eval()
            P = predict(model, D["val"][0], V)
            df = per_image(P, D["val"][1], D["val"][2], D["val"][3], valid, run["seg_threshold"])
            vd = float(df.dice[vdef].mean())
            hist.append({"iter": it, "val_dice_defect": vd, "loss": float(loss.item()), "t": time.time() - t_train})
            print(hist[-1], flush=True)
            if vd > best[0]:
                best = (vd, it)
                torch.save(model.state_dict(), os.path.join(out, "best.pt"))
    t_train = time.time() - t_train
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"))); model.eval()
    for s in ("val", "test"):                                       # test evaluated once, best-val checkpoint
        P = predict(model, D[s][0], V)
        per_image(P, D[s][1], D[s][2], D[s][3], valid, run["seg_threshold"]).to_csv(os.path.join(out, f"{s}_per_image.csv"), index=False)
        if s == "test":
            np.savez_compressed(os.path.join(out, "test_probs.npz"), files=np.array(D[s][3]),
                                probs=(P.cpu().numpy() * 255).round().astype(np.uint8))
    m = {"name": run["name"], "op": op, "seed": seed, "params": n_params(model), "train_s": t_train,
         "s_per_iter": t_train / tc["iters"], "total_s": time.time() - t0, "best_val_dice_defect": best[0],
         "best_iter": best[1], "history": hist, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "run": run}
    _dump(m, os.path.join(out, "metrics.json"))
    _dump({"state": "done"}, os.path.join(out, "status.json"))


# ------------------------------------------------------------------------------------------- aggregation
def _mean_ci(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else float("nan"),
            "ci95": bootstrap_ci(x) if len(x) >= 2 else [None, None]}       # n=1: no CI (rare classes 2, 5)


def aggregate(cfg, max_nuisance=0.05):
    out = os.path.join(cfg["paths"]["results"], "carinthia"); os.makedirs(out, exist_ok=True)
    runs, rows, curves = [], [], []
    for mp in sorted(glob.glob(os.path.join(ckpt_root(cfg), "carinthia_*_s*", "metrics.json"))):
        m, d = json.load(open(mp)), os.path.dirname(mp)
        va, te = pd.read_csv(os.path.join(d, "val_per_image.csv")), pd.read_csv(os.path.join(d, "test_per_image.csv"))
        td = te[te.label.isin(DEFECT)]
        t = pick_threshold(va.max_prob[va.label == 6], max_nuisance)
        pos, neg = td.max_prob.to_numpy(), te.max_prob[te.label == 6].to_numpy()
        tt, cap, nui = curve(pos, neg)
        auc = float(np.mean((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])))
        r = {k: m[k] for k in ("name", "op", "seed", "params", "train_s", "s_per_iter", "best_val_dice_defect", "best_iter", "peak_mem_gb")}
        r.update(test_dice=_mean_ci(td.dice), test_iou=_mean_ci(td.iou),
                 test_pooled_iou=float(td.inter.sum() / td.union.sum()),
                 per_class={int(c): {"dice": _mean_ci(g.dice), "iou": _mean_ci(g.iou)} for c, g in td.groupby("label")},
                 threshold_val=t, val_nuisance=flag_rate(va.max_prob[va.label == 6], t),
                 val_capture=flag_rate(va.max_prob[va.label.isin(DEFECT)], t),
                 test_capture={"k": int((pos >= t).sum()), "n": len(pos), "rate": flag_rate(pos, t), "ci95": bootstrap_ci(pos >= t)},
                 test_nuisance={"k": int((neg >= t).sum()), "n": len(neg), "rate": flag_rate(neg, t), "ci95": bootstrap_ci(neg >= t)},
                 test_roc_auc=auc)
        r["_dice"] = td.set_index("filename").dice
        runs.append(r)
        for c, pc in r["per_class"].items():
            rows.append({"op": r["op"], "seed": r["seed"], "label": c, "n_test": pc["dice"]["n"], "dice": pc["dice"]["mean"],
                         "dice_lo": pc["dice"]["ci95"][0], "dice_hi": pc["dice"]["ci95"][1], "iou": pc["iou"]["mean"],
                         "iou_lo": pc["iou"]["ci95"][0], "iou_hi": pc["iou"]["ci95"][1]})
        curves += [{"op": r["op"], "seed": r["seed"], "threshold": a, "capture": b, "nuisance": c} for a, b, c in zip(tt, cap, nui)]
    if not runs:
        raise SystemExit("no finished carinthia runs")
    for r in runs:                                   # paired image-level bootstrap: Dice(op) - Dice(conv), same seed
        ref = next((q for q in runs if q["op"] == "conv" and q["seed"] == r["seed"]), None)
        if ref is not None and r["op"] != "conv":
            d = (r["_dice"] - ref["_dice"].reindex(r["_dice"].index)).to_numpy()
            r["test_dice_minus_conv_paired"] = _mean_ci(d)
    for r in runs:
        del r["_dice"]
    pc = pd.DataFrame(rows)
    pc.to_csv(os.path.join(out, "carinthia_per_class.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(out, "carinthia_capture_nuisance_curve.csv"), index=False)
    by_op = {}
    for op in dict.fromkeys(r["op"] for r in runs):
        rr = [r for r in runs if r["op"] == op]
        f = {"dice": lambda r: r["test_dice"]["mean"], "iou": lambda r: r["test_iou"]["mean"], "capture": lambda r: r["test_capture"]["rate"],
             "nuisance": lambda r: r["test_nuisance"]["rate"], "val_dice": lambda r: r["best_val_dice_defect"], "auc": lambda r: r["test_roc_auc"]}
        by_op[op] = {"seeds": [r["seed"] for r in rr], "params": rr[0]["params"],
                     "seed_status": "SINGLE-SEED (seed 1): no seed spread" if len(rr) == 1 else f"{len(rr)} seeds", **{k: {
            "per_seed": [g(r) for r in rr], "mean": float(np.mean([g(r) for r in rr])),
            "std": float(np.std([g(r) for r in rr], ddof=1)) if len(rr) > 1 else None} for k, g in f.items()},
            "per_class_dice_per_seed": {int(c): g.sort_values("seed").dice.tolist() for c, g in pc[pc.op == op].groupby("label")}}
    split = json.load(open(SPLIT_JSON))
    summ = {"note": "Carinthia-S real SEM, defect segmentation/detection only; no CD/EPE truth. Test evaluated once per run "
            "(best-val checkpoint). Capture score = max predicted fg probability; threshold = smallest giving <= "
            f"{max_nuisance:.0%} flagged val class-6 images.", "split_counts": split["counts"],
            "seeds_note": "All four variants have seed 1 only (SINGLE-SEED; no seed spread). Seeds 2-3 for conv and dcnv3 "
            "(validation-selected, carinthia_seed_selection.json) were planned but paused for time." if all(
                len(b["seeds"]) == 1 for b in by_op.values()) else "see by_op[*].seed_status",
            "base": yaml.safe_load(open(BASE_YAML)), "runs": runs, "by_op": by_op,
            "rare_classes": {int(c): {"split_counts": {k: split["counts"][k].get(str(c), split["counts"][k].get(c, 0)) for k in split["counts"]},
                                      "test_dice_by_op_seed": {f"{r.op}_s{r.seed}": r.dice for r in g.itertuples()},
                                      "note": "single test image: no CI; values are one image, not a class estimate"}
                             for c, g in pc[pc.label.isin((2, 5))].groupby("label")}}
    for k in ("seed_selection", "noise"):                             # seed_selection: which ops got seeds 2-3 (val-only rule)
        if os.path.exists(os.path.join(out, f"carinthia_{k}.json")):
            summ[k] = json.load(open(os.path.join(out, f"carinthia_{k}.json")))
    _dump(summ, os.path.join(out, "carinthia_summary.json"))
    _plots(cfg, runs, pd.DataFrame(curves), out)
    for op, b in by_op.items():
        print(op, {k: (round(b[k]["mean"], 4), b[k]["per_seed"]) for k in ("dice", "iou", "capture", "nuisance")})


def _plots(cfg, runs, cv, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for (op, s), g in cv.groupby(["op", "seed"]):
        ax.plot(g.nuisance, g.capture, lw=1, label=f"{op} s{s}")
    ax.set(xlabel="nuisance rate (test class 6 flagged)", ylabel="capture rate (test defects flagged)", xlim=(-0.02, 1), ylim=(0, 1.02))
    ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(out, "carinthia_capture_nuisance.png"), dpi=110); plt.close(fig)
    best = max(runs, key=lambda r: r["best_val_dice_defect"])          # chosen on validation
    d = os.path.join(ckpt_root(cfg), best["name"])
    te, z = pd.read_csv(os.path.join(d, "test_per_image.csv")), np.load(os.path.join(d, "test_probs.npz"))
    pm = dict(zip(z["files"], z["probs"]))
    td = te[te.label.isin(DEFECT)].sort_values("dice")
    pick = [("success", td.tail(4).iloc[::-1]), ("failure", td.head(4)),
            ("rare", te[te.label.isin((2, 5))].head(2)), ("class 6", te[te.label == 6].sort_values("max_prob").tail(2).iloc[::-1])]
    items = [(tag, r) for tag, g in pick for _, r in g.iterrows()]
    X, Y, _ = load_arrays(cfg, [r.filename for _, r in items])
    nr = (len(items) + 1) // 2
    fig, ax = plt.subplots(nr, 6, figsize=(10, 1.75 * nr))
    for k, (tag, r) in enumerate(items):
        a = ax[k // 2, 3 * (k % 2):3 * (k % 2) + 3]
        a[0].imshow(X[k], cmap="gray"); a[1].imshow(Y[k], cmap="gray"); a[2].imshow(pm[r.filename], cmap="magma", vmin=0, vmax=255)
        a[0].set_title(f"{tag} c{r.label}", fontsize=7); a[1].set_title("truth", fontsize=7)
        a[2].set_title(f"pred Dice {r.dice:.2f} max {r.max_prob:.2f}", fontsize=7)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"{best['name']} (best val Dice), test images", fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out, "carinthia_grid.png"), dpi=90); plt.close(fig)


# ------------------------------------------------------------------------------------------------- noise
def noise(cfg, n_img=300, win=16, keep=0.1, seed=0):
    """Pixel noise from low-gradient windows (plane-fit residual std) on train-split images. Indicative only."""
    from scipy import ndimage
    split = json.load(open(SPLIT_JSON))
    files = list(np.random.default_rng(seed).choice(split["train"], n_img, replace=False))
    X, Y, L = load_arrays(cfg, files)
    b = cfg["carinthia"]["right_border_px"]
    yy, xx = np.mgrid[:win, :win]
    A = np.c_[np.ones(win * win), yy.ravel(), xx.ravel()]
    per = []
    for img, m in zip(X, Y):
        x = img[:, :-b].astype(np.float64)
        gm = ndimage.gaussian_gradient_magnitude(x, 3.0)
        bad = ndimage.binary_dilation(m[:, :-b], iterations=8)
        cand = []
        for r in range(0, x.shape[0] - win + 1, win):
            for c in range(0, x.shape[1] - win + 1, win):
                if not bad[r:r + win, c:c + win].any():
                    cand.append((gm[r:r + win, c:c + win].mean(), r, c))
        cand.sort()
        sd, mu, rho = [], [], []
        for _, r, c in cand[:max(1, int(len(cand) * keep))]:
            w = x[r:r + win, c:c + win].ravel()
            res = w - A @ np.linalg.lstsq(A, w, rcond=None)[0]
            res2 = res.reshape(win, win)
            sd.append(res.std() * np.sqrt(win * win / (win * win - 3))); mu.append(w.mean())
            rho.append(float((res2[:, 1:] * res2[:, :-1]).mean() / res2.var()))
        per.append({"std": float(np.median(sd)), "mean": float(np.median(mu)), "rel": float(np.median(np.array(sd) / np.array(mu))),
                    "lag1_corr": float(np.median(rho))})
    p = pd.DataFrame(per)
    q = lambda s: {k: float(np.quantile(s, v)) for k, v in (("p10", .1), ("median", .5), ("p90", .9))}
    sc = cfg["sim"]
    e, sg = sc["electrons_per_frame"], sc["detector_sigma"]
    ys = [0.15, 0.35, 0.5, 0.85]
    sim = {f"y{y}": {f"f{f}": float(np.sqrt(y / (e * f) + sg ** 2) / y) for f in sc["frames"]} for y in ys}
    rmed = float(p.rel.median())
    eqf = {f"y{y}": (float(y / (e * (rmed ** 2 * y ** 2 - sg ** 2))) if rmed * y > sg else None) for y in ys}
    res = {"label": "INDICATIVE ONLY: JPEG, unknown source processing/gain/offset; real rel. noise = gray-level std / gray-level mean",
           "method": f"{n_img} train images; {win}x{win} windows away from (dilated) defect masks; lowest {keep:.0%} by Gaussian(3px) "
                     "gradient magnitude per image; plane-fit residual std; per-image medians",
           "real_std_gray": q(p["std"]), "real_mean_gray": q(p["mean"]), "real_rel_noise": q(p.rel), "real_lag1_corr": q(p.lag1_corr),
           "sim_model": f"Poisson({e} e/px/frame x yield x frames)/(...) + N(0,{sg}); rel = sqrt(y/(3f)+{sg}^2)/y; white (lag-1 corr 0)",
           "sim_rel_noise": sim, "equiv_frames_at_median_real_rel": eqf}
    _dump(res, os.path.join(cfg["paths"]["results"], "carinthia", "carinthia_noise.json"))
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------- audit
def audit(cfg):
    root, df = load_index(cfg)
    thr = cfg["carinthia"]["mask_threshold"]
    rows = []
    for lab, g in df.groupby("label"):
        fg, empty, nonbinary, shapes = [], 0, 0, set()
        for p in g.mask_path:
            m = cv2.imread(os.path.join(root, p), cv2.IMREAD_GRAYSCALE)
            shapes.add(m.shape)
            nonbinary += int(not np.isin(m, (0, 255)).all())
            f = float((m > thr).mean())
            fg.append(f)
            empty += int(f == 0)
        rows.append({"label": int(lab), "n_images": len(g), "empty_masks": empty, "nonbinary_masks": nonbinary,
                     "fg_frac_mean": float(np.mean(fg)), "shapes": sorted(map(list, shapes))})
        print(rows[-1])
    out = os.path.join(cfg["paths"]["results"], "data_audit")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "carinthia_audit.json"), "w") as f:
        json.dump({"total": len(df), "classes": rows}, f, indent=1)
    _grid(root, df, os.path.join(out, "carinthia_grid.png"))


def _grid(root, df, out_png, per_class=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = sorted(df.label.unique())
    fig, ax = plt.subplots(len(labs), 2 * per_class, figsize=(2 * per_class * 1.6, len(labs) * 1.7))
    for r, lab in enumerate(labs):
        g = df[df.label == lab].sample(n=min(per_class, (df.label == lab).sum()), random_state=0)
        for j in range(per_class):
            for c in (0, 1):
                a = ax[r, 2 * j + c]
                a.axis("off")
                if j < len(g):
                    p = g.iloc[j]["image_path" if c == 0 else "mask_path"]
                    a.imshow(cv2.imread(os.path.join(root, p), cv2.IMREAD_GRAYSCALE), cmap="gray")
        ax[r, 0].set_title(f"class {lab}", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def write_split(cfg, seed):
    root, df = load_index(cfg)
    s = make_split(df, seed)
    lab = df.set_index("filename").label
    counts = {k: {int(c): int(n) for c, n in lab[v].value_counts().sort_index().items()} for k, v in s.items()}
    os.makedirs(os.path.dirname(SPLIT_JSON), exist_ok=True)
    c6 = [f for f, p in zip(df.filename[df.label == 6], df.mask_path[df.label == 6])
          if binarise(cv2.imread(os.path.join(root, p), cv2.IMREAD_GRAYSCALE)).any()]
    _dump({"seed": seed, "fracs": [0.7, 0.15, 0.15], "rule": "per class: n_val=n_test=max(1,round(0.15n)) if n>=3; rest train",
           "class6_nonempty_masks": {f: [k for k in s if f in s[k]][0] for f in c6}, "counts": counts, **s}, SPLIT_JSON)
    print(counts, "class-6 non-empty masks:", c6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["split", "train", "aggregate", "noise"])
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    cv2.setNumThreads(min(a.threads, 4))
    c = load_config()
    if a.audit:
        audit(c)
    elif a.cmd == "split":
        write_split(c, a.seed)
    elif a.cmd == "train":
        train(c, a.run)
    elif a.cmd == "aggregate":
        aggregate(c)
    elif a.cmd == "noise":
        noise(c)
