"""LithoBench loading and tile-level splits.

Conventions (verified in M1):
  * images are 2048x2048 uint8 PNG, 1 nm per pixel, row = y (down), col = x.
  * target/*.png  : design raster, 255 = feature.
  * litho/*.png   : aerial intensity * 255 (linear). printed == aerial > 0.225.
  * printed/*.png : binarized resist (StdMetal stores it as 3-channel; read as grayscale).
  * resist/*.png  : sigmoid(50 * (aerial - 0.225)) * 255, smooth near the edge.
CLI: `python src/data_lithobench.py --audit` prints the M1 checkpoint and writes results/splits.json.
"""
import os

os.environ.setdefault("HF_HOME", "./.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "./.cache/torch")
os.environ.setdefault("XDG_CACHE_HOME", "./.cache")
os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")

import argparse
import glob
import hashlib
import json

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINDS = ("target", "litho", "printed", "resist")


def load_config(path=None):
    with open(path or os.path.join(ROOT, "configs", "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    data_root = os.environ.get("PROJECT_DATA_ROOT")
    for k in ("lithobench", "carinthia"):
        p = cfg["paths"][k]
        if data_root:
            p = os.path.join(data_root, os.path.relpath(p, "data"))
        cfg["paths"][k] = p if os.path.isabs(p) else os.path.join(ROOT, p)
    cfg["paths"]["results"] = os.path.join(ROOT, cfg["paths"]["results"])
    cfg["paths"]["data_root_abs"] = data_root or os.path.join(ROOT, cfg["paths"]["data_root"])
    return cfg


def list_tiles(lb_root, set_name):
    """Tile IDs that have target, litho and printed images (intersection)."""
    ids = None
    for k in ("target", "litho", "printed"):
        s = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(lb_root, set_name, k, "*.png"))}
        ids = s if ids is None else ids & s
    return sorted(ids)


def load_tile(lb_root, set_name, tile_id, kinds=KINDS):
    """Returns dict kind -> float32 array in [0, 1], shape (2048, 2048)."""
    out = {}
    for k in kinds:
        im = cv2.imread(os.path.join(lb_root, set_name, k, tile_id + ".png"), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"{set_name}/{k}/{tile_id}.png")
        out[k] = im.astype(np.float32) / 255.0
    return out


def _hash_frac(key, seed):
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def make_splits(cfg):
    """Deterministic split by tile: hash(seed, set/tile) < val_fraction -> val."""
    lc, lb = cfg["lithobench"], cfg["paths"]["lithobench"]
    splits = {"train": {}, "val": {}, "test": {}, "tuning": {}}
    for s in lc["train_sets"]:
        ids = list_tiles(lb, s)
        splits["val"][s] = [t for t in ids if _hash_frac(f"{s}/{t}", lc["split_seed"]) < lc["val_fraction"]]
        v = set(splits["val"][s])
        splits["train"][s] = [t for t in ids if t not in v]
    for s in lc["test_sets"]:
        splits["test"][s] = list_tiles(lb, s)
    for s in lc["tuning_sets"]:
        splits["tuning"][s] = list_tiles(lb, s)
    return splits


def _grid(cfg, splits, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lb, lo, hi = cfg["paths"]["lithobench"], cfg["lithobench"]["active_lo_px"], cfg["lithobench"]["active_hi_px"]
    picks = [("MetalSet", splits["train"]["MetalSet"][0]), ("ViaSet", splits["train"]["ViaSet"][0]),
             ("StdMetal", splits["test"]["StdMetal"][0]), ("StdContactTest", splits["test"]["StdContactTest"][0])]
    fig, ax = plt.subplots(len(picks), 4, figsize=(13, 3.3 * len(picks)))
    for r, (s, t) in enumerate(picks):
        d = load_tile(lb, s, t)
        for c, k in enumerate(KINDS):
            ax[r, c].imshow(d[k][lo:hi, lo:hi], cmap="gray")
            ax[r, c].set_title(f"{s}/{t}\n{k}", fontsize=8)
            ax[r, c].axis("off")
    fig.suptitle("LithoBench active region (1280x1280 px = 1280 nm); aerial contour of printed at 0.225")
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def audit(cfg):
    lb, thr = cfg["paths"]["lithobench"], cfg["lithobench"]["aerial_threshold"]
    res_dir = os.path.join(cfg["paths"]["results"], "data_audit")
    os.makedirs(res_dir, exist_ok=True)
    report = {"sets": {}}
    for s in ("MetalSet", "ViaSet", "StdMetal", "StdContact", "StdContactTest"):
        counts = {k: len(glob.glob(os.path.join(lb, s, k, "*.png"))) for k in KINDS}
        ids = list_tiles(lb, s)
        rng = np.random.default_rng(0)
        agree, fg = [], []
        for t in rng.choice(ids, size=min(16, len(ids)), replace=False):
            d = load_tile(lb, s, t, ("litho", "printed"))
            agree.append(float(((d["litho"] > thr) == (d["printed"] > 0.5)).mean()))
            fg.append(float((d["printed"] > 0.5).mean()))
        report["sets"][s] = {"files": counts, "usable_tiles": len(ids),
                             "thr_agreement_min": min(agree), "thr_agreement_mean": float(np.mean(agree)),
                             "printed_fg_mean": float(np.mean(fg))}
        print(f"{s:15s} files={counts} usable={len(ids)} agree(aerial>{thr}, printed) "
              f"min={min(agree):.5f} mean={np.mean(agree):.5f} printed_fg={np.mean(fg):.4f}")
    splits = make_splits(cfg)
    with open(os.path.join(cfg["paths"]["results"], "splits.json"), "w") as f:
        json.dump(splits, f)
    sizes = {k: {s: len(v) for s, v in d.items()} for k, d in splits.items()}
    report["split_sizes"] = sizes
    print("split sizes:", sizes)
    _grid(cfg, splits, os.path.join(res_dir, "lithobench_grid.png"))
    with open(os.path.join(res_dir, "lithobench_audit.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    if a.audit:
        audit(load_config(a.config))
