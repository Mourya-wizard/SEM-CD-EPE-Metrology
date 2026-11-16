"""E17 data preparation: Max Planck IC metal-2 SEM set (Edmond doi 10.17617/3.HY5SYN, CC BY 4.0) -> arrays + split.

  python tools/maxplanck_prepare.py        # reads data/maxplanck/dataset/{sems,labels}/m2, writes data/maxplanck/prepared/ and
                                           # results/maxplanck/maxplanck_split.json
Labels: SVG <path> (tracks, polygons, M/L/Z only; non-zero winding) and <circle> (vias). Target = tracks OR vias.
Only the CENTRAL 80% x 80% of every image is kept: neighbouring SEM images overlap by 10% and the dataset README notes that tracks spanning
images may be missing from a label file, so the outer 10% is both duplicated across images and label-incomplete.
Split: by image, seed 0, 70/15/15 (the kept regions of different images share no pixels).
"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "maxplanck", "dataset")
OUT = os.path.join(ROOT, "data", "maxplanck", "prepared")
H, W = 3536, 4096
Y0, Y1, X0, X1 = int(H * 0.1), int(H * 0.9), int(W * 0.1), int(W * 0.9)      # 353..3182, 409..3686
PATH_RE = re.compile(r'<path d="([^"]+)" fill="([^"]+)"')
CIRC_RE = re.compile(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="([\d.]+)"')
PT_RE = re.compile(r"(-?[\d.]+),(-?[\d.]+)")


def rasterise(svg):
    """-> uint8 (H, W): bit 0 = track, bit 1 = via."""
    t = open(svg).read()
    m = np.zeros((H, W), np.uint8)
    for d, _fill in PATH_RE.findall(t):
        subs = [np.array([[float(a), float(b)] for a, b in PT_RE.findall(s)], np.float32) for s in d.split("M")[1:]]
        subs = [s for s in subs if len(s) >= 3]
        if not subs:
            continue
        allp = np.concatenate(subs)
        x0, y0 = int(np.floor(allp[:, 0].min())) - 1, int(np.floor(allp[:, 1].min())) - 1
        x1, y1 = int(np.ceil(allp[:, 0].max())) + 2, int(np.ceil(allp[:, 1].max())) + 2
        roi = np.zeros((y1 - y0, x1 - x0), np.int16)
        for s in subs:                                                  # signed winding: opposite-orientation subpaths are holes
            area = 0.5 * np.sum(s[:, 0] * np.roll(s[:, 1], -1) - np.roll(s[:, 0], -1) * s[:, 1])
            c = np.zeros_like(roi, np.uint8)
            cv2.fillPoly(c, [np.round(s - [x0, y0]).astype(np.int32)], 1)
            roi += c.astype(np.int16) * (1 if area >= 0 else -1)
        ys, xs = max(y0, 0), max(x0, 0)
        ye, xe = min(y1, H), min(x1, W)
        sub = roi[ys - y0:ye - y0, xs - x0:xe - x0] != 0
        m[ys:ye, xs:xe] |= sub.astype(np.uint8)
    v = np.zeros((H, W), np.uint8)
    for cx, cy, r in CIRC_RE.findall(t):
        cv2.circle(v, (int(round(float(cx))), int(round(float(cy)))), int(round(float(r))), 1, -1)
    return m | (v << 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    ids = sorted(int(re.search(r"(\d+)", os.path.basename(f)).group(1)) for f in glob.glob(os.path.join(SRC, "labels", "m2", "label*.svg")))
    sems = {int(re.search(r"(\d+)", os.path.basename(f)).group(1)) for f in glob.glob(os.path.join(SRC, "sems", "m2", "sem*.png"))}
    assert set(ids) <= sems, "label without SEM image"
    n = len(ids)
    img = np.lib.format.open_memmap(os.path.join(OUT, "img.npy"), "w+", np.uint8, (n, Y1 - Y0, X1 - X0))
    lab = np.lib.format.open_memmap(os.path.join(OUT, "lab.npy"), "w+", np.uint8, (n, Y1 - Y0, X1 - X0))
    stats = []
    for k, i in enumerate(ids):
        im = cv2.imread(os.path.join(SRC, "sems", "m2", f"sem{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
        assert im.shape == (H, W), im.shape
        m = rasterise(os.path.join(SRC, "labels", "m2", f"label{i:04d}.svg"))
        img[k], lab[k] = im[Y0:Y1, X0:X1], m[Y0:Y1, X0:X1]
        stats.append(dict(id=i, fg_frac=float(((m[Y0:Y1, X0:X1] & 3) > 0).mean()), via_frac=float(((m[Y0:Y1, X0:X1] >> 1) & 1).mean())))
        if k % 20 == 0:
            print(k, n, stats[-1], flush=True)
    img.flush(); lab.flush()
    perm = np.random.default_rng(0).permutation(n)
    nt, nv = int(round(0.70 * n)), int(round(0.15 * n))
    split = {"train": sorted(int(ids[j]) for j in perm[:nt]), "val": sorted(int(ids[j]) for j in perm[nt:nt + nv]),
             "test": sorted(int(ids[j]) for j in perm[nt + nv:]), "index_order": [int(i) for i in ids], "crop": [Y0, Y1, X0, X1], "nm_per_px": 14.65}
    os.makedirs(os.path.join(ROOT, "results", "maxplanck"), exist_ok=True)
    json.dump(split, open(os.path.join(ROOT, "results", "maxplanck", "maxplanck_split.json"), "w"), indent=1)
    json.dump(stats, open(os.path.join(ROOT, "results", "maxplanck", "maxplanck_label_stats.json"), "w"), indent=1)
    print({k: len(v) for k, v in split.items() if k in ("train", "val", "test")}, "mean fg fraction", float(np.mean([s["fg_frac"] for s in stats])))


if __name__ == "__main__":
    sys.exit(main())
