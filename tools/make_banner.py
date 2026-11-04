"""README banner from saved data only: simulated SEM at decreasing dose with design edges + exact printed
contour, and real Carinthia-S SEM defects with the DCNv3 predicted outline (representative = Dice closest to
the class median). Needs the local caches (results/raw/cache) and Carinthia data/predictions.

  python tools/make_banner.py   -> results/figures/banner.png
"""
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from skimage import measure  # noqa: E402

import evaluate as ev  # noqa: E402
import registration as rg  # noqa: E402
from data_lithobench import load_config  # noqa: E402

cfg = load_config()
S = cfg["lithobench"]["crop_px"]
out_dir = next(os.path.join(ROOT, "results", d) for d in ("figures", "final") if os.path.isdir(os.path.join(ROOT, "results", d)))

# ---- simulated panel
s, t = [x for x in ev.eval_tiles(cfg, "val") if x[0] == "MetalSet"][3]
d = ev.load_cache(os.path.join(cfg["paths"]["results"], "raw", "cache", "val", s, t + ".npz"))
c = ev.nominal_index(cfg)
A = torch.as_tensor(d["A"][c][None], dtype=torch.float32)
b = torch.as_tensor(d["b"][c][None], dtype=torch.float32)
Fs = rg._warp(torch.as_tensor(d["F"][c].astype(np.float32))[None, None], A, b, S)[0, 0].numpy()
Ts = rg._warp(torch.as_tensor(d["T"].astype(np.float32))[None, None], A, b, S)[0, 0].numpy()
sim = [(f"{fr} frames", ev.eval_noisy(d["clean"][c], fr, int(d["seed"]) * 100 + c, 0, cfg["sim"])) for fr in (64, 8, 2)]

# ---- real panel
cr = cfg["paths"]["carinthia"]
idx = pd.read_csv(os.path.join(cr, cfg["carinthia"]["csv"]), sep=";").set_index("filename")
ck = os.path.join(ROOT, "checkpoints", "carinthia_dcnv3_s1")
per = pd.read_csv(os.path.join(ck, "test_per_image.csv"))
z = np.load(os.path.join(ck, "test_probs.npz"))
pos = {f: i for i, f in enumerate(z["files"])}
real = []
for lab, name in ((4, "pit"), (3, "particle"), (1, "scratch")):
    g = per[per.label == lab]
    r = g.iloc[(g.dice - g.dice.median()).abs().argmin()]
    img = cv2.imread(os.path.join(cr, idx.loc[r.filename, "image_path"]), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(os.path.join(cr, idx.loc[r.filename, "mask_path"]), cv2.IMREAD_GRAYSCALE) > 127
    pr = z["probs"][pos[r.filename]] > 127
    real.append((f"{name} (class {lab})\nDice {r.dice:.2f}", img, gt, pr))

fig = plt.figure(figsize=(18, 3.6))
gs = fig.add_gridspec(1, 6, wspace=0.04)
for i, (title, img) in enumerate(sim):
    a = fig.add_subplot(gs[0, i]); a.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
    for cc in measure.find_contours(Ts, 0.5):
        a.plot(cc[:, 1], cc[:, 0], c="#4fc3f7", lw=0.8, ls="--")
    for cc in measure.find_contours(Fs, 0.0):
        a.plot(cc[:, 1], cc[:, 0], c="#ff5252", lw=1.0)
    a.set_title(f"simulated SEM\n{title}", fontsize=10); a.axis("off")
for i, (title, img, gt, pr) in enumerate(real):
    a = fig.add_subplot(gs[0, 3 + i]); a.imshow(img, cmap="gray")
    for cc in measure.find_contours(gt.astype(float), 0.5):
        a.plot(cc[:, 1], cc[:, 0], c="white", lw=0.8, ls="--")
    for cc in measure.find_contours(pr.astype(float), 0.5):
        a.plot(cc[:, 1], cc[:, 0], c="#76ff03", lw=1.1)
    a.set_title(f"real SEM: {title}", fontsize=10); a.axis("off")
fig.text(0.25, 0.01, "dashed cyan = design   ·   red = exact printed contour (simulation truth)", ha="center", fontsize=9)
fig.text(0.75, 0.01, "dashed white = annotated defect   ·   green = DCNv3 prediction (representative, median-Dice cases)", ha="center", fontsize=9)
fig.savefig(os.path.join(out_dir, "banner.png"), dpi=110, bbox_inches="tight")
print(os.path.join(out_dir, "banner.png"))
