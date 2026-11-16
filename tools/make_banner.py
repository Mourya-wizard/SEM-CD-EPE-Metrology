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

# ---- real panel 2: Max Planck IC metal-2 (E17), DCNv4-36 seed 1, typical (median Dice) and via-rich test tiles
sys.path.insert(0, os.path.join(ROOT, "tools"))
import maxplanck as mp  # noqa: E402
from make_gradcam import load_any  # noqa: E402

mimg, mlab = mp.arrays()
tl = mp.grid_tiles(mp.load_split()[1]["test"], mimg.shape[1:])
dpt = pd.read_csv(os.path.join(ROOT, "checkpoints", "e17_maxplanck_dcnv4_w36_s1", "test_tiles.csv")).dice.to_numpy()
vias = np.array([((mlab[k, y:y + mp.T, x:x + mp.T] >> 1) & 1).mean() for k, y, x in tl])
mmod = load_any(cfg, "e17_maxplanck_dcnv4_w36_s1")[0]
mp_panels = []
for name, j in (("typical tile", int(np.argsort(dpt)[len(dpt) // 2])), ("via-rich tile", int(vias.argmax()))):
    k, y, x = tl[j]
    tile = np.array(mimg[k, y:y + mp.T, x:x + mp.T])
    gt = (np.array(mlab[k, y:y + mp.T, x:x + mp.T]) & 3) > 0
    pr = mp.predict(mmod, torch.as_tensor(tile[None], device="cuda"))[0].cpu().numpy() > 0.5
    mp_panels.append((f"{name}\nDice {dpt[j]:.2f}", tile, gt, pr))

import paperstyle  # noqa: E402

paperstyle.use(17)
sim2 = [sim[0], sim[2]]                                          # 64 and 2 frames
real2 = [real[0], real[2]]                                       # pit, scratch
fig = plt.figure(figsize=(18, 4.25))
gs = fig.add_gridspec(1, 6, wspace=0.05, left=0.005, right=0.995, top=0.86, bottom=0.13)


def show(k, img, title, gt=None, pr=None, col=None, design=None, truth=None, vlim=None):
    a = fig.add_subplot(gs[0, k]); a.imshow(img, cmap="gray", **(vlim or {}))
    if design is not None:
        for cc in measure.find_contours(design, 0.5):
            a.plot(cc[:, 1], cc[:, 0], c="#4fc3f7", lw=1.0, ls="--")
        for cc in measure.find_contours(truth, 0.0):
            a.plot(cc[:, 1], cc[:, 0], c="#ff5252", lw=1.4)
    if gt is not None:
        for cc in measure.find_contours(gt.astype(float), 0.5):
            a.plot(cc[:, 1], cc[:, 0], c="white", lw=1.0, ls="--")
        for cc in measure.find_contours(pr.astype(float), 0.5):
            a.plot(cc[:, 1], cc[:, 0], c=col, lw=1.4)
    a.set_title(title, fontsize=17, linespacing=1.15); a.axis("off")


for k, (title, img) in enumerate(sim2):
    show(k, img, f"Simulated SEM\n{title}", design=Ts, truth=Fs, vlim=dict(vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5)))
for k, (title, img, gt, pr) in enumerate(real2):
    show(2 + k, img, "Carinthia-S" + "\n" + title.split("\n")[0].split(" (")[0] + ", Dice " + title.split("Dice ")[1], gt, pr, "#76ff03")
for k, (title, img, gt, pr) in enumerate(mp_panels):
    show(4 + k, img, "Max Planck metal-2" + "\n" + title.replace("\n", ", "), gt, pr, "#ffb300")
for x, txt in ((1 / 6, "design (dashed cyan), exact printed contour (red)"), (0.5, "annotation (dashed white), DCNv3 prediction (green)"),
               (5 / 6, "label (dashed white), DCNv4 prediction (amber)")):
    fig.text(x, 0.055, txt, ha="center", va="center", fontsize=15)
fig.savefig(os.path.join(out_dir, "banner.png"), dpi=130)
print(os.path.join(out_dir, "banner.png"))
