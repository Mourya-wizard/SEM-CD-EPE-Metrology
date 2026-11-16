"""README figures that are not produced elsewhere, in the shared paper style (tools/paperstyle.py).

  python tools/make_readme_figures.py [name ...]   -> results/figures/{pipeline_overview,key_results,simulator_sheet,carinthia_examples,cd_error_distribution}.png
Needs LithoBench (simulator sheet), Carinthia-S + checkpoint carinthia_dcnv3_s1 (examples) and the cached test measurements (scatter).
"""
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
from skimage import measure  # noqa: E402

import paperstyle  # noqa: E402
from data_lithobench import load_config  # noqa: E402

paperstyle.use(15)
cfg = load_config()
OUT = os.path.join(ROOT, "results", "figures")


def simulator_sheet():
    import sem_sim as ss
    sc, S, M = cfg["sim"], cfg["lithobench"]["crop_px"], cfg["sim"]["margin_px"]
    tile = json.load(open(os.path.join(cfg["paths"]["results"], "splits.json")))["val"]["MetalSet"][0]
    y0, x0 = ss.choose_crop(cfg, "MetalSet", tile, S)
    a, t = ss.load_crop(cfg, "MetalSet", tile, y0, x0, S)
    p = ss.sample_params(np.random.default_rng(7), sc, 1)
    out = ss.simulate(a[None], p, sc, S, seed=7, device="cuda", frames=sc["frames"])
    panels = [("(a) design and exact\nprinted contour", t[M:-M, M:-M], True), ("(b) noise-free\nSEM image", out["clean"][0], False)]
    panels += [(f"({chr(99 + k)}) {fr} frames", out["noisy"][fr][0], False) for k, fr in enumerate((64, 8, 2))]
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.9))
    for a_, (title, img, contour) in zip(ax, panels):
        a_.imshow(img, cmap="gray")
        if contour:
            a_.contour(out["F"][0][M:-M, M:-M], [0], colors="#ff5252", linewidths=1.2)
        a_.set_title(title, fontsize=17, linespacing=1.15); a_.axis("off")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(OUT, "simulator_sheet.png"), dpi=110); plt.close(fig)


def carinthia_examples():
    import carinthia as ca
    names = {1: "scratch", 2: "class 2", 3: "particle", 4: "pit", 5: "class 5", 6: "no defect"}
    ck = os.path.join(ROOT, "checkpoints", "carinthia_dcnv3_s1")
    te = pd.read_csv(os.path.join(ck, "test_per_image.csv"))
    z = np.load(os.path.join(ck, "test_probs.npz"))
    pm = dict(zip(z["files"], z["probs"]))
    td = te[te.label.isin(ca.DEFECT)].sort_values("dice")
    med = lambda g: g.iloc[(g.dice - g.dice.median()).abs().argmin()]  # noqa: E731  class-median case
    picks = ([("typical", med(td[td.label == c])) for c in (4, 3, 1)]
             + [("failure", r) for _, r in td.head(3).iterrows()]
             + [("no defect", r) for _, r in te[te.label == 6].sort_values("max_prob").iloc[[0, -1]].iterrows()])
    X, Y, _ = ca.load_arrays(cfg, [r.filename for _, r in picks])
    fig, ax = plt.subplots(2, 4, figsize=(14, 8.4))
    for k, (a_, (tag, r)) in enumerate(zip(ax.ravel(), picks)):
        a_.imshow(X[k], cmap="gray")
        for c in measure.find_contours(Y[k].astype(float), 0.5):
            a_.plot(c[:, 1], c[:, 0], c="white", lw=1.0, ls="--")
        for c in measure.find_contours((pm[r.filename] > 127).astype(float), 0.5):
            a_.plot(c[:, 1], c[:, 0], c="#76ff03", lw=1.3)
        sub = f"Dice {r.dice:.2f}" if r.label != 6 else f"max. score {r.max_prob:.2f}"
        head = "no defect" if r.label == 6 else f"{tag}: {names[int(r.label)]}"
        a_.set_title(f"{head}\n{sub}", fontsize=16, linespacing=1.15); a_.axis("off")
    fig.text(0.5, 0.015, "annotation (dashed white), DCNv3 prediction (green); test images", ha="center", fontsize=15)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.2)
    fig.savefig(os.path.join(OUT, "carinthia_examples.png"), dpi=100); plt.close(fig)


def cd_error_distribution():
    """CD error (measured - true) at 8 frames, test set, bias-aware registration: classical vs learned, inside and outside the exposure range."""
    import evaluate as ev
    truth = ev.load_truth(cfg, "test")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    bins = np.linspace(-12, 12, 97)
    for name, lab, col in (("classical_est_regbias", "classical edge tracer", "#eb6834"),
                           ("cadtest_dcnv3_semonly_s1_regbias__learned", "learned DCNv3 (SEM only)", "#1baf7a")):
        meas = ev.load_meas(cfg, "test", name)
        m = ev.merged(cfg, meas[(meas.rep == 0) & (meas.dose == 8)], truth)
        m = m[(m.kind == "cd") & m.inside.notna() & ~m.fail]
        for a_, ins in zip(axs, (True, False)):
            e = (m[m.inside == ins].value - m[m.inside == ins].truth).to_numpy()
            a_.hist(e, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=col)
            a_.hist(e, bins=bins, density=True, histtype="step", lw=1.6, color=col, label=f"{lab}: 3σ = {3 * e.std():.1f} nm")
    for a_, title in zip(axs, ("(a) inside the training exposure range", "(b) outside the range (held out)")):
        a_.axvline(0, c="k", lw=0.8); a_.set_xlabel("CD error, measured − true (nm)"); a_.set_title(title, loc="left")
        a_.legend(loc="upper left", fontsize=13); a_.grid(alpha=0.3); a_.set_ylim(0, 0.255)
    axs[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cd_error_distribution.png"), dpi=100); plt.close(fig)


def pipeline_overview():
    """The measurement pipeline on one real validation tile: design, 8-frame SEM, registration, model TSDF, measured contour + CD gauges."""
    import torch
    import gradcam_inputs as g
    item = g.pick_items(cfg, "val", dose=8, per_set=1, offset=5)[0]                     # MetalSet tile with 8 features, nominal condition
    import evaluate as ev
    M = cfg["sim"]["margin_px"]
    layout = ev.load_cache(os.path.join(cfg["paths"]["results"], "raw", "cache", "val", item["set"], item["tile"] + ".npz"))["T"][M:-M, M:-M]
    m, _ = g.load_run(cfg, "cadtest_dcnv3_semonly_s1")
    with torch.no_grad():
        tsdf = m(g.model_input(item, False))[0, 0].float().cpu().numpy()
    img, cad, truth = item["img"], item["cad"], item["truth"]
    cd = item["probes"]
    cd = cd[cd.kind == "cd"]
    mids, gid = [], []                                                                  # spread-out subset: one gauge per 70 px
    for k, q in cd.groupby("gauge_id"):
        c_ = q[["x", "y"]].to_numpy().mean(0)
        if all(np.hypot(*(c_ - o)) > 70 for o in mids):
            mids.append(c_); gid.append(k)
    vl = dict(vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
    lim = float(np.percentile(np.abs(tsdf), 99))
    steps = [("1  Design layout", "layout clip (white = metal);\ndefines where to measure"),
             ("2  Low-dose SEM image", "8-frame image of the printed\npattern (shot noise visible)"),
             ("3  Registration", "layout (cyan) mapped onto the\nimage, removing stage offset"),
             ("4  Contour model", "U-Net predicts a signed distance\nto the edge for every pixel"),
             ("5  Measurement", "zero crossing = sub-pixel contour;\nCD read at layout gauges")]
    fig, ax = plt.subplots(1, 5, figsize=(17, 5.0))
    ax[0].imshow(layout, cmap="gray")
    ax[1].imshow(img, cmap="gray", **vl)
    ax[2].imshow(img, cmap="gray", **vl)
    for c in measure.find_contours(cad, 0.5):
        ax[2].plot(c[:, 1], c[:, 0], c="#4fc3f7", lw=1.3)
    ax[3].imshow(np.clip(tsdf, -lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[3].contour(tsdf, [0], colors="k", linewidths=1.2)
    ax[4].imshow(img, cmap="gray", **vl)
    for c in measure.find_contours(truth, 0.0):
        ax[4].plot(c[:, 1], c[:, 0], c="#ff5252", lw=1.0, ls="--")
    for c in measure.find_contours(tsdf, 0.0):
        ax[4].plot(c[:, 1], c[:, 0], c="#76ff03", lw=1.2)
    for k in gid:
        p = cd[cd.gauge_id == k][["x", "y"]].to_numpy()
        ax[4].plot(p[:, 0], p[:, 1], c="#ffd54f", lw=2.0, solid_capstyle="butt")
    for a_, (t1, t2) in zip(ax, steps):
        a_.set_title(t1, fontsize=17, weight="bold", loc="left"); a_.axis("off")
        a_.text(0.0, -0.05, t2, transform=a_.transAxes, ha="left", va="top", fontsize=14, linespacing=1.3, color="#333")
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=2.2)
    fig.canvas.draw()
    for k in range(4):                                                                   # arrows between steps
        b0, b1 = ax[k].get_position(), ax[k + 1].get_position()
        y = (b0.y0 + b0.y1) / 2
        fig.add_artist(matplotlib.patches.FancyArrowPatch((b0.x1 + 0.002, y), (b1.x0 - 0.002, y), transform=fig.transFigure,
                                                          arrowstyle="-|>", mutation_scale=18, color="#333", lw=1.4))
    fig.text(0.5, 0.015, "Step 5: green = measured contour, red dashed = exact simulated contour, yellow = CD gauges (subset).  "
             "Step 4: red = inside the feature, blue = outside.", ha="center", fontsize=13.5, color="#333")
    fig.savefig(os.path.join(OUT, "pipeline_overview.png"), dpi=110, facecolor="white"); plt.close(fig)


def key_results():
    """Headline-result cards (README top). Times New Roman only. Values come from the result tables cited on each card."""
    from matplotlib.patches import FancyBboxPatch
    with plt.rc_context({"font.family": "Times New Roman", "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
                         "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold"}):
        cards = [("~4×", "lower electron dose", "for the same CD precision\nas a calibrated classical tracer", "E5", "#1f5fa8"),
                 ("38%", "lower EPE uncertainty", "8.3 → 5.2 nm with bias-aware\nregistration of SEM to layout", "E12", "#1baf7a"),
                 ("47% → 2%", "measurement failures", "outside the training exposure\nrange, same registration change", "E12", "#c0392b"),
                 ("41%", "lower width error, real IC", "41.6 → 24.6 nm vs a tuned\nthreshold, Max Planck metal-2", "E17", "#6250d6"),
                 ("62×", "faster edge refinement", "CUDA texture kernel vs 48 CPU\nthreads, 1M points in 1.75 ms", "E10", "#eb6834")]
        fig = plt.figure(figsize=(16, 3.5))
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 22); ax.axis("off")
        w, gap = 18.3, 1.6
        for k, (big, head, sub, exp, col) in enumerate(cards):
            x = gap / 2 + k * (w + gap)
            ax.add_patch(FancyBboxPatch((x, 1.2), w, 19.6, boxstyle="round,pad=0,rounding_size=1.2", fc="#f7f8fa", ec="#d5d9e0", lw=1.2))
            ax.add_patch(FancyBboxPatch((x, 19.6), w, 1.2, boxstyle="square,pad=0", fc=col, ec="none"))
            ax.text(x + w / 2, 14.6, big, ha="center", va="center", fontsize=36, weight="bold", color=col)
            ax.text(x + w / 2, 9.9, head, ha="center", va="center", fontsize=17, weight="bold", color="#1b1b1b")
            ax.text(x + w / 2, 5.9, sub, ha="center", va="center", fontsize=13.5, color="#444", linespacing=1.3)
            ax.text(x + w - 0.8, 2.2, exp, ha="right", va="bottom", fontsize=11.5, color="#888", style="italic")
        fig.savefig(os.path.join(OUT, "key_results.png"), dpi=110, facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["pipeline_overview", "key_results", "simulator_sheet", "carinthia_examples", "cd_error_distribution"]   # or name the figures to build
    for n in todo:
        f = globals()[n]
        f()
        print("done", f.__name__)
