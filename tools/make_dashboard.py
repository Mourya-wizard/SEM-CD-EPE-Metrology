"""Results dashboard (README hero figure) rebuilt from saved result tables only.

  python tools/make_dashboard.py   -> results/figures/hero.png
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle  # noqa: E402

paperstyle.use(14)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rdir(*names):
    for n in names:
        p = os.path.join(ROOT, "results", n)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(names)


EVAL, FIG, KER = rdir("model_eval", "m5"), rdir("figures", "final"), rdir("cuda_kernel", "m8")
C = {"classical": "#eb6834", "U-Net": "#2a78d6", "DCNv3": "#1baf7a", "DCNv2": "#8a6fd1", "DCNv4": "#5b9bd5", "SEM only": "#999999"}


def summ(name):
    return pd.read_csv(os.path.join(EVAL, f"test_summary_{name}.csv"))


def pick(s, kind, dose, col):
    r = s[(s.kind == kind) & (s.dose == dose)]
    return float(r[col].iloc[0])


SEEDS = {"U-Net": ["screen_conv_cad_s1__learned", "confirm_conv_cad_s2__learned", "confirm_conv_cad_s3__learned"],
         "DCNv3": ["screen_dcnv3_cad_s1__learned", "confirm_dcnv3_cad_s2__learned", "confirm_dcnv3_cad_s3__learned"]}
MULTI = {"DCNv2": ["screen_dcnv2_cad_s1__learned"],
         "DCNv4": ["screen_dcnv4_cad_s1__learned", "confirm_dcnv4_cad_s2__learned", "confirm_dcnv4_cad_s3__learned"],
         "SEM only": ["cadtest_dcnv3_semonly_s1__learned", "cadtest_dcnv3_semonly_s2__learned", "cadtest_dcnv3_semonly_s3__learned"]}
MULTI = {k: [n for n in v if os.path.exists(os.path.join(EVAL, f"test_summary_{n}.csv"))] for k, v in MULTI.items()}

fig, axs = plt.subplots(2, 2, figsize=(13, 10.5))
ax = axs.flat
# (a) precision vs dose
frames = [2, 4, 8, 16, 32, 64]
cl = summ("classical_est")
ax[0].plot(frames, [pick(cl, "cd", f, "prec_3s") for f in frames], "o-", c=C["classical"], label="classical")
for lab, names in SEEDS.items():
    v = np.array([[pick(summ(n), "cd", f, "prec_3s") for f in frames] for n in names])
    ax[0].plot(frames, v.mean(0), "o-", c=C[lab], label=f"{lab} (3 seeds)")
    ax[0].fill_between(frames, v.min(0), v.max(0), color=C[lab], alpha=0.2)
ax[0].axhline(pick(cl, "cd", 64, "prec_3s"), ls=":", c=C["classical"], lw=1)
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log"); ax[0].set_xticks(frames); ax[0].set_xticklabels(frames)
ax[0].set_yticks([0.4, 0.6, 1.0, 2.0, 3.0]); ax[0].set_yticklabels(["0.4", "0.6", "1.0", "2.0", "3.0"]); ax[0].minorticks_off(); ax[0].set_xlabel("frames averaged (dose)"); ax[0].set_ylabel("CD precision, 3σ (nm)"); ax[0].set_title("(a) CD precision versus dose", loc="left")
ax[0].legend(fontsize=12); ax[0].grid(alpha=0.3)
# (b) registration: CAD template -> bias-aware (E12) -> ideal, test set, 8 frames, EPE1d inside range
V2 = rdir("registration_test")
v2 = lambda n: pd.read_csv(os.path.join(V2, f"test_summary_{n}.csv"))
groups = [("classical tracer", [v2("classical_cad"), v2("classical_bias"), summ("classical_oracle")]),
          ("learned DCNv3", [v2("baseline_cad"), v2("baseline_bias"), summ("screen_dcnv3_cad_s1_oraclereg__learned")])]
regs = [("CAD template", "#bbbbbb"), ("bias-aware (E12)", "#2a78d6"), ("ideal*", "#1baf7a")]
x = np.arange(len(groups))
for k, (lab, col) in enumerate(regs):
    vals = [pick(g[k], "epe1d", 8, "inside_tmu3s") for _, g in groups]
    xx = x + (k - 1) * 0.27
    ax[1].bar(xx, vals, 0.27, color=col, label=lab)
    for xi, v in zip(xx, vals):
        ax[1].text(xi, v + 0.1, f"{v:.1f}", ha="center", fontsize=12)
ax[1].set_xticks(x); ax[1].set_xticklabels([g[0] for g in groups]); ax[1].set_ylabel("EPE TMU-style, 3σ (nm)")
ax[1].set_title("(b) EPE uncertainty by registration method", loc="left"); ax[1].set_ylim(0, 10.5); ax[1].legend(fontsize=12, ncol=3, loc="upper center"); ax[1].grid(axis="y", alpha=0.3)
ax[1].text(0.0, -0.13, "*ideal: SEM + CAD DCNv3; other bars: SEM-only DCNv3", transform=ax[1].transAxes, fontsize=12, color="#555")
# (c) real SEM, equal width: val Dice gain over conv-36 per seed (E16 Carinthia-S, E17 Max Planck)
real = {"Carinthia-S (E16)": pd.read_csv(os.path.join(rdir("roadmap_gates"), "e16-real-summary.csv")).rename(columns={"val_dice": "d"}),
        "Max Planck (E17)": pd.read_csv(os.path.join(rdir("maxplanck"), "e17-per-seed.csv")).rename(columns={"val_dice_mean": "d"})}
ops = [("conv_w32", "conv-32", "#999999"), ("dcnv2_w36", "DCNv2", C["DCNv2"]), ("dcnv3_w36", "DCNv3", C["DCNv3"]), ("dcnv4_w36", "DCNv4", C["DCNv4"])]
for gi, (gname, df) in enumerate(real.items()):
    ref = df[df.group == "conv_w36"].set_index("seed").d
    for oi, (g, lab, col) in enumerate(ops):
        dd = (df[df.group == g].set_index("seed").d - ref).dropna()
        xc = gi * 5 + oi
        ax[2].bar(xc, dd.mean(), 0.75, color=col, alpha=0.85, label=lab if gi == 0 else None)
        ax[2].plot([xc] * len(dd), dd.values, "k.", ms=5)
ax[2].axhline(0.003, ls="--", c="#c0392b", lw=1); ax[2].axhline(0, c="k", lw=0.6)
ax[2].text(8.4, 0.0032, "gate +0.003", ha="right", va="bottom", fontsize=12, color="#c0392b")
ax[2].set_xticks([1.5, 6.5]); ax[2].set_xticklabels(list(real))
ax[2].set_ylabel("validation Dice − conv-36 (dots: seeds)"); ax[2].set_title("(c) Real SEM: gain over width-matched conv", loc="left")
ax[2].legend(fontsize=12, ncol=2, loc="upper right"); ax[2].grid(axis="y", alpha=0.3)
# (d) kernel
b = pd.read_csv(os.path.join(KER, "benchmark.csv"))
b = b[(b.image_px == 4096) & (b.n_points == 1000000) & (b.order == "random")]
rows = []
for m, lab in (("cpu_openmp", "C++/OpenMP\n48 threads"), ("torch_grid_sample", "PyTorch\ngrid_sample"), ("cuda_manual", "CUDA\nmanual"), ("cuda_texture", "CUDA\ntexture")):
    k = b[(b.method == m) & (b.timing == "kernel")]
    if m.startswith("cuda"):
        k = k[k.block == 128] if (k.block == 128).any() else k
    e = b[(b.method == m) & (b.timing.isin(["end_to_end_reuse"]))]
    rows.append((lab, float(k.median_ms.min()) if len(k) else np.nan, float(e.median_ms.min()) if len(e) else np.nan))
xs = np.arange(len(rows))
ax[3].bar(xs - 0.18, [r[1] for r in rows], 0.36, color="#2a78d6", label="kernel only")
ax[3].bar(xs + 0.18, [r[2] for r in rows], 0.36, color="#9ec5ee", label="end-to-end (incl. transfer)")
for i, r in enumerate(rows):
    ax[3].text(i - 0.18, r[1] * 1.15, f"{r[1]:.2f}", ha="center", fontsize=12)
ax[3].set_yscale("log"); ax[3].set_xticks(xs); ax[3].set_xticklabels([r[0] for r in rows])
ax[3].set_ylabel("time (ms), 10$^6$ edge points"); ax[3].set_title("(d) Edge-refinement runtime", loc="left"); ax[3].legend(fontsize=12, loc="upper right"); ax[3].set_ylim(top=400)
ax[3].grid(axis="y", alpha=0.3)
fig.tight_layout(h_pad=2.5, w_pad=2.0)
fig.savefig(os.path.join(FIG, "hero.png"), dpi=120)
print(os.path.join(FIG, "hero.png"), rows)
