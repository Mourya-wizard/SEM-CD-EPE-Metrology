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

fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
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
ax[0].set_xlabel("frames (dose)"); ax[0].set_ylabel("CD precision 3σ (nm)"); ax[0].set_title("(a) precision vs dose", loc="left")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
# (b) TMU-style by method
labs, vals, lo, hi, cols = ["classical"], [pick(cl, "cd", 8, "inside_tmu3s")], [0], [0], [C["classical"]]
for lab, names in SEEDS.items():
    v = [pick(summ(n), "cd", 8, "inside_tmu3s") for n in names]
    labs.append(f"{lab}\n3 seeds"); vals.append(np.mean(v)); lo.append(np.mean(v) - min(v)); hi.append(max(v) - np.mean(v)); cols.append(C[lab])
for lab, names in MULTI.items():
    v = [pick(summ(n), "cd", 8, "inside_tmu3s") for n in names]
    labs.append(f"{lab}\n{len(v)} seed{'s' if len(v) > 1 else ''}"); vals.append(np.mean(v)); lo.append(np.mean(v) - min(v)); hi.append(max(v) - np.mean(v)); cols.append(C[lab])
ax[1].bar(range(len(vals)), vals, yerr=[lo, hi], color=cols, capsize=3)
for i, v in enumerate(vals):
    ax[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
ax[1].set_xticks(range(len(vals))); ax[1].set_xticklabels(labs, fontsize=7.5); ax[1].set_ylim(5.5, 7.0)
ax[1].set_ylabel("CD TMU-style 3σ (nm)"); ax[1].set_title("(b) accuracy + precision, 8 frames", loc="left"); ax[1].grid(axis="y", alpha=0.3)
# (c) registration
pairs = [("classical", "classical_est", "classical_oracle"), ("DCNv3", "screen_dcnv3_cad_s1__learned", "screen_dcnv3_cad_s1_oraclereg__learned")]
x = np.arange(len(pairs))
est = [pick(summ(a), "epe1d", 8, "inside_tmu3s") for _, a, _ in pairs]
ora = [pick(summ(b), "epe1d", 8, "inside_tmu3s") for _, _, b in pairs]
ax[2].bar(x - 0.18, est, 0.36, color="#bbbbbb", label="CAD-template registration")
ax[2].bar(x + 0.18, ora, 0.36, color=[C[p[0]] for p in pairs], label="ideal registration")
for i in range(len(pairs)):
    ax[2].text(i - 0.18, est[i] + 0.1, f"{est[i]:.1f}", ha="center", fontsize=8); ax[2].text(i + 0.18, ora[i] + 0.1, f"{ora[i]:.1f}", ha="center", fontsize=8)
ax[2].set_xticks(x); ax[2].set_xticklabels([p[0] for p in pairs]); ax[2].set_ylabel("EPE TMU-style 3σ (nm)")
ax[2].set_title("(c) registration dominates EPE", loc="left"); ax[2].legend(fontsize=8); ax[2].grid(axis="y", alpha=0.3)
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
    ax[3].text(i - 0.18, r[1] * 1.15, f"{r[1]:.2f}", ha="center", fontsize=7.5)
ax[3].set_yscale("log"); ax[3].set_xticks(xs); ax[3].set_xticklabels([r[0] for r in rows], fontsize=8)
ax[3].set_ylabel("ms, 1e6 edge points, 4096 px"); ax[3].set_title("(d) edge-refinement runtime", loc="left"); ax[3].legend(fontsize=8)
ax[3].grid(axis="y", alpha=0.3)
fig.suptitle("Test set, simulated SEM (a-c); RTX PRO 5000 Blackwell vs Threadripper PRO 7965WX (d)", fontsize=9, y=0.02)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(FIG, "hero.png"), dpi=120)
print(os.path.join(FIG, "hero.png"), rows)
