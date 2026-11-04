"""LCDU figure from the saved test LCDU table (results/<figures>/lcdu_test.json).

  python tools/plot_lcdu.py   -> results/<figures>/lcdu_test.png
"""
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = next(os.path.join(ROOT, "results", d) for d in ("figures", "final") if os.path.isdir(os.path.join(ROOT, "results", d)))
d = json.load(open(os.path.join(out, "lcdu_test.json")))
C = {"classical": "#eb6834", "U-Net s1": "#2a78d6", "DCNv3 s1": "#1baf7a"}
fig, ax = plt.subplots(figsize=(6.4, 4))
for lab, by in d.items():
    fr = sorted(int(k) for k in by)
    ax.plot(fr, [by[str(f)]["raw_lcdu_3s"] for f in fr], "o-", c=C.get(lab), label=f"{lab}: raw")
    ax.plot(fr, [by[str(f)]["corrected_lcdu_3s"] for f in fr], "s--", c=C.get(lab), alpha=0.7, label=f"{lab}: noise-corrected")
truth = [v["truth_lcdu_3s"] for by in d.values() for v in by.values()]
ax.axhspan(min(truth), max(truth), color="k", alpha=0.15, label="true LCDU (same contacts)")
ax.set_xscale("log", base=2); ax.set_xticks(fr); ax.set_xticklabels(fr)
ax.set_xlabel("frames (dose)"); ax.set_ylabel("contact LCDU 3σ (nm)"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
ax.set_title("Contact LCDU, test set (simulated SEM)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(out, "lcdu_test.png"), dpi=120)
print("saved")
