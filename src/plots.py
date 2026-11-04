"""Figures from SAVED result files only (no recomputation of metrics here).

CLI: python src/plots.py baseline | figures
"""
import os

os.environ.setdefault("HF_HOME", "./.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "./.cache/torch")
os.environ.setdefault("XDG_CACHE_HOME", "./.cache")
os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_lithobench import load_config  # noqa: E402

SIM_NOTE = "simulated SEM (controlled forward model), not real CD-SEM data"


def measured_vs_true(cfg, split, meas_name, out_png, kind="cd", dose=8):
    import evaluate as ev
    meas = ev.load_meas(cfg, split, meas_name)
    m = ev.merged(cfg, meas[(meas.rep == 0) & (meas.dose == dose)], ev.load_truth(cfg, split))
    m = m[(m.kind == kind) & m.inside.notna() & ~m.fail]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for ins, col, lab in ((True, "tab:blue", "inside exposure range"), (False, "tab:orange", "outside (held out)")):
        g = m[m.inside == ins]
        ax.scatter(g.truth, g.value, s=2, alpha=0.3, c=col, label=f"{lab} (n={len(g)})")
    lo, hi = np.percentile(m.truth, [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel(f"true {kind.upper()} (nm)"); ax.set_ylabel(f"measured {kind.upper()} (nm)")
    ax.set_title(f"{meas_name}, {split}, {dose} frames\n{SIM_NOTE}", fontsize=8)
    ax.legend(fontsize=7, markerscale=4)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def sigma_vs_dose(summaries, out_png, kinds=("cd", "epe1d", "contact")):
    """summaries: {label: summary csv path}. Plots precision 3 sigma vs frames per kind."""
    fig, ax = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 3.8))
    for lab, path in summaries.items():
        s = pd.read_csv(path)
        for a, k in zip(ax, kinds):
            g = s[s.kind == k].sort_values("dose")
            a.plot(g.dose, g.prec_3s, "o-", label=lab)
    for a, k in zip(ax, kinds):
        a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xlabel("frames (dose)"); a.set_ylabel("precision 3 sigma (nm)")
        a.set_title(k); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle(SIM_NOTE, fontsize=8); fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def roughness_plot(csv, out_png, title):
    df = pd.read_csv(csv).groupby(["sigma_nm", "dose"]).mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, q in zip(ax, ("ler", "lwr")):
        for sig, g in df.groupby("sigma_nm"):
            a.plot(g.dose, g[f"{q}_raw_3s"], "o-", label=f"raw, sigma={sig}")
            a.plot(g.dose, g[f"{q}_unbiased_3s"], "s--", label=f"PSD-plateau corrected, sigma={sig}")
            a.plot(g.dose, g[f"{q}_repcorr_3s"], "^:", label=f"repeat-noise corrected, sigma={sig}")
            a.axhline(g[f"truth_{q}_3s"].iloc[0], color="k", lw=0.6)
            bad = g[g[f"{q}_rep_snr"] < 2]
            a.scatter(bad.dose, bad[f"{q}_repcorr_3s"], s=80, facecolors="none", edgecolors="r")
        a.set_xscale("log", base=2); a.set_xlabel("frames"); a.set_ylabel(f"{q.upper()} 3 sigma (nm)")
        a.set_title(f"{q.upper()} (black = truth; red ring = SNR < 2)", fontsize=8); a.grid(alpha=0.3)
    ax[0].legend(fontsize=6)
    fig.suptitle(f"{title}; {SIM_NOTE}", fontsize=8); fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def registration_plot(csv, out_png):
    s = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for st, g in s.groupby("set"):
        g = g.sort_values("dose")
        ax.plot(g.dose, g.centre_med, "o-", label=f"{st} centre median")
        ax.plot(g.dose, g.corner_p90, "s--", label=f"{st} corner-max p90")
    ax.set_xscale("log", base=2); ax.set_xlabel("frames"); ax.set_ylabel("registration error (px = nm)")
    ax.set_title("CAD-template registration error vs true stage offset + distortion", fontsize=8)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def final(cfg):
    """README figures for the test set from saved summaries / measurement files only."""
    import evaluate as ev
    d = os.path.join(cfg["paths"]["results"], "figures")
    os.makedirs(d, exist_ok=True)
    m5 = os.path.join(cfg["paths"]["results"], "model_eval")
    sums = {lab: os.path.join(m5, f"test_summary_{n}.csv") for lab, n in (
        ("classical", "classical_est"), ("U-Net s1", "screen_conv_cad_s1__learned"), ("DCNv2 s1", "screen_dcnv2_cad_s1__learned"),
        ("DCNv3 s1", "screen_dcnv3_cad_s1__learned"), ("DCNv4 s1", "screen_dcnv4_cad_s1__learned"),
        ("DCNv3 SEM-only s1", "cadtest_dcnv3_semonly_s1__learned"))}
    sums = {k: v for k, v in sums.items() if os.path.exists(v)}
    sigma_vs_dose(sums, os.path.join(d, "test_sigma_vs_dose.png"))
    if os.path.exists(os.path.join(cfg["paths"]["results"], "raw", "meas", "test", "screen_dcnv3_cad_s1__learned.pkl.gz")):
        measured_vs_true(cfg, "test", "screen_dcnv3_cad_s1__learned", os.path.join(d, "test_cd_meas_vs_true_dcnv3_s1.png"))
        measured_vs_true(cfg, "test", "classical_est", os.path.join(d, "test_cd_meas_vs_true_classical.png"))
    # dose benefit: lowest frames at which each method's CD 3 sigma <= classical 3 sigma at 64 frames
    cl = pd.read_csv(sums["classical"])
    ref = float(cl[(cl.kind == "cd") & (cl.dose == 64)].prec_3s.iloc[0])
    rows = []
    for lab, p in sums.items():
        s = pd.read_csv(p)
        g = s[s.kind == "cd"].set_index("dose").prec_3s.to_dict()
        rows.append(dict(method=lab, cd_prec3s_by_dose=json.dumps({int(k): round(v, 3) for k, v in g.items()}),
                         classical_64fr_ref=ref, frames_to_match=ev.dose_benefit(g, ref)))
    pd.DataFrame(rows).to_csv(os.path.join(d, "dose_benefit_test.csv"), index=False)
    # LCDU on test contacts (nominal condition, 20 reps), per method
    tr = ev.load_truth(cfg, "test")
    out = {}
    for lab, n in (("classical", "classical_est"), ("U-Net s1", "screen_conv_cad_s1__learned"), ("DCNv3 s1", "screen_dcnv3_cad_s1__learned")):
        pth = os.path.join(cfg["paths"]["results"], "raw", "meas", "test", n + ".pkl.gz")
        if os.path.exists(pth):
            m = ev.merged(cfg, ev.load_meas(cfg, "test", n), tr)
            out[lab] = {str(int(dz)): ev.lcdu(g, ev.nominal_index(cfg)) for dz, g in m.groupby("dose")}
    json.dump(out, open(os.path.join(d, "lcdu_test.json"), "w"), indent=1, default=float)


def m3(cfg):
    d = os.path.join(cfg["paths"]["results"], "classical_baseline")
    measured_vs_true(cfg, "val", "classical_est", os.path.join(d, "classical_cd_meas_vs_true_val.png"))
    sigma_vs_dose({"classical (est. reg)": os.path.join(d, "classical_val_summary_estreg.csv")},
                  os.path.join(d, "classical_sigma_vs_dose_val.png"))
    roughness_plot(os.path.join(d, "roughness_classical.csv"), os.path.join(d, "roughness_classical.png"), "classical, line/space set")
    registration_plot(os.path.join(d, "registration_val.csv"), os.path.join(d, "registration_val.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["baseline", "figures"])
    a = ap.parse_args()
    {"baseline": m3, "figures": final}[a.what](load_config())
