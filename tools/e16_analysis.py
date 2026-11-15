"""E16 (equal-width operator comparison): pre-registered gate evaluation on the VALIDATION set. See docs/ROADMAP.md.

  python tools/e16_analysis.py            -> results/roadmap_gates/e16-summary.csv, e16-gates.json (and prints them)

Simulated SEM: val measurements `<run>_regbias__learned` (bias-aware registration, doses 2 and 8, all noise repeats). Line/space roughness: results/figures/roughness_<run>_sigma{1,2}.csv. Real SEM: best
validation Dice (defect classes) of checkpoints/<name>/metrics.json. The test set is NOT used here.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluate as ev  # noqa: E402
from data_lithobench import load_config  # noqa: E402

SIM = {  # group -> {seed: run name}
    "conv_w32": {1: "e16_conv_w32_semonly_s1", 2: "e16_conv_w32_semonly_s2"},
    "conv_w36": {1: "e16_conv_w36_semonly_s1", 2: "e16_conv_w36_semonly_s2"},
    "dcnv2_w36": {1: "e16_dcnv2_w36_semonly_s1", 2: "e16_dcnv2_w36_semonly_s2"},
    "dcnv3_w36": {1: "cadtest_dcnv3_semonly_s1", 2: "cadtest_dcnv3_semonly_s2"},
    "dcnv4_w36": {1: "e16_dcnv4_w36_semonly_s1", 2: "e16_dcnv4_w36_semonly_s2"},
}
REAL = {  # group -> {seed: run name}; DCNv3 and DCNv4 default widths are already 36
    "conv_w32": {s: f"carinthia_conv_s{s}" for s in (1, 2, 3)},
    "conv_w36": {s: f"e16_carinthia_conv_w36_s{s}" for s in (1, 2, 3)},
    "dcnv2_w36": {s: f"e16_carinthia_dcnv2_w36_s{s}" for s in (1, 2, 3)},
    "dcnv3_w36": {s: f"carinthia_dcnv3_s{s}" for s in (1, 2, 3)},
    "dcnv4_w36": {s: f"carinthia_dcnv4_s{s}" for s in (1, 2, 3)},
}


def sim_row(cfg, truth, run):
    m = ev.load_meas(cfg, "val", f"{run}_regbias__learned")
    s = ev.summarize(cfg, m, truth, run, n_boot=0)             # all noise repeats: precision needs them
    out = {}
    for dose in (2, 8):
        r = s[(s.kind == "cd") & (s.dose == dose)].iloc[0]
        out.update({f"prec3s_{dose}fr": float(r.prec_3s), f"tmu_{dose}fr": float(r.inside_tmu3s), f"fail_{dose}fr": float(r.all_fail_rate),
                    f"stress_rmse_{dose}fr": float(r.stress_contrast_rmse)})
    fig = os.path.join(cfg["paths"]["results"], "figures")
    ls = pd.concat([pd.read_csv(os.path.join(fig, f"roughness_{run}_sigma{x}.csv")).assign(sigma=x) for x in (1, 2)])
    ls = ls[ls.dose == 64]
    out["ls_fail_64fr"] = float(ls.fail_rate.mean())
    out["ler_bias_pct_64fr"] = float((100 * (ls.ler_raw_3s / ls.truth_ler_3s - 1)).mean())
    return out


def real_row(name):
    p = os.path.join(ROOT, "checkpoints", name, "metrics.json")
    return float(json.load(open(p))["best_val_dice_defect"]) if os.path.exists(p) else float("nan")


def main():
    cfg = load_config()
    truth = ev.load_truth(cfg, "val")
    rows = []
    for g, seeds in SIM.items():
        for sd, run in seeds.items():
            try:
                rows.append(dict(group=g, seed=sd, run=run, **sim_row(cfg, truth, run)))
            except Exception as e:                       # run not finished / evaluated yet
                print(f"skip {run}: {type(e).__name__}: {e}")
    df = pd.DataFrame(rows)
    real = pd.DataFrame([dict(group=g, seed=sd, run=r, val_dice=real_row(r)) for g, ss in REAL.items() for sd, r in ss.items()])
    out_dir = os.path.join(cfg["paths"]["results"], "roadmap_gates")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "e16-sim-summary.csv"), index=False)
    real.to_csv(os.path.join(out_dir, "e16-real-summary.csv"), index=False)
    res = {"gates": {}}
    if not df.empty and {"conv_w32", "conv_w36"} <= set(df.group):
        g = lambda grp, col: df[df.group == grp].sort_values("seed")[col].to_numpy()          # noqa: E731
        res["G0_conv_w32_vs_w36_prec3s_8fr"] = [g("conv_w32", "prec3s_8fr").tolist(), g("conv_w36", "prec3s_8fr").tolist()]
        for dcn in ("dcnv2_w36", "dcnv3_w36", "dcnv4_w36"):
            if dcn not in set(df.group) or len(g(dcn, "prec3s_8fr")) < 2 or len(g("conv_w36", "prec3s_8fr")) < 2:
                continue
            p, c = g(dcn, "prec3s_8fr"), g("conv_w36", "prec3s_8fr")
            res["gates"][dcn] = {
                "G1a_prec_mean_le_0.95x_conv_and_each_seed_below_conv_min": bool(p.mean() <= 0.95 * c.mean() and (p < c.min()).all()),
                "G1b_tmu_not_worse_0.10": bool(g(dcn, "tmu_8fr").mean() <= g("conv_w36", "tmu_8fr").mean() + 0.10),
                "G1c_fail_not_worse_0.5pt": bool(g(dcn, "fail_8fr").mean() <= g("conv_w36", "fail_8fr").mean() + 0.005),
            }
            res["gates"][dcn]["G1_pass"] = bool(all(res["gates"][dcn].values()))
    if not real.empty:
        piv = real.pivot(index="seed", columns="group", values="val_dice")
        for dcn in ("dcnv2_w36", "dcnv3_w36", "dcnv4_w36"):
            if dcn in piv and "conv_w36" in piv and piv[[dcn, "conv_w36"]].notna().all(axis=None):
                res["gates"].setdefault(dcn, {})["G2_dice_mean_ge_conv_plus_0.003_and_every_seed"] = bool(
                    piv[dcn].mean() >= piv["conv_w36"].mean() + 0.003 and (piv[dcn] > piv["conv_w36"]).all())
        res["real_val_dice_mean"] = piv.mean().round(4).to_dict()
        res["real_val_dice_by_seed"] = {c: piv[c].round(4).tolist() for c in piv}
    json.dump(res, open(os.path.join(out_dir, "e16-gates.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False)); print(real.round(4).to_string(index=False)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
