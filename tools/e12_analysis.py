"""E12 decision: CAD-template registration vs bias-aware registration on the VALIDATION set (dose 8, rep 0).

  python tools/e12_analysis.py <run> [--classical]
Needs results/raw/reg and results/raw/reg_bias for the val split, plus the measurements
  <run>__learned and <run>_regbias__learned  (python src/evaluate.py run-learned with regmode: cad | bias).
Gates (docs/ROADMAP.md): (1) outside-range CD failure <= 50% of the CAD value; (2) inside-range EPE1d TMU-style
<= CAD value + 0.10 nm; (3) ROC AUC of -PSR for detecting a registration centre error > 3 px >= 0.85.
Writes results/registration_e12/summary.json and per-condition tables.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import evaluate as ev  # noqa: E402
import registration as rg  # noqa: E402
from data_lithobench import load_config  # noqa: E402


def auc(score, positive):
    """ROC AUC of `score` (higher = more likely positive) via ranks."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reg_table(cfg, split, dose):
    """Per val job: registration centre error vs the true stage offset for both registrations (+ PSR / bias)."""
    S = cfg["lithobench"]["crop_px"]
    rows = []
    for s, t in ev.eval_tiles(cfg, split):
        z = np.load(os.path.join(cfg["paths"]["results"], "raw", "cache", split, s, t + ".npz"))
        rc = np.load(os.path.join(cfg["paths"]["results"], "raw", "reg", split, s, t + ".npz"))
        rb = np.load(os.path.join(cfg["paths"]["results"], "raw", "reg_bias", split, s, t + ".npz"))
        j = rc["jobs"]
        sel = (j[:, 2] == 0) & (j[:, 1] == dose)
        c = j[sel, 0]
        e_cad, _ = rg.registration_error(rc["A"][sel], rc["b"][sel], z["A"][c], z["b"][c], S)
        e_bias, _ = rg.registration_error(rb["A"][sel], rb["b"][sel], z["A"][c], z["b"][c], S)
        rows.append(pd.DataFrame({"set": s, "tile": t, "cond": c, "err_cad": e_cad, "err_bias": e_bias,
                                  "bias_est": rb["bias"][sel], "psr": rb["psr"][sel], "peak": rb["peak"][sel]}))
    df = pd.concat(rows, ignore_index=True)
    conds = ev.conditions(cfg)
    df["cname"] = df.cond.map({i: c["name"] for i, c in enumerate(conds)})
    df["inside"] = df.cond.map({i: c["inside"] for i, c in enumerate(conds)})
    return df


def metrics(cfg, meas, truth, dose):
    m = ev.merged(cfg, meas[(meas.rep == 0) & (meas.dose == dose)], truth)
    out_cd = m[(m.kind == "cd") & (m.inside == False)]  # noqa: E712
    in_epe = m[(m.kind == "epe1d") & (m.inside == True)]  # noqa: E712
    in_cd = m[(m.kind == "cd") & (m.inside == True)]  # noqa: E712
    return {"outside_cd_fail": float(out_cd.fail.mean()), "inside_epe1d_tmu3s": ev.accuracy_regression(in_epe)["tmu3s"],
            "inside_cd_tmu3s": ev.accuracy_regression(in_cd)["tmu3s"], "inside_cd_fail": float(in_cd.fail.mean())}


if __name__ == "__main__":
    run = sys.argv[1]
    cfg = load_config()
    dose = cfg["eval"]["selection_dose"]
    out = os.path.join(cfg["paths"]["results"], "registration_e12")
    os.makedirs(out, exist_ok=True)
    df = reg_table(cfg, "val", dose)
    df["mislock_cad"], df["mislock_bias"] = df.err_cad > 3.0, df.err_bias > 3.0
    bycond = df.groupby("cname", sort=False).agg(err_cad_med=("err_cad", "median"), err_bias_med=("err_bias", "median"),
                                                  mislock_cad=("mislock_cad", "mean"), mislock_bias=("mislock_bias", "mean"),
                                                  bias_est_mean=("bias_est", "mean"), psr_med=("psr", "median"))
    bycond.to_csv(os.path.join(out, "registration_by_condition.csv"))
    truth = ev.load_truth(cfg, "val")
    res = {"run": run, "dose": dose}
    for tag, name in (("cad", f"{run}__learned"), ("bias", f"{run}_regbias__learned")):
        res[tag] = metrics(cfg, ev.load_meas(cfg, "val", name), truth, dose)
    res["registration"] = {"cad_mislock_rate": float(df.mislock_cad.mean()), "bias_mislock_rate": float(df.mislock_bias.mean()),
                           "cad_err_median_px": float(df.err_cad.median()), "bias_err_median_px": float(df.err_bias.median()),
                           "psr_auc_bias": auc(-df.psr, df.mislock_bias), "psr_auc_bias_outside": auc(-df.psr[df.inside == False], df.mislock_bias[df.inside == False])}  # noqa: E712
    try:
        res["oracle"] = metrics(cfg, ev.load_meas(cfg, "val", f"{run}_oraclereg__learned"), truth, dose)
    except FileNotFoundError:
        pass
    g1 = res["bias"]["outside_cd_fail"] <= 0.5 * res["cad"]["outside_cd_fail"]
    g2 = res["bias"]["inside_epe1d_tmu3s"] <= res["cad"]["inside_epe1d_tmu3s"] + 0.10
    g3 = res["registration"]["psr_auc_bias"] >= 0.85
    res["gates"] = {"G1_outside_failure_halved": bool(g1), "G2_inside_epe_not_worse": bool(g2), "G3_psr_auc_ge_0.85": bool(g3)}
    res["decision"] = {"adopt_bias_registration": bool(g1 and g2), "adopt_psr_flag": bool(g3)}
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
