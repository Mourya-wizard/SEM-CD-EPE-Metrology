"""Trust analysis (M6): does gauge confidence predict measurement error?

Confidence per gauge measurement (one job = tile, condition, dose, rep 0) combines
  (a) seed spread: std of the gauge value across the model seeds (3 runs of the same variant), and
  (b) edge-profile quality: bloom-peak contrast along each probe divided by the image noise level
      (noise = 1.4826 * MAD of the image minus its 1-px Gaussian-smoothed version); min over a gauge's probes.
Combined uncertainty u = mean of the within-set percentile ranks of spread and of (-quality); confidence = -u.
Error = |mean over seeds - truth|. Reported on nominal (inside-range exposure) and stress conditions separately:
Spearman rank correlation between u and error, and risk-coverage curves (error of the kept gauges when the
least-confident ones are rejected). Failed measurements in any seed count as rejected-first (u = +inf).

  python src/confidence.py --runs <run_s1> <run_s2> <run_s3> --method learned|refine --split test --tag <name>
"""
import os

os.environ.setdefault("HF_HOME", "./.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "./.cache/torch")
os.environ.setdefault("XDG_CACHE_HOME", "./.cache")
os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy import ndimage, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev  # noqa: E402
import measure as ms  # noqa: E402
from data_lithobench import load_config  # noqa: E402


def profile_quality(cfg, split, tiles, jobs_by_tile, device="cuda", window=4.0, step=0.25):
    """Per (tile, cond, dose, gauge) edge-profile quality on the same noisy images the methods measured."""
    sc, rows = cfg["sim"], []
    for s, t in tiles:
        d = ev.load_cache(os.path.join(cfg["paths"]["results"], "raw", "cache", split, s, t + ".npz"))
        r = np.load(os.path.join(cfg["paths"]["results"], "raw", "reg", split, s, t + ".npz"))
        jl = {tuple(j): i for i, j in enumerate(r["jobs"].tolist())}
        for (c, fr) in jobs_by_tile[(s, t)]:
            img = ev.eval_noisy(d["clean"][c], fr, int(d["seed"]) * 100 + c, 0, sc).astype(np.float64)
            noise = 1.4826 * np.median(np.abs(img - ndimage.gaussian_filter(img, 1.0) - np.median(img - ndimage.gaussian_filter(img, 1.0))))
            sm = torch.as_tensor(ndimage.gaussian_filter(img, 1.0), device=device)
            j = jl[(c, fr, 0)]
            q, ns, _ = ms.probes_to_sem(d["probes"], r["A"][j], r["b"][j])
            ts = torch.arange(-window, window + 1e-9, step, device=device, dtype=torch.float64)
            qt, nt = torch.as_tensor(q, device=device), torch.as_tensor(ns, device=device)
            v = ms._torch_sample(sm, qt[:, :1] + nt[:, :1] * ts[None], qt[:, 1:] + nt[:, 1:] * ts[None])
            contrast = (v.max(1).values - v.min(1).values).cpu().numpy() / max(noise, 1e-9)
            pq = d["probes"].assign(q=contrast).groupby("gauge_id").q.min()
            rows.append(pd.DataFrame({"set": s, "tile": t, "cond": c, "dose": fr, "gauge_id": pq.index.values, "quality": pq.values}))
    return pd.concat(rows, ignore_index=True)


def risk_coverage(u, err, n_points=50):
    o = np.argsort(u, kind="stable")                 # most confident first
    e = err[o]
    cov = np.linspace(1.0 / n_points, 1.0, n_points)
    out = []
    for c in cov:
        k = max(1, int(round(c * len(e))))
        kept = e[:k]
        out.append(dict(coverage=float(c), n=int(k), mean_abs_err=float(np.nanmean(kept)), rms_err=float(np.sqrt(np.nanmean(kept ** 2))),
                        frac_gt_fail=float(np.nanmean(kept > 10.0))))
    return pd.DataFrame(out)


def analyse(cfg, runs, method, split, tag, kinds=("cd", "epe1d", "contact"), doses=(8, 64)):
    key = {"learned": "learned", "refine": "refine"}[method]
    meas = [ev.load_meas(cfg, split, f"{r}__{key}") for r in runs]
    tr = ev.load_truth(cfg, split)
    conds = ev.conditions(cfg)
    nominal = [i for i, c in enumerate(conds) if c["inside"]]
    stress = [i for i, c in enumerate(conds) if c["inside"] is None]
    cols = ["set", "tile", "cond", "dose", "gauge_id", "kind"]
    base = None
    for k, m in enumerate(meas):
        m = m[(m.rep == 0) & m.kind.isin(kinds) & m.dose.isin(doses) & m.cond.isin(nominal + stress)]
        m = m[cols + ["value", "valid"]].rename(columns={"value": f"v{k}", "valid": f"ok{k}"})
        base = m if base is None else base.merge(m, on=cols)
    base = base.merge(tr[["set", "tile", "cond", "gauge_id", "kind", "truth", "valid_t"]], on=["set", "tile", "cond", "gauge_id", "kind"])
    base = base[base.valid_t]
    V = base[[f"v{k}" for k in range(len(runs))]].values.astype(float)
    OK = base[[f"ok{k}" for k in range(len(runs))]].values.astype(bool) & np.isfinite(V)
    allok = OK.all(1)
    base["spread"] = np.where(allok, np.nanstd(np.where(OK, V, np.nan), 1), np.inf)
    base["err"] = np.abs(np.nanmean(np.where(OK, V, np.nan), 1) - base.truth.values)
    jobs = base[["set", "tile", "cond", "dose"]].drop_duplicates()
    jobs_by_tile = {k: list(zip(g.cond, g.dose)) for k, g in jobs.groupby(["set", "tile"])}
    q = profile_quality(cfg, split, list(jobs_by_tile), jobs_by_tile)
    base = base.merge(q, on=["set", "tile", "cond", "dose", "gauge_id"], how="left")
    res, curves = [], []
    for (kind, dose), g in base.groupby(["kind", "dose"]):
        for regime, cl in (("nominal", nominal), ("stress", stress)):
            h = g[g.cond.isin(cl)].copy()
            if len(h) < 20:
                continue
            fin = np.isfinite(h.spread) & np.isfinite(h.err)
            rs = h.spread.rank(pct=True); rq = (-h.quality).rank(pct=True)
            h["u"] = np.where(fin, 0.5 * (rs + rq), np.inf)
            hh = h[fin]
            rho_c, p_c = stats.spearmanr(hh.u, hh.err)
            rho_s, _ = stats.spearmanr(hh.spread, hh.err)
            rho_q, _ = stats.spearmanr(-hh.quality, hh.err)
            rc = risk_coverage(h.u.values, np.where(np.isfinite(h.err), h.err, 99.0))
            rc = rc.assign(kind=kind, dose=int(dose), regime=regime)
            curves.append(rc)
            full, at80 = rc.iloc[-1], rc.iloc[(rc.coverage - 0.8).abs().argmin()]
            res.append(dict(tag=tag, method=method, kind=kind, dose=int(dose), regime=regime, n=int(len(h)), n_any_seed_failed=int((~fin).sum()),
                            spearman_combined=float(rho_c), p_combined=float(p_c), spearman_spread=float(rho_s), spearman_quality=float(rho_q),
                            mean_abs_err_full=float(full.mean_abs_err), mean_abs_err_cov80=float(at80.mean_abs_err),
                            rms_full=float(full.rms_err), rms_cov80=float(at80.rms_err)))
    out = os.path.join(cfg["paths"]["results"], "confidence")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(res).to_csv(os.path.join(out, f"confidence_{tag}_{method}.csv"), index=False)
    pd.concat(curves).to_csv(os.path.join(out, f"risk_coverage_{tag}_{method}.csv"), index=False)
    return pd.DataFrame(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--method", default="learned")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    print(analyse(load_config(), a.runs, a.method, a.split, a.tag).round(3).to_string())
