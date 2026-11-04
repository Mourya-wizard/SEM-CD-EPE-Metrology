"""Evaluation: cached val/test sets, shared registration, method runner, and metrology metrics.

Pipeline (each step writes files under results/raw/ and is resumable per tile):
  cache    : per tile, per condition -> clean SEM (fp16), truth field F (fp16), true (A,b), probes, truth gauges
  register : per (tile, condition, dose, rep) estimated (A,b) from the CAD template; identical for every method
  run      : method -> per-gauge measurements for all conditions/doses (+ 20 noise reps on the nominal condition)
  report   : metrics json/csv under results/<tag>/
Noise for evaluation is generated with a CPU torch generator from noise_seed(tile_seed*100+cond, frames, rep),
so every process reproduces exactly the same noisy image without storing it.
TMU-style caveat: reference uncertainty is treated as zero because simulated truth is exact.
"""
import os

os.environ.setdefault("HF_HOME", "./.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "./.cache/torch")
os.environ.setdefault("XDG_CACHE_HOME", "./.cache")
os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")

import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gauges as gg  # noqa: E402
import sem_sim as ss  # noqa: E402
from data_lithobench import load_config  # noqa: E402

TMU_CAVEAT = "TMU-style: residual 3-sigma about the OLS line vs simulated truth; reference uncertainty = 0 (exact simulated truth)."
CONTACT_SETS = ("ViaSet", "StdContact", "StdContactTest")


# ----------------------------------------------------------------------------------------------- set-up
def conditions(cfg):
    ec = cfg["eval"]
    out = [dict(name=f"thr{t:.3f}", threshold=t, mode="train", inside=True) for t in ec["thresholds_inside"]]
    out += [dict(name=f"thr{t:.3f}", threshold=t, mode="train", inside=False) for t in ec["thresholds_outside"]]
    out += [dict(name=m, threshold=cfg["sim"]["threshold_nominal"], mode=m, inside=None) for m in ("stress_contrast", "stress_blur")]
    return out


def nominal_index(cfg):
    return [c["name"] for c in conditions(cfg)].index(f"thr{cfg['sim']['threshold_nominal']:.3f}")


def eval_tiles(cfg, split):
    sp = json.load(open(os.path.join(cfg["paths"]["results"], "splits.json")))
    ec = cfg["eval"]
    if split == "val":
        return [(s, t) for s in sp["val"] for t in sp["val"][s][: ec["val_tiles_per_set"]]]
    if split == "test":
        return [(s, t) for s in sp["test"] for t in sp["test"][s]]
    if split == "smoke":
        return [(s, t) for s in sp["val"] for t in sp["val"][s][:2]]
    raise ValueError(split)


def tile_seed(set_name, tile):
    return int(hashlib.sha256(f"{set_name}/{tile}".encode()).hexdigest()[:8], 16)


def raw_dir(cfg, *parts):
    d = os.path.join(cfg["paths"]["results"], "raw", *parts)
    os.makedirs(d, exist_ok=True)
    return d


def _atomic_savez(path, **arrs):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def eval_noisy(clean, frames, seed, rep, sc):
    """Deterministic CPU noise for evaluation (identical in every process)."""
    g = torch.Generator(device="cpu").manual_seed(ss.noise_seed(seed, frames, rep))
    x = torch.as_tensor(np.asarray(clean, np.float32))[None, None]
    return ss.add_noise(x, frames, sc, g)[0, 0].numpy()


# ----------------------------------------------------------------------------------------------- cache
def cache_tile(cfg, split, set_name, tile, device="cuda"):
    path = os.path.join(raw_dir(cfg, "cache", split, set_name), tile + ".npz")
    if os.path.exists(path):
        return path
    sc, gc, S, M = cfg["sim"], cfg["gauges"], cfg["lithobench"]["crop_px"], cfg["sim"]["margin_px"]
    y0, x0 = ss.choose_crop(cfg, set_name, tile, S)
    a, T = ss.load_crop(cfg, set_name, tile, y0, x0, S)
    seed = tile_seed(set_name, tile)
    rng = np.random.default_rng(seed)
    base = ss.sample_params(rng, sc, 1)
    stress = {"stress_contrast": ss.sample_params(rng, sc, 1, "stress_contrast")["contrast"],
              "stress_blur": ss.sample_params(rng, sc, 1, "stress_blur")["blur_px"]}
    probes = gg.build_gauges(T, gc, set_name in CONTACT_SETS, gc["border_px"])
    conds = conditions(cfg)
    clean, Fs, As, bs, truth = [], [], [], [], []
    for ci, c in enumerate(conds):
        p = {k: v.copy() for k, v in base.items()}
        p["threshold"][:] = c["threshold"]
        if c["mode"] == "stress_contrast":
            p["contrast"] = stress["stress_contrast"]
        if c["mode"] == "stress_blur":
            p["blur_px"] = stress["stress_blur"]
        o = ss.simulate(a[None], p, sc, S, seed=seed * 100 + ci, device=device, bumps=True)
        t, ok = gg.truth_offsets(o["F"][0], probes, gc["probe_half_window_px"])
        gv = gg.gauge_values(probes, t, ok)
        gv["cond"] = ci
        truth.append(gv)
        clean.append(o["clean"][0].astype(np.float16)); Fs.append(o["F"][0].astype(np.float16))
        As.append(o["A"][0]); bs.append(o["b"][0])
    tr = pd.concat(truth, ignore_index=True)
    _atomic_savez(path, clean=np.stack(clean), F=np.stack(Fs), A=np.stack(As), b=np.stack(bs), T=T,
                  origin=np.array([x0 - M, y0 - M]), seed=seed,
                  params=json.dumps({k: float(v[0]) for k, v in base.items()} | {k: float(v[0]) for k, v in stress.items()}),
                  probes=probes.to_records(index=False, column_dtypes={"kind": "U8"}),
                  truth=tr.to_records(index=False, column_dtypes={"kind": "U8"}))
    return path


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["probes"] = pd.DataFrame.from_records(d["probes"])
    d["truth"] = pd.DataFrame.from_records(d["truth"])
    return d


# ----------------------------------------------------------------------------------------------- registration
def jobs_for_tile(cfg, set_name, tile, split):
    """(cond, dose, rep) jobs: all conditions x doses rep 0; nominal x doses reps 1..R-1 on precision tiles."""
    ec = cfg["eval"]
    nom = nominal_index(cfg)
    jobs = [(ci, fr, 0) for ci in range(len(conditions(cfg))) for fr in cfg["sim"]["frames"]]
    prec = [t for s_, t in eval_tiles(cfg, split) if s_ == set_name][: ec["precision_tiles_per_set"]]
    if tile in prec:
        jobs += [(nom, fr, r) for fr in cfg["sim"]["frames"] for r in range(1, ec["precision_reps"])]
    return jobs


def register_tile(cfg, split, set_name, tile, device="cuda"):
    import registration as rg
    path = os.path.join(raw_dir(cfg, "reg", split, set_name), tile + ".npz")
    if os.path.exists(path):
        return path
    rc, sc, M = cfg["registration"], cfg["sim"], cfg["sim"]["margin_px"]
    d = load_cache(os.path.join(raw_dir(cfg, "cache", split, set_name), tile + ".npz"))
    tm, wt = rg.design_template(d["T"][None], rc)
    tmt, wtt = torch.as_tensor(tm, device=device)[:, None], torch.as_tensor(wt, device=device)[:, None]
    jobs = jobs_for_tile(cfg, set_name, tile, split)
    A = np.zeros((len(jobs), 2, 2)); b = np.zeros((len(jobs), 2))
    bs = 32
    for i in range(0, len(jobs), bs):
        chunk = jobs[i:i + bs]
        imgs = np.stack([eval_noisy(d["clean"][c], fr, int(d["seed"]) * 100 + c, r, sc) for c, fr, r in chunk])
        x = torch.as_tensor(imgs, device=device)[:, None]
        Ae, be = rg.register(x, tmt.expand(len(chunk), -1, -1, -1), rc, M, weight=wtt.expand(len(chunk), -1, -1, -1))
        A[i:i + len(chunk)], b[i:i + len(chunk)] = Ae.cpu().numpy(), be.cpu().numpy()
    _atomic_savez(path, jobs=np.array(jobs), A=A, b=b)
    return path


# ----------------------------------------------------------------------------------------------- methods
def _measure_tile_classical(args):
    cfg, split, set_name, tile, cparams, reg_mode, rep0_only = args
    import measure as ms
    torch.set_num_threads(1)
    sc = cfg["sim"]
    d = load_cache(os.path.join(cfg["paths"]["results"], "raw", "cache", split, set_name, tile + ".npz"))
    r = np.load(os.path.join(cfg["paths"]["results"], "raw", "reg", split, set_name, tile + ".npz"))
    rows = []
    for j, (c, fr, rep) in enumerate(r["jobs"]):
        if (reg_mode == "oracle" or rep0_only) and rep != 0:
            continue
        img = eval_noisy(d["clean"][c], int(fr), int(d["seed"]) * 100 + int(c), int(rep), sc)
        A, b = (d["A"][c], d["b"][c]) if reg_mode == "oracle" else (r["A"][j], r["b"][j])
        cp = cparams[str(int(fr))]
        t, v = ms.measure_classical(img, d["probes"], A, b, cp)
        t = t - cp.get("edge_offset_px", 0.0)
        gv = gg.gauge_values(d["probes"], t, v)
        rows.append(pd.DataFrame({"gauge_id": gv.gauge_id.values.astype(np.int32), "kind": gv.kind.values,
                                  "value": gv.value.values.astype(np.float32), "valid": gv.valid.values.astype(bool),
                                  "cond": np.int16(c), "dose": np.int16(fr), "rep": np.int16(rep)}))
    out = pd.concat(rows, ignore_index=True)
    out["set"], out["tile"] = set_name, tile
    return out


def classical_params(cfg, tuned=None):
    base = {k: cfg["classical"][k] for k in ("sigma", "frac", "search_px", "prom_radius_px", "min_prom", "step", "w_in", "w_out")}
    out = {}
    for fr in cfg["sim"]["frames"]:
        p = dict(base)
        if tuned:
            p.update(tuned[str(fr)])
        out[str(fr)] = p
    return out


def _limit_threads():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    torch.set_num_threads(1)


def run_classical(cfg, split, cparams, reg_mode="est", workers=None, tiles=None, rep0_only=False, label=""):
    """CPU method: one process per tile, at most cfg['eval']['max_cpu_workers'] processes, 1 thread each.
    A dead worker raises BrokenProcessPool instead of hanging; progress is printed every 10 tiles."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    workers = min(workers or cfg["eval"]["max_cpu_workers"], cfg["eval"]["max_cpu_workers"])
    tiles = tiles or eval_tiles(cfg, split)
    args = [(cfg, split, s, t, cparams, reg_mode, rep0_only) for s, t in tiles]
    parts, t0 = [], time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_limit_threads) as ex:
        futs = [ex.submit(_measure_tile_classical, a) for a in args]
        for i, f in enumerate(as_completed(futs)):
            parts.append(f.result(timeout=600))
            if (i + 1) % 10 == 0 or i + 1 == len(futs):
                print(f"  classical {label} {reg_mode} {i + 1}/{len(futs)} tiles {time.time() - t0:.0f}s", flush=True)
    return pd.concat(parts, ignore_index=True)


def save_meas(df, cfg, split, name):
    path = os.path.join(raw_dir(cfg, "meas", split), name + ".pkl.gz")
    df.to_pickle(path + ".tmp", compression="gzip")
    os.replace(path + ".tmp", path)
    return path


def load_meas(cfg, split, name):
    return pd.read_pickle(os.path.join(cfg["paths"]["results"], "raw", "meas", split, name + ".pkl.gz"), compression="gzip")


def load_truth(cfg, split, tiles=None):
    parts = []
    for s, t in tiles or eval_tiles(cfg, split):
        z = np.load(os.path.join(cfg["paths"]["results"], "raw", "cache", split, s, t + ".npz"))
        tr = pd.DataFrame.from_records(z["truth"])
        dcd = pd.DataFrame.from_records(z["probes"]).groupby("gauge_id").design_cd.first()
        tr["design_cd"] = tr.gauge_id.map(dcd).values
        tr["set"], tr["tile"] = s, t
        parts.append(tr)
    tr = pd.concat(parts, ignore_index=True).rename(columns={"value": "truth", "valid": "valid_t"})
    return tr


# ----------------------------------------------------------------------------------------------- metrics
def merged(cfg, meas, truth):
    m = meas.merge(truth, on=["set", "tile", "cond", "gauge_id", "kind"], how="inner")
    m = m[m.valid_t].copy()
    m["err"] = m.value - m.truth
    fail_nm = cfg["eval"]["fail_abs_nm"]
    m["fail"] = ~m.valid | ~np.isfinite(m.err) | (np.abs(m.err) > fail_nm)
    conds = conditions(cfg)
    m["inside"] = m.cond.map({i: c["inside"] for i, c in enumerate(conds)})
    m["cname"] = m.cond.map({i: c["name"] for i, c in enumerate(conds)})
    return m


def _ols(x, y):
    if len(x) < 3:
        return dict(slope=np.nan, offset=np.nan, r2=np.nan, resid_sd=np.nan, n=len(x))
    X = np.stack([x, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    r2 = 1 - res.var() / y.var() if y.var() > 0 else np.nan
    return dict(slope=float(coef[0]), offset=float(coef[1]), r2=float(r2), resid_sd=float(res.std(ddof=2)), n=int(len(x)))


def accuracy_regression(m):
    """m: merged rows for one (method, kind, dose), rep 0. Bias/RMSE on non-failed; failure rate on all."""
    ok = m[~m.fail]
    r = dict(n=int(len(m)), fail_rate=float(m.fail.mean()) if len(m) else np.nan,
             bias=float(ok.err.mean()) if len(ok) else np.nan, rmse=float(np.sqrt((ok.err ** 2).mean())) if len(ok) else np.nan)
    o = _ols(ok.truth.values.astype(float), ok.value.values.astype(float))
    r.update({k: o[k] for k in ("slope", "offset", "r2", "n")} | {"tmu3s": 3 * o["resid_sd"]})
    r["n_fit"] = o["n"]
    return r


def precision(m):
    """Per-gauge std over noise reps (>= 10 valid reps), pooled RMS -> 3 sigma."""
    ok = m[~m.fail]
    g = ok.groupby(["set", "tile", "cond", "gauge_id"]).value
    sd = g.std(ddof=1)[g.count() >= 10]
    return dict(three_sigma=float(3 * np.sqrt((sd ** 2).mean())) if len(sd) else np.nan, n_gauges=int(len(sd)))


def bootstrap_ci(m, fn, n_boot=1000, seed=0, unit="tile"):
    """Tile-level bootstrap 95% CI of fn(df) -> float."""
    rng = np.random.default_rng(seed)
    units = m[["set", unit]].drop_duplicates().values.tolist()
    groups = {tuple(k): g for k, g in m.groupby(["set", unit])}
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(units), len(units))
        df = pd.concat([groups[tuple(units[i])] for i in pick], ignore_index=True)
        vals.append(fn(df))
    v = np.array(vals, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]


def _tile_stats(m):
    """Per-tile sufficient statistics (non-failed rows): n, sx, sy, sxx, sxy, syy for OLS of value on truth."""
    ok = m[~m.fail]
    x, y = ok.truth.astype(float), ok.value.astype(float)
    g = pd.DataFrame({"k": list(zip(ok.set, ok.tile)), "x": x, "y": y, "xx": x * x, "xy": x * y, "yy": y * y})
    s = g.groupby("k").agg(n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"), sxx=("xx", "sum"), sxy=("xy", "sum"), syy=("yy", "sum"))
    return s


def _ols_from_sums(n, sx, sy, sxx, sxy, syy):
    mx, my = sx / n, sy / n
    vxx, vxy, vyy = sxx / n - mx * mx, sxy / n - mx * my, syy / n - my * my
    slope = vxy / vxx
    resid_var = max(vyy - slope * vxy, 0.0) * n / max(n - 2, 1)
    return slope, 3 * np.sqrt(resid_var)


def fast_bootstrap(m, what, n_boot=1000, seed=0):
    """Tile-level bootstrap 95% CI via resampled tile weights on per-tile sufficient statistics.
    what: 'tmu3s' | 'slope' (OLS of value on truth, non-failed rows) or 'prec3s' (pooled per-gauge rep variance)."""
    rng = np.random.default_rng(seed)
    if what == "prec3s":
        ok = m[~m.fail]
        gs = ok.groupby(["set", "tile", "cond", "gauge_id"]).value.agg(["var", "count"])
        gs = gs[gs["count"] >= 10]
        if not len(gs):
            return [np.nan, np.nan]
        per = gs.groupby(level=[0, 1])["var"].agg(["sum", "count"])
        S, C = per["sum"].values, per["count"].values
        vals = []
        for _ in range(n_boot):
            w = rng.multinomial(len(S), np.full(len(S), 1 / len(S)))
            vals.append(3 * np.sqrt((w * S).sum() / max((w * C).sum(), 1)))
    else:
        st = _tile_stats(m)
        if len(st) < 2:
            return [np.nan, np.nan]
        A = st[["n", "sx", "sy", "sxx", "sxy", "syy"]].values.astype(float)
        vals = []
        for _ in range(n_boot):
            w = rng.multinomial(len(A), np.full(len(A), 1 / len(A)))
            tot = (w[:, None] * A).sum(0)
            if tot[0] < 3:
                continue
            sl, tmu = _ols_from_sums(*tot)
            vals.append(tmu if what == "tmu3s" else sl)
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]


def summarize(cfg, meas, truth, method, n_boot=1000):
    """Headline tables per kind and dose. Regression uses rep 0 over the exposure-like conditions."""
    m = merged(cfg, meas, truth)
    nom = nominal_index(cfg)
    rows = []
    for (kind, dose), g in m.groupby(["kind", "dose"]):
        r0 = g[(g.rep == 0) & g.inside.notna()]
        row = dict(method=method, kind=kind, dose=int(dose))
        for nm, sub in (("all", r0), ("inside", r0[r0.inside == True]), ("outside", r0[r0.inside == False])):  # noqa: E712
            a = accuracy_regression(sub)
            row.update({f"{nm}_{k}": v for k, v in a.items()})
        for nm in ("stress_contrast", "stress_blur"):
            a = accuracy_regression(g[(g.rep == 0) & (g.cname == nm)])
            row.update({f"{nm}_{k}": a[k] for k in ("bias", "rmse", "fail_rate", "n")})
        p = precision(g[g.cond == nom])
        row.update({"prec_3s": p["three_sigma"], "prec_n_gauges": p["n_gauges"]})
        if n_boot and kind in ("cd", "epe1d", "contact"):
            ins = r0[r0.inside == True]  # noqa: E712
            row["inside_tmu3s_ci"] = fast_bootstrap(ins, "tmu3s", n_boot)
            row["inside_slope_ci"] = fast_bootstrap(ins, "slope", n_boot)
            pg = g[g.cond == nom]
            if pg.rep.max() > 0:
                row["prec_3s_ci"] = fast_bootstrap(pg, "prec3s", n_boot)
        rows.append(row)
    return pd.DataFrame(rows)


def dose_benefit(prec_by_dose, ref_3s):
    """Lowest dose (frames) at which 3 sigma <= ref_3s, interpolating log(3s) linearly in log2(frames)."""
    fr = np.array(sorted(prec_by_dose)); v = np.array([prec_by_dose[f] for f in fr], float)
    ok = np.isfinite(v)
    fr, v = fr[ok], v[ok]
    if not len(v) or v.min() > ref_3s:
        return np.nan
    if v[0] <= ref_3s:
        return float(fr[0])
    for i in range(1, len(fr)):
        if v[i] <= ref_3s:
            x0, x1, y0, y1 = np.log2(fr[i - 1]), np.log2(fr[i]), np.log(v[i - 1]), np.log(v[i])
            return float(2 ** (x0 + (np.log(ref_3s) - y0) * (x1 - x0) / (y1 - y0)))
    return np.nan


# ----------------------------------------------------------------------------------------------- roughness (PSD)
def linespace_set(cfg):
    """Small documented line/space set: vertical lines, pitch P, aerial = thr + amp*cos(2 pi x / P).
    Returns aerial crop (S+2M)^2 and design raster."""
    rc, S, M = cfg["roughness"], cfg["lithobench"]["crop_px"], cfg["sim"]["margin_px"]
    H = S + 2 * M
    x = np.arange(H, dtype=np.float64)
    a1 = cfg["sim"]["threshold_nominal"] + rc["amp"] * np.cos(2 * np.pi * (x - rc["phase_px"]) / rc["pitch_px"])
    a = np.tile(a1[None], (H, 1)).astype(np.float32)
    return a, a > cfg["sim"]["threshold_nominal"]


def psd_unbiased(series, hf_frac):
    """series (E, L) edge (or width) displacements along the edge. Detrend each, PSD = |FFT|^2 / L.
    Noise floor = mean PSD over the top hf_frac of frequencies. Returns dict raw/noise/unbiased variance, SNR."""
    y = series - series.mean(1, keepdims=True)
    L = y.shape[1]
    tt = np.arange(L) - (L - 1) / 2
    y = y - (y @ tt)[:, None] * tt[None] / (tt @ tt)
    P = (np.abs(np.fft.rfft(y, axis=1)) ** 2 / L).mean(0)
    f = np.fft.rfftfreq(L)
    w = np.where((f == 0) | (f == 0.5), 1.0, 2.0)          # one-sided weights so sum(w*P)/L = variance
    var_raw = float((w * P).sum() / L)
    hf = f >= (0.5 - 0.5 * hf_frac)
    floor = float(P[hf].mean())
    var_noise = floor * (w.sum() / L)
    var_unb = var_raw - var_noise
    snr = np.sqrt(var_unb / var_noise) if var_unb > 0 and var_noise > 0 else 0.0
    return dict(raw_3s=3 * np.sqrt(var_raw), noise_3s=3 * np.sqrt(var_noise),
                unbiased_3s=3 * np.sqrt(var_unb) if var_unb > 0 else 0.0, snr=float(snr), freq=f, psd=P)


def linespace_probes(T, border):
    pts, nrm, _ = gg.design_edge_points(T, 0, 1, border)
    v = nrm[:, 1] == 0
    pr = pd.DataFrame({"gauge_id": np.arange(v.sum()), "kind": "epe1d", "x": pts[v, 0], "y": pts[v, 1],
                       "nx": nrm[v, 0], "ny": nrm[v, 1], "design_cd": np.nan, "cx": np.nan, "cy": np.nan})
    return pr


def _edge_series(pr, t):
    """Group per-probe offsets into per-edge series (E, L) ordered by y, plus per-line width series."""
    edges, xs, ns = [], [], []
    for (x, nx), g in pr.assign(t=t).groupby(["x", "nx"]):
        g = g.sort_values("y")
        edges.append(g.t.values); xs.append(x); ns.append(nx)
    L = min(len(e) for e in edges)
    E = np.stack([e[:L] for e in edges])
    xs, ns = np.array(xs), np.array(ns)
    widths = []
    for i in np.nonzero(ns < 0)[0]:                        # left edge of a line; its right edge is the next +x edge
        j = np.nonzero((ns > 0) & (xs > xs[i]))[0]
        if len(j):
            j = j[np.argmin(xs[j])]
            widths.append(E[i] + E[j])
    return E, (np.stack(widths) if widths else np.zeros((0, L)))


def roughness_eval(cfg, method_fn, device="cuda"):
    """LER/LWR on the line/space set with ORACLE registration (periodic patterns make design registration
    ambiguous by one pitch; a constant registration offset is removed by detrending anyway).
    method_fn(img, probes, A, b, dose) -> (t_design, valid). Returns DataFrame rows per (sigma, dose)."""
    rc, sc, S, M = cfg["roughness"], cfg["sim"], cfg["lithobench"]["crop_px"], cfg["sim"]["margin_px"]
    a, T = linespace_set(cfg)
    pr = linespace_probes(T, cfg["gauges"]["border_px"])
    rows = []
    for si, sig in enumerate(rc["rough_sigma_nm"]):
        for rep in range(rc["reps"]):
            seed = 777000 + 100 * si + rep
            p = ss.sample_params(np.random.default_rng(seed), sc, 1)
            p["threshold"][:] = sc["threshold_nominal"]; p["rough_sigma_nm"][:] = sig; p["rough_corr_nm"][:] = rc["rough_corr_nm"]
            o = ss.simulate(a[None], p, sc, S, seed=seed, device=device, bumps=False)
            tt, ok = gg.truth_offsets(o["F"][0], pr, cfg["gauges"]["probe_half_window_px"])
            Et, Wt = _edge_series(pr, tt)
            truth_ler = psd_unbiased(Et, rc["hf_frac"])["raw_3s"]
            truth_lwr = psd_unbiased(Wt, rc["hf_frac"])["raw_3s"]
            for fr in sc["frames"]:
                series = []
                for nrep in (0, 1):                      # two noise realisations of the same geometry
                    img = eval_noisy(o["clean"][0], fr, seed, nrep, sc)
                    t, v = method_fn(img, pr, o["A"][0], o["b"][0], fr)
                    fail = float(1 - np.mean(v)) if nrep == 0 else fail
                    t = pd.Series(t).interpolate(limit_direction="both").values   # fill isolated failures along the edge
                    series.append(_edge_series(pr, t))
                (E, Wd), (E1, W1) = series
                le, lw = psd_unbiased(E, rc["hf_frac"]), psd_unbiased(Wd, rc["hf_frac"])
                # repeat-based noise: var(x1 - x0)/2 after per-edge detrending (no whiteness assumption)
                nl = psd_unbiased(E1 - E, rc["hf_frac"])["raw_3s"] / np.sqrt(2)
                nw = psd_unbiased(W1 - Wd, rc["hf_frac"])["raw_3s"] / np.sqrt(2)
                rc_l = np.sqrt(max(le["raw_3s"] ** 2 - nl ** 2, 0)); rc_w = np.sqrt(max(lw["raw_3s"] ** 2 - nw ** 2, 0))
                rows.append(dict(sigma_nm=sig, rep=rep, dose=fr, fail_rate=fail, truth_ler_3s=truth_ler, truth_lwr_3s=truth_lwr,
                                 ler_raw_3s=le["raw_3s"], ler_unbiased_3s=le["unbiased_3s"], ler_noise_3s=le["noise_3s"], ler_snr=le["snr"],
                                 ler_repnoise_3s=nl, ler_repcorr_3s=rc_l, ler_rep_snr=rc_l / nl if nl > 0 else np.inf,
                                 lwr_raw_3s=lw["raw_3s"], lwr_unbiased_3s=lw["unbiased_3s"], lwr_noise_3s=lw["noise_3s"], lwr_snr=lw["snr"],
                                 lwr_repnoise_3s=nw, lwr_repcorr_3s=rc_w, lwr_rep_snr=rc_w / nw if nw > 0 else np.inf))
    df = pd.DataFrame(rows)
    df["ler_unreliable"] = (df.ler_snr < rc["snr_min"]) | (df.ler_rep_snr < rc["snr_min"])
    df["lwr_unreliable"] = (df.lwr_snr < rc["snr_min"]) | (df.lwr_rep_snr < rc["snr_min"])
    return df


def lcdu(m, nom):
    """Contact LCDU at the nominal condition. Groups contacts by design CD (1 nm bins); pooled within-group
    variances. raw = 3 sd of rep-0 measurements; noise = 3 sqrt(mean per-contact variance over reps);
    corrected = 3 sqrt(raw^2 - noise^2); truth = 3 sd of true CDs of the same contacts."""
    c = m[(m.kind == "contact") & (m.cond == nom)]
    c = c.assign(dgrp=np.round(c.design_cd))
    ok = c[~c.fail]
    g = ok.groupby(["set", "tile", "gauge_id"])
    stats = g.agg(n=("value", "count"), var_rep=("value", "var"), truth=("truth", "first"), dgrp=("dgrp", "first"))
    r0 = ok[ok.rep == 0].set_index(["set", "tile", "gauge_id"]).value
    stats = stats[stats.n >= 10].join(r0.rename("meas0"), how="inner")
    if len(stats) < 3:
        return dict(n_contacts=int(len(stats)))
    num_t = num_r = den = 0.0
    for _, s in stats.groupby("dgrp"):
        if len(s) < 2:
            continue
        num_t += s.truth.var() * (len(s) - 1); num_r += s.meas0.var() * (len(s) - 1); den += len(s) - 1
    var_t, var_r, var_n = float(num_t / den), float(num_r / den), float(stats.var_rep.mean())
    return dict(n_contacts=int(len(stats)), truth_lcdu_3s=3 * np.sqrt(var_t), raw_lcdu_3s=3 * np.sqrt(var_r),
                noise_3s=3 * np.sqrt(var_n), corrected_lcdu_3s=float(3 * np.sqrt(var_r - var_n)) if var_r > var_n else 0.0,
                note="contacts grouped by design CD (1 nm bins); nominal condition; per-dose split in lcdu_by_dose")


def registration_report(cfg, split):
    rows = []
    S = cfg["lithobench"]["crop_px"]
    import registration as rg
    for s, t in eval_tiles(cfg, split):
        z = np.load(os.path.join(cfg["paths"]["results"], "raw", "cache", split, s, t + ".npz"))
        r = np.load(os.path.join(cfg["paths"]["results"], "raw", "reg", split, s, t + ".npz"))
        j = r["jobs"]
        sel = j[:, 2] == 0
        ec, em = rg.registration_error(r["A"][sel], r["b"][sel], z["A"][j[sel, 0]], z["b"][j[sel, 0]], S)
        rows.append(pd.DataFrame({"set": s, "tile": t, "cond": j[sel, 0], "dose": j[sel, 1], "centre_err": ec, "corner_max_err": em}))
    df = pd.concat(rows, ignore_index=True)
    summ = df.groupby(["set", "dose"]).agg(centre_med=("centre_err", "median"), centre_p90=("centre_err", lambda x: np.percentile(x, 90)),
                                          corner_med=("corner_max_err", "median"), corner_p90=("corner_max_err", lambda x: np.percentile(x, 90)),
                                          n=("centre_err", "size")).reset_index()
    return df, summ


def uniform_shift_demo(cfg, cparams, shift_px=3.0, n_tiles=8, device="cuda"):
    """Shift the printed pattern (truth AND image) uniformly by +shift_px in x while the design stays fixed.
    True EPE on +x / -x edges changes by +-shift_px; registration absorbs it, so measured EPE barely changes."""
    import measure as ms
    import registration as rg
    sc, gc, S, M, rc = cfg["sim"], cfg["gauges"], cfg["lithobench"]["crop_px"], cfg["sim"]["margin_px"], cfg["registration"]
    out = []
    for s, t in eval_tiles(cfg, "val")[:n_tiles // 2] + eval_tiles(cfg, "val")[-(n_tiles // 2):]:
        y0, x0 = ss.choose_crop(cfg, s, t, S)
        a, T = ss.load_crop(cfg, s, t, y0, x0, S)
        a_shift = ss.ndimage.shift(a, (0, shift_px), order=3, mode="nearest").astype(np.float32)
        pr = gg.build_gauges(T, gc, s in CONTACT_SETS, gc["border_px"])
        pr = pr[pr.kind == "epe1d"].reset_index(drop=True)
        tm, wt = rg.design_template(T[None], rc)
        res = {}
        for nm, aa in (("orig", a), ("shifted", a_shift)):
            p = ss.nominal_params(sc, 1)
            o = ss.simulate(aa[None], p, sc, S, seed=5, device=device, bumps=False, frames=[64])
            tt, ok = gg.truth_offsets(o["F"][0], pr, gc["probe_half_window_px"])
            A, b = rg.register(torch.as_tensor(o["noisy"][64], device=device)[:, None], torch.as_tensor(tm, device=device)[:, None],
                               rc, M, weight=torch.as_tensor(wt, device=device)[:, None])
            tm_, vm = ms.measure_classical(o["noisy"][64][0], pr, A[0].cpu().numpy(), b[0].cpu().numpy(), cparams["64"])
            res[nm] = (tt, ok, tm_, vm)
        sgn = pr.nx.values
        okk = res["orig"][1] & res["shifted"][1] & res["orig"][3] & res["shifted"][3] & (sgn != 0)
        dt = (res["shifted"][0] - res["orig"][0])[okk] * sgn[okk]
        dm = (res["shifted"][2] - res["orig"][2])[okk] * sgn[okk]
        out.append(dict(set=s, tile=t, n=int(okk.sum()), true_dEPE_signed=float(np.median(dt)), measured_dEPE_signed=float(np.median(dm))))
    return pd.DataFrame(out)


def tune_classical(cfg, split="val", workers=None, n_tiles_per_set=20):
    """Per dose: choose (sigma, frac) minimising mean(TMU-style CD, TMU-style contact) on INSIDE-range conditions
    (rep 0) with failure rate within +0.02 of the best; then set edge_offset_px = median epe1d error with ORACLE
    registration (calibration of the threshold offset, as a recipe would be calibrated)."""
    tiles = [(s, t) for s in ("MetalSet", "ViaSet") for t in [t for s2, t in eval_tiles(cfg, split) if s2 == s][:n_tiles_per_set]]
    tr = load_truth(cfg, split, tiles)
    inside = [i for i, c in enumerate(conditions(cfg)) if c["inside"]]
    res = []
    for sg in cfg["classical"]["tune_sigma"]:
        for fr_ in cfg["classical"]["tune_frac"]:
            cp = classical_params(cfg, {str(f): {"sigma": sg, "frac": fr_} for f in cfg["sim"]["frames"]})
            meas = run_classical(cfg, split, cp, "est", workers, tiles, rep0_only=True, label=f"tune s={sg} f={fr_}")
            m = merged(cfg, meas[meas.rep == 0], tr)
            m = m[m.cond.isin(inside)]
            for dose, g in m.groupby("dose"):
                a_cd = accuracy_regression(g[g.kind == "cd"]); a_ct = accuracy_regression(g[g.kind == "contact"])
                res.append(dict(sigma=sg, frac=fr_, dose=int(dose), cd_tmu3s=a_cd["tmu3s"], contact_tmu3s=a_ct["tmu3s"],
                                fail=float(g[g.kind.isin(["cd", "contact"])].fail.mean())))
    df = pd.DataFrame(res)
    df["score"] = df[["cd_tmu3s", "contact_tmu3s"]].mean(1)
    tuned = {}
    for dose, g in df.groupby("dose"):
        g = g[g.fail <= g.fail.min() + 0.02]
        b = g.loc[g.score.idxmin()]
        tuned[str(int(dose))] = {"sigma": float(b.sigma), "frac": float(b.frac)}
    cp = classical_params(cfg, tuned)
    meas = run_classical(cfg, split, cp, "oracle", workers, tiles, label="offset-calibration")
    m = merged(cfg, meas[meas.rep == 0], tr)
    m = m[m.cond.isin(inside) & (m.kind == "epe1d") & ~m.fail]
    for dose, g in m.groupby("dose"):
        tuned[str(int(dose))]["edge_offset_px"] = float(g.err.median())
    return tuned, df


def gds_sample(cfg, cparams, split="val", n_tiles=3):
    """Export truth contours (layer 1) and classical-measured contact polygons / EPE points (layer 2) to GDS
    in design (tile) coordinates and check the geometry round-trips."""
    import measure as ms
    out_dir = os.path.join(cfg["paths"]["results"], "gds")
    os.makedirs(out_dir, exist_ok=True)
    rep = []
    tiles = [x for x in eval_tiles(cfg, split) if x[0] == "ViaSet"][:1] + [x for x in eval_tiles(cfg, split) if x[0] == "MetalSet"][:n_tiles - 1]
    nom = nominal_index(cfg)
    for s, t in tiles:
        d = load_cache(os.path.join(cfg["paths"]["results"], "raw", "cache", split, s, t + ".npz"))
        F = d["F"][nom].astype(np.float32)
        img = eval_noisy(d["clean"][nom], 64, int(d["seed"]) * 100 + nom, 0, cfg["sim"])
        pr = d["probes"]
        tm, vm = ms.measure_classical(img, pr, d["A"][nom], d["b"][nom], cparams["64"])
        tm = tm - cparams["64"].get("edge_offset_px", 0.0)
        meas_c = []
        for gid, g in pr.assign(t=tm, v=vm).groupby("gauge_id"):
            g = g[g.v]
            if len(g) >= 3 and g.kind.iloc[0] == "contact":
                pts = g[["x", "y"]].values + g[["nx", "ny"]].values * g.t.values[:, None]
                meas_c.append(np.vstack([pts, pts[:1]]))
        layers = {1: gg.contours_from_field(F), 2: meas_c}
        path = os.path.join(out_dir, f"{s}_{t}.gds")
        gg.export_gds(path, layers, tuple(d["origin"]))
        ok, md, n = gg.gds_roundtrip_check(path, layers, tuple(d["origin"]))
        rep.append(dict(file=os.path.basename(path), roundtrip_ok=bool(ok), max_vertex_diff_nm=md, n_vertices=n,
                        n_truth_contours=len(layers[1]), n_measured_contacts=len(meas_c)))
    return pd.DataFrame(rep)


def refine_params_for(cfg, dose):
    """Frozen (not tuned) learned+refine params: classical per-dose sigma/frac/edge offset from M3 tuning,
    window w_in/w_out from cfg['refine'] around the learned edge."""
    tuned = json.load(open(os.path.join(cfg["paths"]["results"], "classical_baseline", "classical_tuning.json")))[str(int(dose))]
    rp = {k: cfg["classical"][k] for k in ("step", "min_prom")}
    rp.update(sigma=tuned["sigma"], frac=tuned["frac"], edge_offset_px=tuned["edge_offset_px"],
              w_in=cfg["refine"]["w_in"], w_out=cfg["refine"]["w_out"], peak_inset_px=cfg["refine"]["peak_inset_px"])
    return rp


def run_learned(cfg, run_name, split, reg_mode="est", device="cuda", tiles=None, max_jobs=None):
    """Evaluate checkpoints/<run_name>/best.pt on every cached job of `split`.
    Saves meas/<split>/<run_name>[_oraclereg]__learned.pkl.gz and __refine.pkl.gz (same columns as classical).
    Also returns inference timing (median ms / image, batch 8, bf16) for the operator table."""
    import measure as ms
    import train as tr
    from models import build_model
    from scipy import ndimage
    ck = torch.load(os.path.join(os.path.dirname(cfg["paths"]["results"]), "checkpoints", run_name, "best.pt"),
                    map_location=device, weights_only=False)
    run = ck["run"]
    model = build_model(cfg["model"], run["op"], 2 if run["cad"] else 1).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    sc, S, trunc, hw = cfg["sim"], cfg["lithobench"]["crop_px"], cfg["model"]["tsdf_trunc_px"], cfg["gauges"]["probe_half_window_px"]
    rps = {fr: refine_params_for(cfg, fr) for fr in sc["frames"]}
    rows = {"learned": [], "learned+refine": []}
    times = []
    t0 = time.time()
    tiles = tiles or eval_tiles(cfg, split)
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(cfg["eval"]["noise_threads"])      # CPU noise in parallel threads (torch releases the GIL)
    for ti, (s, t) in enumerate(tiles):
        d = load_cache(os.path.join(cfg["paths"]["results"], "raw", "cache", split, s, t + ".npz"))
        agg = gg.GaugeAggregator(d["probes"])
        r = np.load(os.path.join(cfg["paths"]["results"], "raw", "reg", split, s, t + ".npz"))
        jobs = [tuple(j) for j in r["jobs"].tolist()]
        if reg_mode == "oracle":
            jobs = [j for j in jobs if j[2] == 0]
        jobs = jobs[:max_jobs] if max_jobs else jobs
        jidx = {tuple(j): i for i, j in enumerate(r["jobs"].tolist())}
        for i in range(0, len(jobs), 8):
            chunk = jobs[i:i + 8]
            imgs, cads, regs = list(pool.map(lambda j: eval_noisy(d["clean"][j[0]], j[1], int(d["seed"]) * 100 + j[0], j[2], sc), chunk)), [], []
            for c, fr, rep in chunk:
                A, b = (d["A"][c], d["b"][c]) if reg_mode == "oracle" else (r["A"][jidx[(c, fr, rep)]], r["b"][jidx[(c, fr, rep)]])
                regs.append((A, b))
                if run["cad"]:
                    # CAD input always uses the ESTIMATED registration (as in training); oracle only changes the measurement mapping
                    j = jidx[(c, fr, rep)]
                    cads.append(tr.cad_in_sem(d["T"], r["A"][j], r["b"][j], S, device))
            torch.cuda.synchronize(); ta = time.perf_counter()
            tsdf = tr.predict_tsdf(model, imgs, cads if run["cad"] else None, device) * trunc
            torch.cuda.synchronize(); times.append((time.perf_counter() - ta) * 1000 / len(chunk))
            for k, (c, fr, rep) in enumerate(chunk):
                rp = rps[fr]
                sm = torch.as_tensor(ndimage.gaussian_filter(imgs[k].astype(np.float64), rp["sigma"]), device=device)
                res = ms.measure_learned_gpu(torch.as_tensor(tsdf[k], device=device), sm, d["probes"], regs[k][0], regs[k][1], hw, rp, device)
                for name, (tt, vv) in res.items():
                    gv = agg.values(tt, vv)
                    rows[name].append(pd.DataFrame({"gauge_id": gv.gauge_id.values.astype(np.int32), "kind": gv.kind.values,
                                                    "value": gv.value.values.astype(np.float32), "valid": gv.valid.values.astype(bool),
                                                    "cond": np.int16(c), "dose": np.int16(fr), "rep": np.int16(rep), "set": s, "tile": t}))
        if (ti + 1) % 10 == 0 or ti + 1 == len(tiles):
            print(f"  run-learned {run_name} {split} {reg_mode} {ti + 1}/{len(tiles)} tiles {time.time() - t0:.0f}s", flush=True)
    suffix = "" if reg_mode == "est" else "_oraclereg"
    out = {}
    for name, key in (("learned", "learned"), ("learned+refine", "refine")):
        df = pd.concat(rows[name], ignore_index=True)
        save_meas(df, cfg, split, f"{run_name}{suffix}__{key}")
        out[name] = df
    return out, float(np.median(times[1:] if len(times) > 1 else times))


def _sel_stats(cfg, meas, truth, dose):
    """Per-tile sufficient statistics for the frozen selection metric (inside conditions, rep 0, given dose)."""
    m = merged(cfg, meas[(meas.rep == 0) & (meas.dose == dose)], truth)
    m = m[m.inside == True]  # noqa: E712
    out = {}
    for k in ("cd", "contact"):
        out[k] = _tile_stats(m[m.kind == k])
    fail = m[m.kind.isin(["cd", "contact"])].groupby(["set", "tile"]).fail.agg(["sum", "count"])
    return out, fail


def _sel_value(stats, w=None):
    vals = []
    for k, st in stats.items():
        A = st[["n", "sx", "sy", "sxx", "sxy", "syy"]].values.astype(float)
        tot = (A if w is None else w.reindex(st.index).fillna(0).values[:, None] * A).sum(0)
        vals.append(_ols_from_sums(*tot)[1])
    return float(np.mean(vals))


def screening_table(cfg, names, split="val", dose=None, n_boot=1000, seed=0):
    """Frozen selection metric on the FULL split for each meas name (e.g. screen_dcnv3_cad_s1__learned):
    mean of TMU-style CD and contact (inside conditions, rep 0, selection dose), tile-bootstrap CI, failure rate,
    and a PAIRED bootstrap of the difference to the best name (same resampled tiles)."""
    dose = dose or cfg["eval"]["selection_dose"]
    tr = load_truth(cfg, split)
    stats = {n: _sel_stats(cfg, load_meas(cfg, split, n), tr, dose) for n in names}
    tiles = sorted(set().union(*[set(st["cd"].index) | set(st["contact"].index) for st, _ in stats.values()]))
    rng = np.random.default_rng(seed)
    boot = {n: [] for n in names}
    for _ in range(n_boot):
        w = pd.Series(rng.multinomial(len(tiles), np.full(len(tiles), 1 / len(tiles))), index=pd.Index(tiles))
        for n in names:
            boot[n].append(_sel_value(stats[n][0], w))
    rows = []
    for n in names:
        st, fail = stats[n]
        rows.append(dict(name=n, sel_metric=_sel_value(st), sel_ci=[float(np.percentile(boot[n], 2.5)), float(np.percentile(boot[n], 97.5))],
                         cd_tmu3s=_sel_value({"cd": st["cd"]}), contact_tmu3s=_sel_value({"contact": st["contact"]}),
                         fail_rate=float(fail["sum"].sum() / fail["count"].sum())))
    df = pd.DataFrame(rows)
    best = df.loc[df.sel_metric.idxmin(), "name"]
    for i, n in enumerate(df.name):
        d = np.array(boot[n]) - np.array(boot[best])
        df.loc[i, "diff_vs_best"] = float(df.loc[i, "sel_metric"] - df.sel_metric.min())
        df.loc[i, "diff_ci_lo"], df.loc[i, "diff_ci_hi"] = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        df.loc[i, "indistinguishable_from_best"] = bool(np.percentile(d, 2.5) <= 0 <= np.percentile(d, 97.5)) if n != best else True
    df["best"] = best
    return df


def m3_report(cfg, workers=None):
    """M3 checkpoint: classical tuning (val), classical val summary (estimated + oracle registration),
    registration recovery, uniform-shift demonstration, roughness, LCDU, GDS sample."""
    out = os.path.join(cfg["paths"]["results"], "classical_baseline")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    tj = os.path.join(out, "classical_tuning.json")
    if os.path.exists(tj):                      # tuning is frozen once written (validation only)
        tuned = json.load(open(tj))
    else:
        tuned, grid = tune_classical(cfg, "val", workers)
        grid.to_csv(os.path.join(out, "classical_tuning_grid.csv"), index=False)
        json.dump(tuned, open(tj, "w"), indent=1)
    print("tuned classical:", tuned, f"{time.time() - t0:.0f}s", flush=True)
    cp = classical_params(cfg, tuned)
    tr = load_truth(cfg, "val")
    for reg_mode in ("est", "oracle"):
        meas = run_classical(cfg, "val", cp, reg_mode, workers, label="val")
        save_meas(meas, cfg, "val", f"classical_{reg_mode}")
        sm = summarize(cfg, meas, tr, f"classical[{reg_mode}-reg]", n_boot=1000 if reg_mode == "est" else 0)
        sm.to_csv(os.path.join(out, f"classical_val_summary_{reg_mode}reg.csv"), index=False)
        if reg_mode == "est":
            m = merged(cfg, meas, tr)
            lc = {str(int(dz)): lcdu(g, nominal_index(cfg)) for dz, g in m.groupby("dose")}
            json.dump(lc, open(os.path.join(out, "lcdu_classical_val.json"), "w"), indent=1, default=float)
        print(f"classical val {reg_mode}-reg done {time.time() - t0:.0f}s", flush=True)
    reg_rows, reg_sum = registration_report(cfg, "val")
    reg_sum.to_csv(os.path.join(out, "registration_val.csv"), index=False)
    uniform_shift_demo(cfg, cp).to_csv(os.path.join(out, "uniform_shift_demo.csv"), index=False)
    import measure as ms

    def cls_fn(img, pr, A, b, fr):
        t, v = ms.measure_classical(img, pr, A, b, cp[str(fr)])
        return t - cp[str(fr)].get("edge_offset_px", 0.0), v
    roughness_eval(cfg, cls_fn).to_csv(os.path.join(out, "roughness_classical.csv"), index=False)
    gds_sample(cfg, cp).to_csv(os.path.join(out, "gds_roundtrip.csv"), index=False)
    print(f"m3 report done {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------------------- CLI
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cache", "register", "baseline", "run-learned"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--run", help="run-learned: eval job yaml {name, kind: eval, ckpt: <train run name>, split, reg: est|oracle}")
    a = ap.parse_args()
    cfg = load_config()
    if a.step == "baseline":
        return m3_report(cfg, a.workers)
    if a.step == "run-learned":
        import yaml
        job = yaml.safe_load(open(a.run))
        torch.cuda.set_per_process_memory_fraction(float(job.get("mem_fraction", cfg["queue"]["mem_fraction"])))
        _, ms_img = run_learned(cfg, job["ckpt"], job["split"], job.get("reg", "est"))
        d = os.path.join(cfg["paths"]["results"], "raw", "meas", job["split"])
        json.dump({"ckpt": job["ckpt"], "split": job["split"], "reg": job.get("reg", "est"), "infer_ms_per_img_bs8": ms_img},
                  open(os.path.join(d, f"{job['ckpt']}{'' if job.get('reg', 'est') == 'est' else '_oraclereg'}__timing.json"), "w"))
        return
    tiles = eval_tiles(cfg, a.split)
    t0 = time.time()
    for i, (s, t) in enumerate(tiles):
        (cache_tile if a.step == "cache" else register_tile)(cfg, a.split, s, t)
        if i % 25 == 0:
            print(f"{a.step} {a.split} {i + 1}/{len(tiles)} {time.time() - t0:.0f}s", flush=True)
    print(f"{a.step} {a.split} done {len(tiles)} tiles {time.time() - t0:.0f}s")


if __name__ == "__main__":
    _cli()
