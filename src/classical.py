"""Classical baseline: Gaussian filtering + bloom-peak search + outer-slope threshold edge along each normal.

`refine_edges` is the 1D profile refinement shared by the classical method, the learned+refinement method
(measure.py) and the C++/CUDA port (cuda/). All coordinates here are SEM px; t is the offset along the unit
normal (+ = outward from the feature).
"""
import numpy as np
from scipy import ndimage


def smooth(img, sigma):
    return ndimage.gaussian_filter(np.asarray(img, np.float64), sigma) if sigma > 0 else np.asarray(img, np.float64)


def sample_profiles(img, pts, normals, ts):
    """Bilinear samples (N, K) of img at pts + t * normal for t in ts. Outside the image: edge value (clamped)."""
    xs = pts[:, 0:1] + normals[:, 0:1] * ts[None]
    ys = pts[:, 1:2] + normals[:, 1:2] * ts[None]
    return ndimage.map_coordinates(img, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(xs.shape)


def refine_edges(img, pts, normals, t0, rp):
    """Sub-pixel edge refinement around an initial estimate t0 (N,).

    Algorithm (identical in cuda/edge_refine.cu):
      1. sample K = (w_in + w_out)/step + 1 points at t = t0 - w_in + k*step (bilinear)
      2. peak = argmax over samples with t <= t0 + w_in (sample index; no sub-sample peak refinement)
      3. base = min of samples from the peak to the outer end
      4. level = base + frac * (peak - base); first sample after the peak with value < level; linear interpolation
      5. valid = (peak - base >= min_prom) and crossing found and peak not on the first sample
    Returns t (N,) float64 (nan where invalid) and valid (N,) bool.
    """
    w_in, w_out, step, frac, min_prom = rp["w_in"], rp["w_out"], rp["step"], rp["frac"], rp["min_prom"]
    K = int(round((w_in + w_out) / step)) + 1
    rel = -w_in + step * np.arange(K)
    N = len(pts)
    ts = (np.asarray(t0, np.float64)[:, None] + rel[None])
    xs = pts[:, 0:1] + normals[:, 0:1] * ts
    ys = pts[:, 1:2] + normals[:, 1:2] * ts
    v = ndimage.map_coordinates(img, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(N, K)
    kin = int(round(2 * w_in / step)) + 1
    kp = np.argmax(v[:, :kin], 1)
    r = np.arange(N)
    vp = v[r, kp]
    tail = np.where(np.arange(K)[None] >= kp[:, None], v, np.inf)
    base = tail.min(1)
    level = base + frac * (vp - base)
    below = (np.arange(K)[None] > kp[:, None]) & (v < level[:, None])
    has = below.any(1)
    kc = np.where(has, np.argmax(below, 1), 1)
    va, vb = v[r, kc - 1], v[r, kc]
    tc = rel[kc - 1] + step * (va - level) / np.where(va - vb != 0, va - vb, 1.0)
    valid = has & (vp - base >= min_prom) & (kp > 0)
    t = np.where(valid, t0 + tc, np.nan)
    return t, valid


def coarse_peaks(img, pts, normals, cp):
    """Stage 1: local maximum of the profile nearest t = 0 within +-W with prominence >= min_prom.
    Returns t_peak (N,) (nan if none)."""
    ts = np.arange(-cp["search_px"], cp["search_px"] + 1e-9, cp["step"])
    v = sample_profiles(img, pts, normals, ts)
    lm = np.zeros_like(v, bool)
    lm[:, 1:-1] = (v[:, 1:-1] >= v[:, :-2]) & (v[:, 1:-1] > v[:, 2:])
    rad = max(1, int(round(cp["prom_radius_px"] / cp["step"])))
    lo = ndimage.minimum_filter1d(v, 2 * rad + 1, axis=1, mode="nearest")
    lm &= (v - lo) >= cp["min_prom"]
    d = np.where(lm, np.abs(ts)[None], np.inf)
    k = np.argmin(d, 1)
    return np.where(np.isfinite(d[np.arange(len(k)), k]), ts[k], np.nan)


def measure_edges(sem, sem_pts, sem_nrm, cp):
    """Classical edge offsets in SEM px for probes already mapped to SEM coords. cp = per-dose params dict
    (sigma, frac, ...). Returns t (N,), valid (N,)."""
    img = smooth(sem, cp["sigma"])
    tp = coarse_peaks(img, sem_pts, sem_nrm, cp)
    ok = np.isfinite(tp)
    t = np.full(len(sem_pts), np.nan)
    v = np.zeros(len(sem_pts), bool)
    if ok.any():
        # refinement window centred on the bloom peak: the edge lies on its outer slope
        t[ok], v[ok] = refine_edges(img, sem_pts[ok], sem_nrm[ok], tp[ok], cp)
    return t, v
