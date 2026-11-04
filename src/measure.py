"""Image + gauges -> measurements, for every method, using one shared registration.

Probes live in design-crop px; they are mapped to SEM px with the ESTIMATED registration (A, b): p = A q + b.
An offset t_sem along the mapped SEM normal is converted back to a design-normal offset with
t_design = t_sem * (n . A n_sem), which is exact for the affine model.
"""
import numpy as np

import classical


def probes_to_sem(probes, A, b):
    P = probes[["x", "y"]].values
    N = probes[["nx", "ny"]].values
    Ai = np.linalg.inv(A)
    q = (Ai @ (P - b).T).T
    ns = (Ai @ N.T).T
    ns /= np.linalg.norm(ns, axis=1, keepdims=True)
    scale = np.sum(N * (A @ ns.T).T, 1)
    return q, ns, scale


def measure_classical(sem, probes, A, b, cp):
    q, ns, scale = probes_to_sem(probes, A, b)
    t, v = classical.measure_edges(sem, q, ns, cp)
    return t * scale, v


def tsdf_crossing(tsdf, q, ns, half_window, step=0.05):
    """Zero crossing of a predicted TSDF (+ inside) along the SEM normal nearest t = 0 (bilinear, as truth)."""
    from sem_sim import crossings_along
    t, nc = crossings_along(tsdf, q, ns, half_window, step, order=1)
    return t, np.isfinite(t)


def _torch_sample(img, xs, ys):
    """Bilinear sampling with clamped coordinates (== map_coordinates order=1 mode='nearest'). img (H,W) torch."""
    import torch
    H, W = img.shape
    xs = xs.clamp(0, W - 1); ys = ys.clamp(0, H - 1)
    x0 = xs.floor().clamp(max=W - 2); y0 = ys.floor().clamp(max=H - 2)
    fx, fy = xs - x0, ys - y0
    x0, y0 = x0.long(), y0.long()
    v00, v01 = img[y0, x0], img[y0, x0 + 1]
    v10, v11 = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy


def torch_crossings(field, q, ns, half_window, step=0.05):
    """GPU version of sem_sim.crossings_along(order=1): offset to the zero crossing nearest t=0 (nan if none)."""
    import torch
    ts = torch.arange(-half_window, half_window + 1e-9, step, device=field.device, dtype=torch.float64)
    xs = q[:, :1] + ns[:, :1] * ts[None]; ys = q[:, 1:] + ns[:, 1:] * ts[None]
    v = _torch_sample(field.double(), xs, ys)
    s = torch.signbit(v)
    ch = s[:, 1:] != s[:, :-1]
    t0 = ts[:-1][None] + step * v[:, :-1] / (v[:, :-1] - v[:, 1:] + 1e-30)
    t0 = torch.where(ch, t0, torch.full_like(t0, float("inf")))
    k = t0.abs().argmin(1)
    best = t0.gather(1, k[:, None])[:, 0]
    return torch.where(torch.isfinite(best), best, torch.full_like(best, float("nan")))


def torch_refine_edges(img, q, ns, t0, rp):
    """GPU float64 port of classical.refine_edges (same algorithm; parity-tested in tests)."""
    import torch
    K = int(round((rp["w_in"] + rp["w_out"]) / rp["step"])) + 1
    rel = -rp["w_in"] + rp["step"] * torch.arange(K, device=img.device, dtype=torch.float64)
    ts = t0[:, None] + rel[None]
    v = _torch_sample(img.double(), q[:, :1] + ns[:, :1] * ts, q[:, 1:] + ns[:, 1:] * ts)
    kin = int(round(2 * rp["w_in"] / rp["step"])) + 1
    kp = v[:, :kin].argmax(1)
    vp = v.gather(1, kp[:, None])[:, 0]
    ar = torch.arange(K, device=img.device)[None]
    base = torch.where(ar >= kp[:, None], v, torch.full_like(v, float("inf"))).min(1).values
    level = base + rp["frac"] * (vp - base)
    below = (ar > kp[:, None]) & (v < level[:, None])
    has = below.any(1)
    kc = torch.where(has, below.float().argmax(1), torch.ones_like(kp))
    va, vb = v.gather(1, (kc - 1)[:, None])[:, 0], v.gather(1, kc[:, None])[:, 0]
    den = torch.where(va - vb != 0, va - vb, torch.ones_like(va))
    tc = rel[kc - 1] + rp["step"] * (va - level) / den
    valid = has & (vp - base >= rp["min_prom"]) & (kp > 0)
    return torch.where(valid, t0 + tc, torch.full_like(tc, float("nan"))), valid


def measure_learned_gpu(tsdf, sem_smooth, probes, A, b, half_window, refine_params, device="cuda"):
    """GPU learned-only and learned+refine measurement. tsdf, sem_smooth: (S,S) torch on device (px units /
    pre-filtered SEM). Returns {"learned": (t, v), "learned+refine": (t, v)} as numpy, design-normal offsets."""
    import torch
    q, ns, scale = probes_to_sem(probes, A, b)
    qt = torch.as_tensor(q, device=device, dtype=torch.float64)
    nt = torch.as_tensor(ns, device=device, dtype=torch.float64)
    t = torch_crossings(tsdf, qt, nt, half_window)
    v = torch.isfinite(t)
    out = {"learned": (t.cpu().numpy() * scale, v.cpu().numpy())}
    if refine_params is not None:
        t2 = torch.full_like(t, float("nan")); v2 = torch.zeros_like(v)
        if v.any():
            r, rv = torch_refine_edges(sem_smooth, qt[v], nt[v], t[v] - refine_params["peak_inset_px"], refine_params)
            t2[v], v2[v] = r - refine_params.get("edge_offset_px", 0.0), rv
        out["learned+refine"] = (t2.cpu().numpy() * scale, v2.cpu().numpy())
    return out


def measure_learned(tsdf, sem, probes, A, b, half_window, refine_params=None, sigma=0.0):
    """Learned-only (TSDF zero crossing) and, if refine_params given, learned + 1D SEM profile refinement
    within a small window around the learned estimate. Returns dict name -> (t_design, valid)."""
    q, ns, scale = probes_to_sem(probes, A, b)
    t, v = tsdf_crossing(tsdf, q, ns, half_window)
    out = {"learned": (t * scale, v)}
    if refine_params is not None:
        img = classical.smooth(sem, sigma)
        t2 = np.full(len(q), np.nan); v2 = np.zeros(len(q), bool)
        if v.any():
            # centre the refinement window on the learned edge shifted inward by the expected peak-to-edge distance
            c = t[v] - refine_params["peak_inset_px"]
            t2[v], v2[v] = classical.refine_edges(img, q[v], ns[v], c, refine_params)
        out["learned+refine"] = (t2 * scale, v2)
    return out
