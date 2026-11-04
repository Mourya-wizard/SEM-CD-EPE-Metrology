"""Controlled SEM forward model with exact truth (NOT a physical electron-matter simulator).

Coordinates: design crop arrays are [row=y, col=x], 1 nm/px, pixel centres at integer coords.
A design crop of size S+2M (M = margin) holds the truth field F; the SEM image is S x S.
SEM pixel (u=col, v=row) maps to design-crop coords  p = A @ [u, v] + b   (A 2x2, b 2, in px; includes margin).
Truth contour = zero level set of the bilinear interpolant of F  (F > 0 inside printed feature).
All batched functions take torch tensors on one device; params is a dict of (B,) tensors.
"""
import os

os.environ.setdefault("HF_HOME", "./.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "./.cache/torch")
os.environ.setdefault("XDG_CACHE_HOME", "./.cache")
os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")

import math

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import ndimage

PARAM_KEYS = ("threshold", "level_out", "contrast", "bloom_amp", "bloom_width_px", "bloom_offset_px", "blur_px",
              "rough_sigma_nm", "rough_corr_nm", "scale_x", "scale_y", "rot_rad", "shear", "shift_x", "shift_y")


# ----------------------------------------------------------------------------------------------- aerial field
def hybrid_aerial(litho, resist, sc):
    """litho, resist: float arrays in [0,1] (uint8/255). Returns smoothed float32 aerial estimate."""
    a = litho.astype(np.float64)
    r = np.clip(resist.astype(np.float64) * 255.0, 0.5, 254.5)
    lo, hi = sc["resist_band"]
    w = np.clip((np.minimum(r - lo, hi - r)) / sc["resist_ramp"], 0.0, 1.0)
    ar = np.log(r / (255.0 - r)) / sc["resist_steepness"] + sc["threshold_nominal"]
    h = w * ar + (1.0 - w) * a
    return ndimage.gaussian_filter(h, sc["field_smooth_px"]).astype(np.float32)


# ----------------------------------------------------------------------------------------------- params
def sample_params(rng, sc, n, mode="train", threshold=None):
    """rng: np.random.Generator. mode: train | stress_contrast | stress_blur | nominal. Returns dict of np arrays."""
    tr = sc["train"]
    u = lambda lo_hi: rng.uniform(lo_hi[0], lo_hi[1], n)
    p = {"threshold": u(sc["threshold_train"]) if threshold is None else np.full(n, float(threshold)),
         "level_out": u(tr["level_out"]), "contrast": u(tr["contrast"]), "bloom_amp": u(tr["bloom_amp"]),
         "bloom_width_px": u(tr["bloom_width_px"]), "bloom_offset_px": u(tr["bloom_offset_px"]),
         "blur_px": u(tr["blur_px"]), "rough_sigma_nm": u(tr["rough_sigma_nm"]),
         "rough_corr_nm": u(tr["rough_corr_nm"]), "scale_x": u(tr["scale"]), "scale_y": u(tr["scale"]),
         "rot_rad": u(tr["rot_rad"]), "shear": u(tr["shear"]),
         "shift_x": u(tr["shift_px"]), "shift_y": u(tr["shift_px"])}
    if mode == "stress_contrast":
        p["contrast"] = u(sc["stress_contrast"])
    elif mode == "stress_blur":
        p["blur_px"] = u(sc["stress_blur_px"])
    elif mode != "train":
        raise ValueError(mode)
    return p


def to_torch(p, device):
    return {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device) for k, v in p.items()}


# ----------------------------------------------------------------------------------------------- helpers
def _grad_mag(F):
    gy = (F[..., 2:, 1:-1] - F[..., :-2, 1:-1]) * 0.5
    gx = (F[..., 1:-1, 2:] - F[..., 1:-1, :-2]) * 0.5
    return Fn.pad(torch.sqrt(gx * gx + gy * gy), (1, 1, 1, 1), mode="replicate")


def _blur(x, sigma):
    """Per-sample separable Gaussian blur. x (B,1,H,W), sigma (B,) px."""
    B = x.shape[0]
    rad = int(math.ceil(3.5 * float(sigma.max()))) if B else 1
    t = torch.arange(-rad, rad + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t[None] / sigma[:, None].clamp(min=1e-3)) ** 2)
    k = k / k.sum(1, keepdim=True)
    y = x.reshape(1, B, *x.shape[-2:])
    y = Fn.conv2d(Fn.pad(y, (rad, rad, 0, 0), mode="reflect"), k[:, None, None, :], groups=B)
    y = Fn.conv2d(Fn.pad(y, (0, 0, rad, rad), mode="reflect"), k[:, None, :, None], groups=B)
    return y.reshape(x.shape)


def affine_of(p, S, M):
    """SEM px -> design-crop px. Returns A (B,2,2), b (B,2). Distortion is about the image centre."""
    c = (S - 1) / 2.0
    cr, sr = torch.cos(p["rot_rad"]), torch.sin(p["rot_rad"])
    sx, sy, sh = 1 + p["scale_x"], 1 + p["scale_y"], p["shear"]
    A = torch.stack([torch.stack([sx * cr, -sr + sh], -1), torch.stack([sr, sy * cr], -1)], -2)
    cvec = torch.full_like(torch.stack([p["shift_x"], p["shift_y"]], -1), c)
    b = cvec + M + torch.stack([p["shift_x"], p["shift_y"]], -1) - (A @ cvec[..., None])[..., 0]
    return A, b


# ----------------------------------------------------------------------------------------------- truth
def build_truth(aerial, p, sc, gen, bumps=True):
    """aerial (B,1,H,W) design-crop hybrid aerial. Returns F (B,1,H,W) and list of bump arrays (k,4) [y,x,shift,rad]."""
    B, _, H, W = aerial.shape
    F0 = aerial - p["threshold"][:, None, None, None]
    g0 = _grad_mag(aerial)
    # correlated roughness: white noise * Gaussian kernel (sigma_f = corr/sqrt(2) -> autocorr exp(-r^2/(2 corr^2)))
    noise = torch.randn(B, 1, H, W, generator=gen, device=aerial.device)
    sf = p["rough_corr_nm"] / math.sqrt(2.0)
    n = _blur(noise, sf)
    n = n / n.flatten(1).std(1)[:, None, None, None].clamp(min=1e-8)
    F = F0 - p["rough_sigma_nm"][:, None, None, None] * n * g0      # edge moves outward by -sigma*n... sign irrelevant
    bump_list = []
    if bumps:
        yy = torch.arange(H, device=aerial.device, dtype=aerial.dtype)[:, None]
        xx = torch.arange(W, device=aerial.device, dtype=aerial.dtype)[None, :]
        kmin, kmax = sc["bumps_per_crop"]
        for i in range(B):
            k = int(torch.randint(kmin, kmax + 1, (1,), generator=gen, device=aerial.device))
            m = (F0[i, 0].abs() < 0.5 * g0[i, 0]) & (g0[i, 0] > sc["sd_eps"])   # within ~0.5 px of nominal edge
            m[: sc["margin_px"] * 2] = False; m[-sc["margin_px"] * 2:] = False
            m[:, : sc["margin_px"] * 2] = False; m[:, -sc["margin_px"] * 2:] = False
            idx = torch.nonzero(m.flatten()).flatten()
            rows = []
            for _ in range(k if len(idx) else 0):
                j = idx[torch.randint(len(idx), (1,), generator=gen, device=aerial.device)]
                cy, cx = (j // W).float(), (j % W).float()
                lo, hi = sc["bump_shift_nm"]; rlo, rhi = sc["bump_radius_px"]
                r2 = torch.rand(2, generator=gen, device=aerial.device)
                dsh, rad = lo + (hi - lo) * r2[0], rlo + (rhi - rlo) * r2[1]
                amp = dsh * g0[i, 0].flatten()[j]
                F[i, 0] += amp * torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2))
                rows.append(torch.stack([cy.flatten()[0], cx.flatten()[0], dsh, rad]))
            bump_list.append(torch.stack(rows).cpu().numpy() if rows else np.zeros((0, 4), np.float32))
    else:
        bump_list = [np.zeros((0, 4), np.float32) for _ in range(B)]
    return F, bump_list


def signed_distance(F, sc):
    """First-order signed distance in px (+ inside feature), clipped."""
    d = F / torch.sqrt(_grad_mag(F) ** 2 + sc["sd_eps"] ** 2)
    return d.clamp(-sc["sd_clip_px"], sc["sd_clip_px"])


# ----------------------------------------------------------------------------------------------- image formation
def render_clean(F, p, sc, S):
    """Noise-free SEM image (B,1,S,S) in emission-yield units, plus affine (A, b)."""
    M = sc["margin_px"]
    d = signed_distance(F, sc)
    v = lambda k: p[k][:, None, None, None]
    img = v("level_out") + v("contrast") * torch.sigmoid(d / sc["edge_softness_px"])
    img = img + v("bloom_amp") * torch.exp(-0.5 * ((d - v("bloom_offset_px")) / v("bloom_width_px")) ** 2)
    A, b = affine_of(p, S, M)
    H, W = F.shape[-2:]
    uu, vv = torch.meshgrid(torch.arange(S, device=F.device, dtype=F.dtype),
                            torch.arange(S, device=F.device, dtype=F.dtype), indexing="xy")
    uv = torch.stack([uu, vv], -1).reshape(1, -1, 2, 1)
    xy = (A[:, None] @ uv)[..., 0] + b[:, None]                      # (B, S*S, 2) design-crop px
    grid = torch.stack([2 * xy[..., 0] / (W - 1) - 1, 2 * xy[..., 1] / (H - 1) - 1], -1).reshape(-1, S, S, 2)
    warped = Fn.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return _blur(warped, p["blur_px"]), A, b


def add_noise(clean, frames, sc, gen):
    """Poisson shot noise for `frames` frame-averaged acquisition plus Gaussian detector noise."""
    lam = clean.clamp(min=0) * (frames * sc["electrons_per_frame"])
    counts = torch.poisson(lam, generator=gen)
    out = counts / (frames * sc["electrons_per_frame"])
    return out + sc["detector_sigma"] * torch.randn(clean.shape, generator=gen, device=clean.device)


def noise_seed(base_seed, frames, rep):
    """Deterministic seed for a noise realisation of a cached clean image."""
    return (int(base_seed) * 1000003 + int(frames) * 7919 + int(rep)) % (2 ** 62)


def simulate(aerial_crop, p_np, sc, S, seed, device="cuda", bumps=True, frames=None):
    """One-call sample: aerial_crop (B,H,W) np, p_np dict of np (B,). Exactly reproducible from seed.
    Returns dict with F (B,H,W) np, clean (B,S,S), noisy {frames: (B,S,S)}, A (B,2,2), b (B,2), bumps."""
    gen = torch.Generator(device=device).manual_seed(int(seed))
    a = torch.as_tensor(aerial_crop, device=device, dtype=torch.float32)[:, None]
    p = to_torch(p_np, device)
    F, bl = build_truth(a, p, sc, gen, bumps=bumps)
    clean, A, b = render_clean(F, p, sc, S)
    out = {"F": F[:, 0].cpu().numpy(), "clean": clean[:, 0].cpu().numpy(), "A": A.cpu().numpy(),
           "b": b.cpu().numpy(), "bumps": bl, "noisy": {}}
    for fr in (frames or []):
        g2 = torch.Generator(device=device).manual_seed(noise_seed(seed, fr, 0))
        out["noisy"][fr] = add_noise(clean, fr, sc, g2)[:, 0].cpu().numpy()
    return out


# ----------------------------------------------------------------------------------------------- truth queries
def crossings_along(field, pts, normals, half_window, step=0.05, order=1):
    """Signed offset t (px) along unit normal from each point to the zero crossing of `field` nearest t=0.
    field (H,W) np; pts (N,2) [x,y]; normals (N,2) [nx,ny]. Interpolation order 1 = bilinear (truth definition).
    Returns t (N,) float64, nan where no crossing; also count of crossings in the window (N,)."""
    ts = np.arange(-half_window, half_window + 1e-9, step)
    xs = pts[:, 0:1] + normals[:, 0:1] * ts[None]
    ys = pts[:, 1:2] + normals[:, 1:2] * ts[None]
    v = ndimage.map_coordinates(field.astype(np.float64), [ys.ravel(), xs.ravel()], order=order,
                                mode="nearest").reshape(xs.shape)
    s = np.signbit(v)
    ch = s[:, 1:] != s[:, :-1]
    n_cross = ch.sum(1)
    t0 = ts[:-1][None] + step * v[:, :-1] / (v[:, :-1] - v[:, 1:] + 1e-30)
    t0 = np.where(ch, t0, np.inf)
    best = np.take_along_axis(t0, np.argmin(np.abs(t0), 1)[:, None], 1)[:, 0]
    best[~np.isfinite(best)] = np.nan
    return best, n_cross


def design_to_sem(xy, A, b):
    """Design-crop px (N,2) -> SEM px using one sample's A (2,2), b (2)."""
    return (np.linalg.inv(A) @ (xy - b).T).T


# ----------------------------------------------------------------------------------------------- crops
def load_crop(cfg, set_name, tile_id, y0, x0, S):
    """SEM field of view [y0, y0+S) x [x0, x0+S) in tile px. Returns (aerial, design) crops of size S+2M,
    i.e. tile rows [y0-M, y0+S+M). aerial: float32 hybrid field; design: bool target raster."""
    from data_lithobench import load_tile
    sc, M = cfg["sim"], cfg["sim"]["margin_px"]
    pad = 4 * int(math.ceil(sc["field_smooth_px"])) + 2
    d = load_tile(cfg["paths"]["lithobench"], set_name, tile_id, ("target", "litho", "resist"))
    ys, xs = slice(y0 - M - pad, y0 + S + M + pad), slice(x0 - M - pad, x0 + S + M + pad)
    a = hybrid_aerial(d["litho"][ys, xs], d["resist"][ys, xs], sc)[pad:-pad, pad:-pad]
    t = d["target"][y0 - M:y0 + S + M, x0 - M:x0 + S + M] > 0.5
    return a, t


def choose_crop(cfg, set_name, tile_id, S, n_grid=3):
    """Deterministic FOV origin inside the active region maximising design edge length."""
    from data_lithobench import load_tile
    lc = cfg["lithobench"]
    t = load_tile(cfg["paths"]["lithobench"], set_name, tile_id, ("target",))["target"] > 0.5
    best, arg = -1, None
    for y0 in np.linspace(lc["active_lo_px"], lc["active_hi_px"] - S, n_grid).astype(int):
        for x0 in np.linspace(lc["active_lo_px"], lc["active_hi_px"] - S, n_grid).astype(int):
            w = t[y0:y0 + S, x0:x0 + S]
            score = int(np.abs(np.diff(w, axis=0)).sum() + np.abs(np.diff(w, axis=1)).sum())
            if score > best:
                best, arg = score, (int(y0), int(x0))
    return arg


# ----------------------------------------------------------------------------------------------- M2 checks
def profile_measure(img, pts, normals, half_window, mode="peak", level=None, step=0.02):
    """Direct profile measurement along normals on an image (cubic interpolation).
    mode 'peak': sub-pixel argmax; mode 'level': crossing of `level` nearest t=0. Returns offsets t (N,)."""
    if mode == "level":
        return crossings_along(img - level, pts, normals, half_window, step, order=3)[0]
    ts = np.arange(-half_window, half_window + 1e-9, step)
    xs = pts[:, 0:1] + normals[:, 0:1] * ts[None]
    ys = pts[:, 1:2] + normals[:, 1:2] * ts[None]
    v = ndimage.map_coordinates(img.astype(np.float64), [ys.ravel(), xs.ravel()], order=3,
                                mode="nearest").reshape(xs.shape)
    k = np.clip(np.argmax(v, 1), 1, len(ts) - 2)
    r = np.arange(len(k))
    y0, y1, y2 = v[r, k - 1], v[r, k], v[r, k + 1]
    den = y0 - 2 * y1 + y2
    frac = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    return ts[k] + frac * step


def nominal_params(sc, n, **over):
    """Fixed mid-range parameters (no roughness, no distortion) with optional overrides."""
    fixed = {"threshold": sc["threshold_nominal"], "level_out": 0.25, "contrast": 0.15, "bloom_amp": 0.45,
             "bloom_width_px": 2.0, "bloom_offset_px": 0.0, "blur_px": 1.5, "rough_sigma_nm": 0.0,
             "rough_corr_nm": 10.0, "scale_x": 0.0, "scale_y": 0.0, "rot_rad": 0.0, "shear": 0.0,
             "shift_x": 0.0, "shift_y": 0.0}
    fixed.update(over)
    return {k: np.full(n, float(fixed[k])) for k in PARAM_KEYS}


def check_profile_recovery(cfg, set_name, tile_id, device="cuda", mode="peak", **over):
    """Noise-free, distortion-free render; direct profile measurement along design normals vs truth crossing.
    mode 'peak' = bloom maximum, 'level' = crossing of level_out + contrast/2 (meaningful when bloom_amp = 0).
    Returns errors (measured - truth) in px along the outward normal for unambiguous straight-edge points."""
    from gauges import design_edge_points
    sc, S, M = cfg["sim"], cfg["lithobench"]["crop_px"], cfg["sim"]["margin_px"]
    y0, x0 = choose_crop(cfg, set_name, tile_id, S)
    a, t = load_crop(cfg, set_name, tile_id, y0, x0, S)
    p = nominal_params(sc, 1, **over)
    out = simulate(a[None], p, sc, S, seed=1, device=device, bumps=False)
    pts, nrm, _ = design_edge_points(t, 10, stride=2, border_px=M + 12)
    tt, nc = crossings_along(out["F"][0], pts, nrm, 10.0)
    ok = (nc == 1) & np.isfinite(tt)
    pts, nrm, tt = pts[ok], nrm[ok], tt[ok]
    sem_pts = design_to_sem(pts, out["A"][0], out["b"][0])          # identity affine here: shift by margin
    meas = profile_measure(out["clean"][0], sem_pts, nrm, 10.0, mode, level=p["level_out"][0] + p["contrast"][0] / 2)
    e = meas - tt
    return e[np.isfinite(e)]


def edge_shift_vs_threshold(cfg, set_name, tile_ids, thresholds):
    """Truth edge displacement (px, + outward) relative to nominal threshold, on straight design edges."""
    from gauges import design_edge_points
    sc, S, M = cfg["sim"], cfg["lithobench"]["crop_px"], cfg["sim"]["margin_px"]
    res = {float(t): [] for t in thresholds}
    for tid in tile_ids:
        y0, x0 = choose_crop(cfg, set_name, tid, S)
        a, tm = load_crop(cfg, set_name, tid, y0, x0, S)
        pts, nrm, _ = design_edge_points(tm, 10, stride=4, border_px=M + 16)
        base, nc = crossings_along(a - sc["threshold_nominal"], pts, nrm, 15.0)
        ok = (nc == 1) & np.isfinite(base)
        for t in thresholds:
            tt, nc2 = crossings_along(a - t, pts, nrm, 15.0)
            g = ok & (nc2 == 1) & np.isfinite(tt)
            res[float(t)].append(tt[g] - base[g])
    return {t: np.concatenate(v) if v else np.zeros(0) for t, v in res.items()}


def m2_report(cfg, device="cuda"):
    import json
    sc = cfg["sim"]
    rd = os.path.join(cfg["paths"]["results"], "simulator_checks")
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(cfg["paths"]["results"], "splits.json")) as f:
        splits = json.load(f)
    val_metal, val_via = splits["val"]["MetalSet"][:6], splits["val"]["ViaSet"][:6]
    rep = {"tiles": {"MetalSet": val_metal, "ViaSet": val_via}}
    # 1) truth recovery without bloom offset: peak on pure-bloom render, level on bloom-free render
    # 2) known bloom offsets (expected bias = -offset along outward normal)
    # 3) peak location WITH material contrast: an inherent bias of peak-based edge finding (reported, not a truth error)
    cases = [("recovery_peak_pure_bloom", "peak", dict(contrast=0.0), 0.0),
             ("recovery_level_no_bloom", "level", dict(bloom_amp=0.0), 0.0),
             ("bloom_offset_+1.0_pure_bloom", "peak", dict(contrast=0.0, bloom_offset_px=1.0), -1.0),
             ("bloom_offset_-1.0_pure_bloom", "peak", dict(contrast=0.0, bloom_offset_px=-1.0), 1.0),
             ("peak_with_contrast_offset_0", "peak", dict(), None),
             ("peak_with_contrast_offset_+1.0", "peak", dict(bloom_offset_px=1.0), None)]
    for name, mode, over, expect in cases:
        errs = np.concatenate([check_profile_recovery(cfg, s, t, device, mode, **over)
                               for s, ids in (("MetalSet", val_metal), ("ViaSet", val_via)) for t in ids])
        r = {"mode": mode, "overrides": over, "n_points": int(errs.size), "median_err_px": float(np.median(errs)),
             "median_abs_err_px": float(np.median(np.abs(errs))),
             "p95_abs_err_px": float(np.percentile(np.abs(errs), 95)), "expected_bias_px": expect}
        rep[name] = r
        print(f"{name:32s} n={r['n_points']} median err {r['median_err_px']:+.4f} median|err| "
              f"{r['median_abs_err_px']:.4f} p95|err| {r['p95_abs_err_px']:.4f} expected {expect}")
    ths = [0.19, 0.20, 0.205, 0.215, 0.225, 0.235, 0.245, 0.25, 0.26]   # 3) exposure-like sweep, truth only
    sh = edge_shift_vs_threshold(cfg, "MetalSet", val_metal, ths)
    rep["exposure_sweep_metal"] = {}
    for t in ths:
        v = sh[t]
        r = {"n": int(v.size), "median_edge_shift_px": float(np.median(v)), "cd_change_2x_median_nm": float(2 * np.median(v)),
             "iqr_edge_shift_px": float(np.subtract(*np.percentile(v, [75, 25]))),
             "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}
        rep["exposure_sweep_metal"][f"{t:.3f}"] = r
        print(f"threshold {t:.3f}: n={r['n']} median edge shift {r['median_edge_shift_px']:+.3f}px "
              f"(~CD {r['cd_change_2x_median_nm']:+.2f}nm) IQR {r['iqr_edge_shift_px']:.3f} p5..p95 {r['p5']:+.2f}..{r['p95']:+.2f}")
    with open(os.path.join(rd, "sim_checks.json"), "w") as f:
        json.dump(rep, f, indent=1)
    contact_sheet(cfg, "MetalSet", val_metal[0], os.path.join(rd, "contact_sheet_metal.png"), device)
    contact_sheet(cfg, "ViaSet", val_via[0], os.path.join(rd, "contact_sheet_via.png"), device)
    return rep


def contact_sheet(cfg, set_name, tile_id, out_png, device="cuda"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sc, S, M = cfg["sim"], cfg["lithobench"]["crop_px"], cfg["sim"]["margin_px"]
    y0, x0 = choose_crop(cfg, set_name, tile_id, S)
    a, t = load_crop(cfg, set_name, tile_id, y0, x0, S)
    p = sample_params(np.random.default_rng(7), sc, 1)
    out = simulate(a[None], p, sc, S, seed=7, device=device, frames=sc["frames"])
    fig, ax = plt.subplots(2, 4, figsize=(16, 8.4))
    ax = ax.ravel()
    ax[0].imshow(t[M:-M, M:-M], cmap="gray")
    ax[0].contour(out["F"][0][M:-M, M:-M], [0], colors="r", linewidths=0.6)
    ax[0].set_title("design (white) + truth contour (red)")
    ax[1].imshow(out["clean"][0], cmap="gray"); ax[1].set_title("noise-free SEM")
    for i, fr in enumerate(sc["frames"]):
        ax[2 + i].imshow(out["noisy"][fr][0], cmap="gray"); ax[2 + i].set_title(f"{fr} frames")
    for x in ax:
        x.axis("off")
    fig.suptitle(f"{set_name}/{tile_id} FOV y0={y0} x0={x0}; simulated (controlled forward model, not physical)")
    fig.tight_layout(); fig.savefig(out_png, dpi=70); plt.close(fig)


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_lithobench import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2-report", action="store_true")
    if ap.parse_args().m2_report:
        m2_report(load_config())
