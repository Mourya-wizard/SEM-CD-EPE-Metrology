"""SEM-to-design registration: coarse translation (FFT cross-correlation of edge maps) + affine refinement.

Output follows sem_sim's convention: design-crop px p = A @ [u, v] + b for SEM px (u=col, v=row).
The same estimated (A, b) is used by every measurement method. NOTE: a uniform pattern translation is
indistinguishable from a stage offset and is removed here, so it is NOT a valid EPE sensitivity test.
"""
import os

os.environ.setdefault("HF_HOME", "./.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "./.cache/torch")
os.environ.setdefault("XDG_CACHE_HOME", "./.cache")
os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")

import numpy as np
import torch
import torch.nn.functional as Fn

from sem_sim import _blur, _grad_mag


def design_template(T, rc):
    """Matched CAD raster: render the design (B,H,W bool np) with the nominal SEM model (material step +
    edge bloom over the exact Euclidean signed distance of the design raster).
    Returns (template, weight) np float32 (B,H,W); weight ramps 0 -> 1 over rc['corner_weight_px'] away from
    design corners / line ends, where printed shapes deviate most from the design (pullback, rounding)."""
    from scipy import ndimage
    tm, wt = [], []
    for t in np.asarray(T, bool):
        d = np.where(t, ndimage.distance_transform_edt(t) - 0.5, 0.5 - ndimage.distance_transform_edt(~t))
        z = np.clip(-d / 0.5, -50, 50)
        img = rc["tmpl_contrast"] / (1 + np.exp(z)) + rc["tmpl_bloom_amp"] * np.exp(-0.5 * (d / rc["tmpl_bloom_width_px"]) ** 2)
        tm.append(img.astype(np.float32))
        s = t[:-1, :-1].astype(int) + t[1:, :-1] + t[:-1, 1:] + t[1:, 1:]
        corner = np.zeros_like(t)
        corner[:-1, :-1] = (s == 1) | (s == 3)
        cw = rc["corner_weight_px"]
        w = np.clip(ndimage.distance_transform_edt(~corner) / cw, 0, 1) if (corner.any() and cw > 0) else np.ones(t.shape)
        wt.append(w.astype(np.float32))
    return np.stack(tm), np.stack(wt)


def smooth_norm(x, sigma):
    """x (B,1,H,W) -> Gaussian-smoothed, zero-mean / unit-std per sample (intensity matching)."""
    s = torch.full((x.shape[0],), float(sigma), device=x.device, dtype=x.dtype)
    e = _blur(x, s)
    e = e - e.mean((-1, -2), keepdim=True)
    return e / e.flatten(1).std(1)[:, None, None, None].clamp(min=1e-8)


def edge_map(x, sigma):
    """x (B,1,H,W) -> gradient magnitude of Gaussian-smoothed x, zero-mean / unit-std per sample."""
    s = torch.full((x.shape[0],), float(sigma), device=x.device, dtype=x.dtype)
    e = _grad_mag(_blur(x, s))
    e = e - e.mean((-1, -2), keepdim=True)
    return e / e.flatten(1).std(1)[:, None, None, None].clamp(min=1e-8)


def _coarse(es, ed, M):
    """Integer+parabolic peak of cross-correlation for shifts in [0, 2M]. Returns (B,2) [dx, dy] = design - SEM."""
    B, _, S, _ = es.shape
    H, W = ed.shape[-2:]
    pad = torch.zeros(B, 1, H, W, device=es.device, dtype=es.dtype)
    pad[..., :S, :S] = es
    c = torch.fft.irfft2(torch.fft.rfft2(ed) * torch.conj(torch.fft.rfft2(pad)), s=(H, W))[:, 0, :2 * M + 1, :2 * M + 1]
    flat = c.flatten(1).argmax(1)
    iy, ix = flat // (2 * M + 1), flat % (2 * M + 1)
    out = []
    for i in range(B):
        y, x = int(iy[i]), int(ix[i])
        def sub(a, b_, cc):
            d = a - 2 * b_ + cc
            return float(0.5 * (a - cc) / d) if abs(float(d)) > 1e-12 else 0.0
        fx = sub(c[i, y, x - 1], c[i, y, x], c[i, y, x + 1]) if 0 < x < 2 * M else 0.0
        fy = sub(c[i, y - 1, x], c[i, y, x], c[i, y + 1, x]) if 0 < y < 2 * M else 0.0
        out.append([x + fx, y + fy])
    return torch.tensor(out, device=es.device, dtype=es.dtype)


def _affine(theta, c, D_SCALE):
    """theta (B,6) = [tx, ty, d11, d12, d21, d22]; p = (I + D*D_SCALE)(uv - c) + c + t."""
    B = theta.shape[0]
    A = torch.eye(2, device=theta.device, dtype=theta.dtype).expand(B, 2, 2) + D_SCALE * theta[:, 2:].reshape(B, 2, 2)
    cvec = torch.full((B, 2), c, device=theta.device, dtype=theta.dtype)
    b = cvec + theta[:, :2] - (A @ cvec[..., None])[..., 0]
    return A, b


def _warp(img, A, b, S):
    H, W = img.shape[-2:]
    uu, vv = torch.meshgrid(torch.arange(S, device=img.device, dtype=img.dtype),
                            torch.arange(S, device=img.device, dtype=img.dtype), indexing="xy")
    uv = torch.stack([uu, vv], -1).reshape(1, -1, 2, 1)
    xy = (A[:, None] @ uv)[..., 0] + b[:, None]
    grid = torch.stack([2 * xy[..., 0] / (W - 1) - 1, 2 * xy[..., 1] / (H - 1) - 1], -1).reshape(-1, S, S, 2)
    return Fn.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def register(sem, design, rc, M, weight=None, refine=None):
    """sem (B,1,S,S) float; design (B,1,S+2M,S+2M) float = matched template from design_template();
    weight (B,1,S+2M,S+2M) optional template weight. rc = cfg['registration'].
    refine: 'none' | 'translation' | 'affine' (default rc['refine']); affine terms carry a Gaussian prior
    with std rc['affine_prior'] (dimensionless) to stay well-posed on sparse patterns.
    Returns A (B,2,2), b (B,2) torch (design-crop px = A [u,v] + b)."""
    refine = refine or rc["refine"]
    S = sem.shape[-1]
    c = (S - 1) / 2.0
    fm = edge_map if rc["feature"] == "edge" else smooth_norm
    wgt = torch.ones_like(design) if weight is None else weight
    es, ed = fm(sem, rc["coarse_sigma_px"]), fm(design, rc["coarse_sigma_px"]) * wgt
    t0 = _coarse(es, ed, M)                                     # p = uv + t0
    theta0 = torch.zeros(sem.shape[0], 6, device=sem.device, dtype=sem.dtype)
    theta0[:, :2] = t0                                          # with A = I: b = t
    if refine == "none":
        return _affine(theta0, c, rc["d_scale"])
    es, ed = fm(sem, rc["fine_sigma_px"]), fm(design, rc["fine_sigma_px"])
    bd = rc["ncc_border_px"]
    es_in = es[..., bd:-bd, bd:-bd]
    theta = theta0.clone().requires_grad_(True)
    mask = torch.ones(6, device=sem.device, dtype=sem.dtype)
    if refine == "translation":
        mask[2:] = 0
    prior_scale = rc["affine_prior"] / rc["d_scale"]            # prior std in theta units
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=rc["lbfgs_iters"], line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        th = theta * mask
        A, b = _affine(th, c, rc["d_scale"])
        ww = _warp(wgt, A, b, S)[..., bd:-bd, bd:-bd]
        w = _warp(ed, A, b, S)[..., bd:-bd, bd:-bd]
        sw = ww.sum((-1, -2, -3)).clamp(min=1e-6)
        mw = (w * ww).sum((-1, -2, -3)) / sw
        ms = (es_in * ww).sum((-1, -2, -3)) / sw
        wc, sc_ = w - mw[:, None, None, None], es_in - ms[:, None, None, None]
        cov = (ww * wc * sc_).sum((-1, -2, -3)) / sw
        var_w = (ww * wc * wc).sum((-1, -2, -3)) / sw
        var_s = (ww * sc_ * sc_).sum((-1, -2, -3)) / sw
        ncc = cov / torch.sqrt(var_w * var_s).clamp(min=1e-10)
        # weak Gaussian prior on affine terms (NCC changes by ~1e-2 per px of misalignment)
        loss = -ncc.sum() + rc["affine_prior_weight"] * 0.5 * ((th[:, 2:] / prior_scale) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        A, b = _affine(theta.detach() * mask, c, rc["d_scale"])
    return A, b


def registration_error(A_est, b_est, A_true, b_true, S):
    """Per-sample (centre translation error px, max displacement error over FOV corners px). np arrays."""
    pts = np.array([[0, 0], [S - 1, 0], [0, S - 1], [S - 1, S - 1], [(S - 1) / 2, (S - 1) / 2]], float)
    pe = np.einsum("bij,nj->bni", A_est, pts) + b_est[:, None]
    pt = np.einsum("bij,nj->bni", A_true, pts) + b_true[:, None]
    d = np.linalg.norm(pe - pt, axis=-1)
    return d[:, -1], d.max(1)


def warp_design_to_sem(design, A, b, S):
    """Design raster (B,1,H,W) -> SEM frame (B,1,S,S) using registration (A,b). Used for CAD model input."""
    return _warp(design, A, b, S)
