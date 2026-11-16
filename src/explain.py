"""Explainability (E11): Grad-CAM for the metrology TSDF models and the Carinthia-S segmentation models,
learned-offset maps of the deformable operators, and CAD-channel attribution.

Scalars that Grad-CAM differentiates:
  metrology     : sum over CD-gauge probes of the predicted edge offset (differentiable zero crossing of the TSDF along
                  the gauge normals), i.e. "which image regions move the predicted CD"
  segmentation  : sum of the defect logits inside the predicted (or, if empty, annotated) defect region
Layers (UNet in models.py): the deepest deformable block `mid[2]` (1/16 resolution), the last deformable block
`dec_op['1']` (1/4) and the final decoder features `dec[3]` (full resolution).
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

import measure as ms

STRIDE = {"mid": 16, "dec_op1": 4, "last": 1, "enc_op2": 4, "enc_op3": 8, "dec_op0": 8}


def layers_of(model):
    return {"enc_op2": model.enc_op["2"], "enc_op3": model.enc_op["3"], "mid": model.mid[2], "dec_op0": model.dec_op["0"],
            "dec_op1": model.dec_op["1"], "last": model.dec[3]}


def grad_cam(model, x, score_fn, layers=None):
    """x (1,C,S,S). Returns {layer: CAM (S,S) in [0,1]}, input attribution per channel (C,), and the raw output."""
    layers = layers or layers_of(model)
    acts = {}
    hs = [m.register_forward_hook(lambda _m, _i, o, k=k: acts.__setitem__(k, o)) for k, m in layers.items()]
    x = x.clone().requires_grad_(True)
    try:
        out = model(x)
        score = score_fn(out)
        keys = list(acts)
        grads = torch.autograd.grad(score, [acts[k] for k in keys] + [x])
    finally:
        for h in hs:
            h.remove()
    cams = {}
    for k, g in zip(keys, grads[:-1]):
        a = acts[k].detach()
        cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cams[k] = (cam / cam.max().clamp(min=1e-12)).cpu().numpy()
    attr = (grads[-1] * x.detach()).abs().sum((0, 2, 3)).cpu().numpy()
    return cams, attr / max(attr.sum(), 1e-12), out.detach(), float(score)


def metrology_score(cfg, probes, A, b):
    """Score function for grad_cam on a TSDF model output (1,1,S,S, units of the truncation distance)."""
    cd = probes[probes.kind.isin(("cd", "contact"))].reset_index(drop=True)      # MetalSet tiles carry CD gauges, ViaSet tiles contact gauges (v1 used "cd" only: empty score, blank CAMs on ViaSet)
    q, ns, scale = ms.probes_to_sem(cd, A, b)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    qt, nt, sc = (torch.as_tensor(v, device=dev) for v in (q, ns, scale))
    trunc, hw = cfg["model"]["tsdf_trunc_px"], cfg["gauges"]["probe_half_window_px"]

    def fn(out):
        t = ms.torch_crossings(out[0, 0] * trunc, qt, nt, hw) * sc
        return torch.nan_to_num(t, nan=0.0).sum()
    return fn


def segmentation_score(gt):
    """Score function for a segmentation logit map (1,1,H,W): summed logits over the predicted defect region."""
    gt_t = torch.as_tensor(gt, dtype=torch.float32)

    def fn(out):
        reg = (out.detach() > 0).float()
        if reg.sum() == 0:
            reg = gt_t.to(out.device)[None, None]
        return (out * reg).sum()
    return fn


def offset_magnitude(model, x, level="dec_op1"):
    """Mean magnitude (feature px) of the sampling offsets of the deformable block at `level`, as a map at the block's
    resolution, or None for models without deformable operators."""
    blk = layers_of(model)[level]
    op = blk.op
    name = type(op).__name__
    mod = {"DCNv2Op": getattr(op, "offset_mask", None), "DCNv3Op": getattr(getattr(op, "op", None), "offset", None),
           "DCNv4Reimpl": getattr(op, "offset_mask", None)}.get(name)
    if mod is None:
        return None
    store = {}
    h = mod.register_forward_hook(lambda _m, _i, o: store.__setitem__("o", o.detach()))
    with torch.no_grad():
        model(x)
    h.remove()
    o = store["o"].float()
    if name == "DCNv2Op":                       # (B, 27, h, w): 18 offsets as (dy, dx) per point, then the mask
        B, _, hh, ww = o.shape
        return o[:, :18].reshape(B, 9, 2, hh, ww).norm(dim=2).mean(1)[0].cpu().numpy()
    if name == "DCNv4Reimpl":                   # (B, h, w, pad): per group [2P offsets, P mask]
        B, hh, ww, _ = o.shape
        o = o[..., : op.K * 3].reshape(B, hh, ww, op.g, 3 * op.P)[..., : 2 * op.P]
    return o.reshape(*o.shape[:3], -1, 2).norm(dim=-1).mean(-1)[0].cpu().numpy()


def contour_distance(mask):
    """Distance (px) of every pixel to the nearest boundary of a boolean mask."""
    mask = np.asarray(mask, bool)
    if mask.all() or not mask.any():
        return np.full(mask.shape, 1e3)
    return np.where(mask, ndimage.distance_transform_edt(mask), ndimage.distance_transform_edt(~mask))


def focus_stats(cam, dist, near=(4, 12), inside=None, far=16):
    """Share of CAM mass within `near` px of the contour (with the chance level = area share), and, if `inside` (boolean
    feature mask) is given, the mass inside features and the mass in the background farther than `far` px from any edge."""
    w = cam / max(cam.sum(), 1e-12)
    out = {}
    for r in near:
        sel = dist < r
        out[f"cam_within_{r}px"] = float(w[sel].sum())
        out[f"area_within_{r}px"] = float(sel.mean())
    out["cam_mean_dist_px"] = float((w * np.minimum(dist, 64)).sum())
    if inside is not None:
        inside = np.asarray(inside, bool)
        bg = (~inside) & (dist >= far)
        out["cam_inside_features"], out["area_inside_features"] = float(w[inside].sum()), float(inside.mean())
        out[f"cam_far_background_{far}px"], out[f"area_far_background_{far}px"] = float(w[bg].sum()), float(bg.mean())
    return out


def offset_stats(mag, dist, stride, near=4):
    """Mean offset magnitude (input px) near the contour vs elsewhere; mag is a map at 1/stride resolution."""
    d = F.interpolate(torch.as_tensor(dist, dtype=torch.float32)[None, None], size=mag.shape, mode="area")[0, 0].numpy()
    m = mag * stride
    sel = d < near
    return {"offset_near_px": float(m[sel].mean()) if sel.any() else float("nan"),
            "offset_far_px": float(m[~sel].mean()) if (~sel).any() else float("nan"), "offset_mean_px": float(m.mean())}
