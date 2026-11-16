"""Roadmap v2 mechanics (E11 Grad-CAM, E12 bias-aware registration, E13 auxiliary head, E14 spectrum loss).
No datasets needed; synthetic inputs only."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import explain as ex  # noqa: E402
import registration as rg  # noqa: E402
from models import UNet, build_for_run  # noqa: E402
from train import spectral_loss  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml")))
RC = CFG["registration"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("op", ["conv", "dcnv2", "dcnv3", "dcnv4"])
def test_gradcam_and_offsets(op):
    torch.manual_seed(0)
    m = UNet(in_ch=2, width=16, op=op).to(DEV).eval()
    x = torch.randn(1, 2, 64, 64, device=DEV)
    cams, attr, out, _ = ex.grad_cam(m, x, lambda o: o.mean())
    assert set(cams) == {"enc_op2", "enc_op3", "mid", "dec_op0", "dec_op1", "last"}   # five operator blocks + last decoder layer
    for c in cams.values():
        assert c.shape == (64, 64) and c.min() >= 0 and c.max() <= 1 + 1e-6
    assert attr.shape == (2,) and abs(attr.sum() - 1) < 1e-5
    mag = ex.offset_magnitude(m, x)
    if op == "conv":
        assert mag is None
    else:                                            # offsets start at zero; perturb the predictor to get a signal
        blk = m.dec_op["1"].op
        lin = blk.op.offset if op == "dcnv3" else blk.offset_mask
        torch.nn.init.normal_(lin.weight, std=0.05)
        mag = ex.offset_magnitude(m, x)
        assert mag.shape == (16, 16) and np.isfinite(mag).all() and mag.max() > 0


def test_metrology_score_is_differentiable():
    S = 64
    yy, xx = torch.meshgrid(torch.arange(S, dtype=torch.float32), torch.arange(S, dtype=torch.float32), indexing="ij")
    field = ((20 - (xx - 31.5).abs()) / CFG["model"]["tsdf_trunc_px"]).clamp(-1, 1)[None, None].to(DEV).requires_grad_(True)
    probes = pd.DataFrame({"kind": "cd", "x": [11.5, 51.5], "y": [32.0, 32.0], "nx": [-1.0, 1.0], "ny": [0.0, 0.0]})
    fn = ex.metrology_score(CFG, probes, np.eye(2), np.zeros(2))
    s = fn(field)
    g = torch.autograd.grad(s, field)[0]
    assert torch.isfinite(g).all() and g.abs().sum() > 0
    assert abs(float(s) - 0.0) < 0.2             # edges at x = 11.5 and 51.5 -> zero offset


def test_bias_registration_recovers_bias_and_shift():
    from scipy import ndimage
    M, S = 32, 128
    H = S + 2 * M
    T = np.zeros((H, H), bool)
    T[40:70, 50:120] = True
    T[90:140, 60:80] = True
    T[100:130, 110:150] = True
    rc = dict(RC, bias_grid=[-6, -4, -2, 0, 2, 4, 6])
    tm, _ = rg.design_template(T[None], rc, bias_px=4.0)           # "printed" pattern: edges grown by 4 px
    dx, dy = 37, 29
    sem = torch.as_tensor(tm[0][dy:dy + S, dx:dx + S], device=DEV)[None, None]
    feats = rg.bias_features(T, rc, DEV)
    A, b, bias, pk, psr = rg.register_biased(sem, feats, rc, M)
    assert float(bias[0]) == 4.0
    sh = rg._coarse_scored(rg.smooth_norm(sem, rc["coarse_sigma_px"]), feats[5:6], M)[0][0]    # bias +4 template index 5
    assert float(sh[0]) == pytest.approx(dx, abs=0.3) and float(sh[1]) == pytest.approx(dy, abs=0.3)
    es = rg.smooth_norm(sem, rc["coarse_sigma_px"])
    wrong = rg._coarse_scored(es, feats[0:1], M)[1][0]                                         # bias -6 template
    assert float(pk[0]) > float(wrong) and np.isfinite(float(psr[0]))


def test_psr_is_low_for_noise():
    torch.manual_seed(1)
    M, S = 32, 128
    es = torch.randn(1, 1, S, S, device=DEV)
    ed = torch.randn(1, 1, S + 2 * M, S + 2 * M, device=DEV)
    _, _, psr = rg._coarse_scored(es, ed, M)
    assert float(psr[0]) < 8


def test_aux_head_and_run_builder():
    m = UNet(in_ch=1, width=16, op="conv", aux_offset=True).to(DEV)
    out, off = m(torch.randn(2, 1, 64, 64, device=DEV), return_aux=True)
    assert out.shape == (2, 1, 64, 64) and off.shape == (2,)
    base = UNet(in_ch=1, width=16, op="conv").to(DEV)
    o2, a2 = base(torch.randn(1, 1, 64, 64, device=DEV), return_aux=True)
    assert a2 is None and o2.shape == (1, 1, 64, 64)
    run = {"op": "conv", "cad": False, "train": {"aux_offset_weight": 0.05}}
    assert build_for_run(CFG["model"], run).offset_head is not None
    assert build_for_run(CFG["model"], {"op": "conv", "cad": False}).offset_head is None


def test_spectral_loss_penalises_smoothing():
    torch.manual_seed(2)
    S = 128
    x = torch.arange(S, dtype=torch.float32)
    rough = (torch.randn(S).cumsum(0) * 0.0)                      # deterministic base
    edge = 40 + 1.5 * torch.randn(S)                              # rough vertical edge position per row
    tgt = ((edge[:, None] - x[None, :]) / 8).clamp(-1, 1)[None, None].to(DEV)
    smooth_edge = torch.as_tensor(np.convolve(edge.numpy(), np.ones(9) / 9, mode="same"), dtype=torch.float32)
    sm = ((smooth_edge[:, None] - x[None, :]) / 8).clamp(-1, 1)[None, None].to(DEV)
    assert float(spectral_loss(tgt, tgt)) == pytest.approx(0.0, abs=1e-6)
    assert float(spectral_loss(sm, tgt)) > 0.2
    del rough
