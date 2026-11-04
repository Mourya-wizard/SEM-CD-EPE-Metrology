"""M4 operator tests: shape, finite outputs/gradients, backward, start-as-regular-conv, float precision."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from models import UNet, n_params  # noqa: E402
from operators import OPS, make_op  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
C = 64


@pytest.mark.parametrize("name", list(OPS))
def test_shape_finite_backward(name):
    torch.manual_seed(0)
    op = make_op(name, C).to(DEV)
    if name == "dcnv4":   # official DCNv4 init (zero offset_mask) gives zero output and zero input-grad at init
        torch.nn.init.normal_(op.offset_mask.weight, std=1e-2)
    x = torch.randn(2, C, 24, 20, device=DEV, requires_grad=True)
    y = op(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for n, p in op.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


@pytest.mark.parametrize("name", ["dcnv2", "dcnv3", "dcnv4"])
def test_offsets_get_gradients(name):
    """Zero-initialised offset predictors must still receive gradient (otherwise the op never deforms)."""
    torch.manual_seed(1)
    op = make_op(name, C).to(DEV)
    if name == "dcnv4":                       # official DCNv4 init gives zero modulation -> zero output at init;
        torch.nn.init.normal_(op.offset_mask.weight, std=1e-2)   # perturb so offsets get a gradient path
    op(torch.randn(2, C, 16, 16, device=DEV)).square().mean().backward()
    w = op.op.offset.weight if name == "dcnv3" else op.offset_mask.weight
    assert w.grad.abs().sum() > 0


def test_dcnv2_starts_as_regular_conv():
    torch.backends.cudnn.allow_tf32 = False      # compare in true fp32 (cuDNN conv defaults to TF32)
    torch.manual_seed(2)
    op = make_op("dcnv2", C).to(DEV)
    conv = torch.nn.Conv2d(C, C, 3, padding=1).to(DEV)
    conv.weight.data.copy_(op.dcn.weight.data); conv.bias.data.copy_(op.dcn.bias.data)
    x = torch.randn(1, C, 16, 16, device=DEV)
    ok = torch.allclose(op(x), conv(x), atol=1e-4)
    torch.backends.cudnn.allow_tf32 = True
    assert ok


@pytest.mark.parametrize("name", list(OPS))
def test_autocast_precision(name):
    """Under bf16 autocast the DCN sampling runs in float32 internally; outputs must stay close to float32."""
    torch.manual_seed(3)
    op = make_op(name, C).to(DEV)
    if name == "dcnv4":
        torch.nn.init.normal_(op.offset_mask.weight, std=1e-2)
    x = torch.randn(2, C, 16, 16, device=DEV)
    y32 = op(x)
    with torch.autocast(DEV, dtype=torch.bfloat16):
        y16 = op(x)
    rel = ((y16.float() - y32).norm() / y32.norm().clamp(min=1e-8)).item()
    assert torch.isfinite(y16).all()
    assert rel < (0.02 if name == "conv" else 1e-5), rel     # DCN ops are float32 inside autocast


def test_param_counts_within_10pct():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml")))
    w = cfg["model"]["width"]
    base = n_params(UNet(2, w["conv"], "conv"))
    for op in ("dcnv2", "dcnv3", "dcnv4"):
        assert abs(n_params(UNet(2, w[op], op)) / base - 1) <= 0.10, op


@pytest.mark.parametrize("name", list(OPS))
def test_unet_overfits_one_batch(name):
    """One-batch overfit on a synthetic TSDF task: loss must drop by >= 80%."""
    torch.manual_seed(4)
    m = UNet(2, 16, name).to(DEV)
    yy, xx = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    tgt = ((12 - torch.hypot(xx - 30, yy - 34)) / 8).clamp(-1, 1)[None, None].to(DEV).repeat(2, 1, 1, 1)
    x = torch.cat([tgt + 0.1 * torch.randn_like(tgt), (tgt > 0).float()], 1)
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    l0 = None
    for _ in range(150):
        opt.zero_grad()
        with torch.autocast(DEV, dtype=torch.bfloat16):
            loss = (m(x).float() - tgt).abs().mean()
        loss.backward(); opt.step()
        l0 = l0 if l0 is not None else loss.item()
    assert loss.item() < 0.2 * l0, (l0, loss.item())
