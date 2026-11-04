"""Common operator-block interface: op(x: (N,C,H,W)) -> (N,C,H,W).

Sources (see README, operator table):
  conv   : torch.nn.Conv2d 3x3 (regular convolution baseline)
  dcnv2  : torchvision.ops.DeformConv2d (official torchvision op), learned offsets + modulation mask from a
           zero-initialised 3x3 conv; mask = 2*sigmoid(m) so the op starts exactly as a regular convolution.
  dcnv3  : OFFICIAL InternImage PyTorch fallback DCNv3_pytorch (third_party/dcnv3, MIT), unchanged.
  dcnv4  : PyTorch REIMPLEMENTATION (NOT the official CUDA operator; its build failed, see third_party/dcnv4_reference).
           Official module structure + un-normalised modulation, using the vendored DCNv3 PyTorch core.
DCN sampling runs in float32 even under autocast (grid_sample-based sampling; see tests/test_operators.py).
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from torchvision.ops import DeformConv2d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from third_party.dcnv3 import DCNv3_pytorch, dcnv3_core_pytorch  # noqa: E402

GROUP_CHANNELS = 16          # channels per DCNv3/DCNv4 group (DCNv4 official requires a multiple of 16)


class ConvOp(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class DCNv2Op(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.offset_mask = nn.Conv2d(c, 27, 3, padding=1)
        constant_(self.offset_mask.weight, 0.0)
        constant_(self.offset_mask.bias, 0.0)
        self.dcn = DeformConv2d(c, c, 3, padding=1)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            om = self.offset_mask(x)
            off, m = om[:, :18], 2 * torch.sigmoid(om[:, 18:])
            return self.dcn(x, off, m)


class DCNv3Op(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.op = DCNv3_pytorch(c, kernel_size=3, group=c // GROUP_CHANNELS)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return self.op(x.float().permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DCNv4Reimpl(nn.Module):
    """NOT OFFICIAL. Reimplementation of the DCNv4 module semantics (OpenGVLab/DCNv4 modules/dcnv4.py):
    value_proj -> sampling with offsets and UN-normalised modulation from one Linear (offset_mask, zero init,
    3K outputs padded to a multiple of 8, per-group layout [2P offsets, P modulation]) -> output_proj."""

    def __init__(self, c, kernel_size=3, offset_scale=1.0):
        super().__init__()
        self.c, self.g, self.k = c, c // GROUP_CHANNELS, kernel_size
        self.P = kernel_size * kernel_size
        self.K = self.g * self.P
        self.offset_scale = offset_scale
        self.value_proj = nn.Linear(c, c)
        self.offset_mask = nn.Linear(c, int(math.ceil(self.K * 3 / 8) * 8))
        self.output_proj = nn.Linear(c, c)
        constant_(self.offset_mask.weight, 0.0); constant_(self.offset_mask.bias, 0.0)
        xavier_uniform_(self.value_proj.weight); constant_(self.value_proj.bias, 0.0)
        xavier_uniform_(self.output_proj.weight); constant_(self.output_proj.bias, 0.0)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            x = x.float().permute(0, 2, 3, 1)                      # N,H,W,C
            N, H, W, C = x.shape
            v = self.value_proj(x)
            om = self.offset_mask(x)[..., : self.K * 3].reshape(N, H, W, self.g, 3 * self.P)
            off = om[..., : 2 * self.P].reshape(N, H, W, -1)
            mask = om[..., 2 * self.P:].reshape(N, H, W, -1)       # no softmax (DCNv4)
            y = dcnv3_core_pytorch(v, off, mask, self.k, self.k, 1, 1, 1, 1, 1, 1, self.g, GROUP_CHANNELS,
                                   self.offset_scale, 0)
            return self.output_proj(y).permute(0, 3, 1, 2)


OPS = {"conv": ConvOp, "dcnv2": DCNv2Op, "dcnv3": DCNv3Op, "dcnv4": DCNv4Reimpl}
SOURCES = {
    "conv": "torch.nn.Conv2d 3x3",
    "dcnv2": "torchvision.ops.DeformConv2d (official torchvision op)",
    "dcnv3": "OpenGVLab/InternImage DCNv3_pytorch, official PyTorch fallback, vendored @31c962d (MIT)",
    "dcnv4": "PyTorch REIMPLEMENTATION of OpenGVLab/DCNv4 module semantics (NOT official; official op @4b848f7 failed to build: torch header C2872 under nvcc+MSVC)",
}


def make_op(name, c):
    return OPS[name](c)
