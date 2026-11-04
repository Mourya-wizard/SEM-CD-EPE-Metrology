"""Compact U-Net shell with a selectable operator; outputs a truncated signed distance map (TSDF).

Shell (fixed for all variants): stem 3x3 -> 4 encoder levels (widths w, 2w, 4w, 8w; stride-2 conv downsampling)
-> bottleneck (8w) -> 3 decoder levels (bilinear up + skip concat). Every level has one ConvBlock
(3x3 conv - GroupNorm - GELU). Operator blocks (pre-norm residual: x + op(GELU(GN(x)))) sit at the SAME five
positions in every variant: encoder levels 2 and 3, bottleneck, decoder levels 3 and 2 (1/4 .. 1/16 resolution).
Only the operator changes; the base width w is adjusted per variant to keep parameter counts within ~10%.
Output: 1 channel, TSDF in units of the truncation distance (target = clip(sd_px / trunc, -1, 1), + inside).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators import make_op


def gn(c):
    return nn.GroupNorm(math.gcd(c, 8), c)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv, self.norm = nn.Conv2d(cin, cout, 3, padding=1), gn(cout)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))


class OpBlock(nn.Module):
    def __init__(self, c, op):
        super().__init__()
        self.norm, self.op = gn(c), make_op(op, c)

    def forward(self, x):
        return x + self.op(F.gelu(self.norm(x)))


class UNet(nn.Module):
    def __init__(self, in_ch=2, width=32, op="conv"):
        super().__init__()
        w = [width, 2 * width, 4 * width, 8 * width]
        self.stem = ConvBlock(in_ch, w[0])
        self.enc = nn.ModuleList([ConvBlock(w[0], w[0])] + [ConvBlock(w[i - 1], w[i]) for i in range(1, 4)])
        self.down = nn.ModuleList([nn.Conv2d(w[i], w[i], 3, stride=2, padding=1) for i in range(3)])
        self.enc_op = nn.ModuleDict({"2": OpBlock(w[2], op), "3": OpBlock(w[3], op)})
        self.mid = nn.Sequential(nn.Conv2d(w[3], w[3], 3, stride=2, padding=1), ConvBlock(w[3], w[3]), OpBlock(w[3], op))
        self.dec = nn.ModuleList([ConvBlock(w[3] + w[3], w[3]), ConvBlock(w[3] + w[2], w[2]),
                                  ConvBlock(w[2] + w[1], w[1]), ConvBlock(w[1] + w[0], w[0])])
        self.dec_op = nn.ModuleDict({"0": OpBlock(w[3], op), "1": OpBlock(w[2], op)})
        self.head = nn.Conv2d(w[0], 1, 1)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for i in range(4):
            x = self.enc[i](x)
            if str(i) in self.enc_op:
                x = self.enc_op[str(i)](x)
            skips.append(x)
            if i < 3:
                x = self.down[i](x)
        x = self.mid(x)
        for j in range(4):
            s = skips[3 - j]
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = self.dec[j](torch.cat([x, s], 1))
            if str(j) in self.dec_op:
                x = self.dec_op[str(j)](x)
        return self.head(x)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_model(mc, op, in_ch):
    """mc = cfg['model']; width per operator from mc['width'][op] (documented param matching)."""
    return UNet(in_ch=in_ch, width=mc["width"][op], op=op)
