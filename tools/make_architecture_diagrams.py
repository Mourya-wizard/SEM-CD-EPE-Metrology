"""Architecture diagrams of the TSDF U-Net variants (conv / DCNv2 / DCNv3 / DCNv4) -> results/architecture/.

  python tools/make_architecture_diagrams.py

Shapes and parameter counts are read from the real models in src/models.py (not typed by hand): a forward pass with
hooks records every block's output size, so the diagram cannot drift from the code.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data_lithobench import load_config  # noqa: E402
from models import build_model, n_params  # noqa: E402

OUT = os.path.join(ROOT, "results", "architecture")
NAMES = {"conv": "Regular convolution", "dcnv2": "DCNv2", "dcnv3": "DCNv3", "dcnv4": "DCNv4 (PyTorch reimplementation)"}
OPCOL = {"conv": "#9aa5b1", "dcnv2": "#f0a04b", "dcnv3": "#4fae6b", "dcnv4": "#9b7fd1"}
CONVCOL, IOCOL = "#cfe3f5", "#eeeeee"
W, H, GAP, DX, DY = 2.2, 1.45, 0.25, 1.2, 2.3


def shapes(model, size, in_ch):
    """Output (C, H, W) of every named block, recorded with forward hooks."""
    rec, hs = {}, []

    def reg(name, mod):
        hs.append(mod.register_forward_hook(lambda _m, _i, o, n=name: rec.__setitem__(n, tuple(o.shape[1:]))))
    reg("stem", model.stem)
    for i in range(4):
        reg(f"enc{i}", model.enc[i])
    for i in range(3):
        reg(f"down{i}", model.down[i])
    for k, m in model.enc_op.items():
        reg(f"encop{k}", m)
    reg("middown", model.mid[0]); reg("midconv", model.mid[1]); reg("midop", model.mid[2])
    for j in range(4):
        reg(f"dec{j}", model.dec[j])
    for k, m in model.dec_op.items():
        reg(f"decop{k}", m)
    reg("head", model.head)
    with torch.no_grad():
        model.eval()(torch.zeros(1, in_ch, size, size))
    for h in hs:
        h.remove()
    return rec


def box(ax, x, y, title, lines, color, edge="#333", bold=False, w=W):
    ax.add_patch(FancyBboxPatch((x, y), w, H, boxstyle="round,pad=0.02,rounding_size=0.12", fc=color, ec=edge, lw=1.3))
    ax.text(x + w / 2, y + H - 0.2, title, ha="center", va="center", fontsize=8.5, weight="bold" if bold else "normal")
    for k, t in enumerate(lines):
        ax.text(x + w / 2, y + H - 0.5 - 0.27 * k, t, ha="center", va="center", fontsize=7.2, color="#222")
    return (x, y, w)


def arrow(ax, p, q, text=None, style="-|>", ls="-", color="#333", rad=0.0, off=(0, 0.12), fs=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, lw=1.2, ls=ls, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha="center", va="bottom", fontsize=fs, color=color)


def sz(s):
    return f"{s[1]}x{s[2]}x{s[0]}"


def unet_figure(cfg, op):
    mc, size = cfg["model"], cfg["lithobench"]["crop_px"]
    w = mc["width"][op]
    ch = [w, 2 * w, 4 * w, 8 * w]
    m2, m1 = build_model(mc, op, 2), build_model(mc, op, 1)
    r = shapes(m2, size, 2)
    oc = OPCOL[op]
    fig, ax = plt.subplots(figsize=(21, 13.5))
    ax.set_xlim(-0.7, 19.6); ax.set_ylim(-5 * DY - 0.9, 2.9); ax.axis("off")
    y = lambda L: -L * DY
    opname = {"conv": "Conv 3x3", "dcnv2": "DCNv2 3x3", "dcnv3": "DCNv3 3x3", "dcnv4": "DCNv4 3x3"}[op]

    def conv(x, L, title, cin, cout, key, bold=False):
        return box(ax, x, y(L), title, [f"3x3 conv {cin}->{cout}, GN, GELU", sz(r[key])], CONVCOL, bold=bold)

    def opblk(x, L, c, key):
        g = [f"{c // 16} groups of 16 ch"] if op in ("dcnv3", "dcnv4") else []
        return box(ax, x, y(L), f"{opname} block", [f"x + {opname.split()[0]}(GELU(GN(x)))", *g, sz(r[key])], oc, edge="#222", bold=True)

    # ---- encoder
    ex = lambda L: L * DX
    bx = {}
    bx["stem"] = box(ax, ex(0), y(0), "Stem", [f"3x3 conv 1 (or 2)->{w}, GN, GELU", sz(r["stem"])], CONVCOL)
    bx["enc0"] = conv(ex(0) + W + GAP, 0, "Encoder 0", w, w, "enc0")
    bx["enc1"] = conv(ex(1), 1, "Encoder 1", w, 2 * w, "enc1")
    bx["enc2"] = conv(ex(2), 2, "Encoder 2", 2 * w, 4 * w, "enc2")
    bx["encop2"] = opblk(ex(2) + W + GAP, 2, 4 * w, "encop2")
    bx["enc3"] = conv(ex(3), 3, "Encoder 3", 4 * w, 8 * w, "enc3")
    bx["encop3"] = opblk(ex(3) + W + GAP, 3, 8 * w, "encop3")
    mx = 6.6
    bx["middown"] = box(ax, mx, y(4), "Bottleneck down", ["3x3 conv, stride 2", f"{sz(r['middown'])}"], IOCOL)
    bx["midconv"] = conv(mx + W + GAP, 4, "Bottleneck conv", 8 * w, 8 * w, "midconv")
    bx["midop"] = opblk(mx + 2 * (W + GAP), 4, 8 * w, "midop")
    # ---- decoder (right-aligned, mirrored)
    XR = 18.7
    dxend = lambda L: XR - (3 - L) * DX
    bx["dec0"] = conv(dxend(3) - 2 * W - GAP, 3, "Decoder 0", 16 * w, 8 * w, "dec0")
    bx["decop0"] = opblk(dxend(3) - W, 3, 8 * w, "decop0")
    bx["dec1"] = conv(dxend(2) - 2 * W - GAP, 2, "Decoder 1", 12 * w, 4 * w, "dec1")
    bx["decop1"] = opblk(dxend(2) - W, 2, 4 * w, "decop1")
    bx["dec2"] = conv(dxend(1) - W, 1, "Decoder 2", 6 * w, 2 * w, "dec2")
    bx["dec3"] = conv(dxend(0) - 2 * W - GAP, 0, "Decoder 3", 3 * w, w, "dec3")
    bx["head"] = box(ax, dxend(0) - W, y(0), "Head", [f"1x1 conv {w}->1", f"TSDF {sz(r['head'])}"], "#f6d9d9", bold=True)

    def right(k): x0, y0, w_ = bx[k]; return (x0 + w_, y0 + H / 2)
    def left(k): x0, y0, w_ = bx[k]; return (x0, y0 + H / 2)
    def top(k): x0, y0, w_ = bx[k]; return (x0 + w_ / 2, y0 + H)
    def bot(k): x0, y0, w_ = bx[k]; return (x0 + w_ / 2, y0)

    arrow(ax, right("stem"), left("enc0"))
    arrow(ax, right("enc2"), left("encop2")); arrow(ax, right("enc3"), left("encop3"))
    arrow(ax, right("middown"), left("midconv")); arrow(ax, right("midconv"), left("midop"))
    arrow(ax, right("dec0"), left("decop0")); arrow(ax, right("dec1"), left("decop1")); arrow(ax, right("dec3"), left("head"))
    # downsampling (stride-2 3x3 conv, C->C)
    down_t = lambda i, s: f"3x3 conv, s2\n{s}"
    arrow(ax, bot("enc0"), top("enc1"), f"3x3 conv, stride 2  ({sz(r['down0'])})", off=(1.6, -0.35))
    arrow(ax, bot("enc1"), top("enc2"), f"3x3 conv, stride 2  ({sz(r['down1'])})", off=(1.6, -0.35))
    arrow(ax, bot("encop2"), top("enc3"), f"3x3 conv, stride 2  ({sz(r['down2'])})", off=(1.6, -0.35), rad=0.0)
    arrow(ax, bot("encop3"), top("middown"), "", rad=0.0)
    # upsampling (bilinear x2)
    arrow(ax, top("midop"), bot("dec0"), "bilinear x2", off=(-0.5, -0.4))
    arrow(ax, top("decop0"), bot("dec1"), "bilinear x2", off=(-0.5, -0.4))
    arrow(ax, top("decop1"), bot("dec2"), "bilinear x2", off=(-0.5, -0.4))
    arrow(ax, top("dec2"), bot("dec3"), "bilinear x2", off=(-0.5, -0.4))
    # skips
    for src, dst, lab in (("enc0", "dec3", f"skip 0: {sz(r['enc0'])}"), ("enc1", "dec2", f"skip 1: {sz(r['enc1'])}"),
                          ("encop2", "dec1", f"skip 2: {sz(r['encop2'])}"), ("encop3", "dec0", f"skip 3: {sz(r['encop3'])}")):
        arrow(ax, right(src), left(dst), lab + " (concat)", ls="--", color="#c0392b", off=(0, 0.1))
    # input / output
    ax.text(-0.35, y(0) + H + 0.55, "Input 512x512: channel 1 = SEM (8-64 frame mean)\n[channel 2 = CAD template, SEM+CAD models only]",
            fontsize=8.5, ha="left", va="center")
    arrow(ax, (0.9, y(0) + H + 0.28), (0.9, y(0) + H + 0.02))
    # Grad-CAM markers
    for key, lab in (("midop", "Grad-CAM 'mid' (1/16)"), ("decop1", "Grad-CAM 'dec_op1' (1/4)"), ("dec3", "Grad-CAM 'last' (1/1)")):
        x0, y0, w_ = bx[key]
        ax.add_patch(FancyBboxPatch((x0 - 0.06, y0 - 0.06), w_ + 0.12, H + 0.12, boxstyle="round,pad=0.02,rounding_size=0.14",
                                    fc="none", ec="#d62728", lw=2.2, ls=(0, (3, 2))))
        ax.text(x0 + w_ / 2, y0 - 0.28, lab, ha="center", va="center", fontsize=8, color="#d62728", weight="bold")
    # legend
    lx, ly = 0.0, y(4) - 0.2
    for k, (c, t) in enumerate(((CONVCOL, "ConvBlock: 3x3 conv - GroupNorm - GELU"), (oc, f"Operator block ({NAMES[op]}), pre-norm residual"),
                                (IOCOL, "Stride-2 conv"), ("#f6d9d9", "Output head"))):
        ax.add_patch(FancyBboxPatch((lx, ly - 0.55 * k), 0.4, 0.3, boxstyle="round,pad=0.01", fc=c, ec="#333"))
        ax.text(lx + 0.55, ly - 0.55 * k + 0.15, t, va="center", fontsize=8)
    ax.text(lx, ly - 2.45, "Red dashes: layers used for Grad-CAM (E11)\nRed arrows: skip connections (channel concat)\n"
            "Sizes are H x W x C for a 512x512 input.", fontsize=8, va="top", color="#444")
    n = f"{n_params(m1) / 1e6:.2f} M (SEM-only) / {n_params(m2) / 1e6:.2f} M (SEM+CAD)"
    ax.set_title(f"TSDF U-Net with {NAMES[op]} operator blocks   |   base width w = {w}, channels {ch}, parameters {n}\n"
                 "Operator blocks at 5 fixed positions: encoder levels 2 and 3 (1/4, 1/8), bottleneck (1/16), decoder levels 0 and 1 (1/8, 1/4)",
                 fontsize=11.5, pad=12)
    fig.savefig(os.path.join(OUT, f"unet_{op}.png"), dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, f"unet_{op}.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dict(op=op, width=w, channels=ch, params_semonly=n_params(m1), params_cad=n_params(m2),
                op_blocks_params=sum(p.numel() for mm in [*m1.enc_op.values(), m1.mid[2], *m1.dec_op.values()] for p in mm.parameters()))


def operator_figure(cfg):
    """Inside one operator block: regular conv, DCNv2, DCNv3, DCNv4 (C = channels at the block, example C = 8w)."""
    fig, axs = plt.subplots(2, 2, figsize=(17, 10.5))
    specs = {
        "conv": ("Regular convolution", [("3x3 conv C->C", CONVCOL)], None, None),
        "dcnv2": ("DCNv2  (torchvision DeformConv2d)", None, None, None),
        "dcnv3": ("DCNv3  (official InternImage PyTorch path)", None, None, None),
        "dcnv4": ("DCNv4  (PyTorch reimplementation of the module)", None, None, None)}

    def stack(ax, items, title, note):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); ax.set_title(title, fontsize=11)
        for (x, y_, w_, h_, txt, col) in items:
            ax.add_patch(FancyBboxPatch((x, y_), w_, h_, boxstyle="round,pad=0.02,rounding_size=0.15", fc=col, ec="#333", lw=1.2))
            ax.text(x + w_ / 2, y_ + h_ / 2, txt, ha="center", va="center", fontsize=8.2)
        ax.text(0.2, 0.3, note, fontsize=7.8, va="bottom", color="#444")

    def a(ax, p, q, t=None, off=(0.15, 0)):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.1, color="#333"))
        if t:
            ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], t, fontsize=7.3, color="#444", va="center")

    g, c_ = OPCOL, "#ffffff"
    # conv
    ax = axs[0, 0]
    stack(ax, [(3, 7.0, 4, 1.1, "input x  (H x W x C)", IOCOL), (3, 4.8, 4, 1.3, "3x3 conv  C -> C\n(9 C^2 + C weights)", g["conv"]),
               (3, 2.6, 4, 1.1, "output  (H x W x C)", IOCOL)], "Regular convolution (baseline)",
          "Fixed 3x3 sampling grid.")
    a(ax, (5, 7.0), (5, 6.1)); a(ax, (5, 4.8), (5, 3.7))
    # dcnv2
    ax = axs[0, 1]
    stack(ax, [(0.4, 7.8, 3.2, 1.0, "input x", IOCOL), (0.4, 5.3, 3.2, 1.5, "3x3 conv C -> 27\n(zero-initialised)", c_),
               (4.4, 5.3, 5.0, 1.5, "27 channels split into\n18 offsets (dy, dx for 9 points)\nand 9 mask values, m = 2 sigmoid(.)", "#fbe3c4"),
               (3.0, 2.4, 4.4, 1.5, "DeformConv2d 3x3  C -> C\nsamples x at p + offset, times mask", g["dcnv2"]),
               (3.0, 0.7, 4.4, 1.0, "output (H x W x C)", IOCOL)],
          "DCNv2: one offset set and one mask for all channels", "Zero init: starts as a regular conv. Float32 inside bf16 autocast.")
    a(ax, (2.0, 7.8), (2.0, 6.8)); a(ax, (3.6, 6.05), (4.4, 6.05)); a(ax, (6.9, 5.3), (6.0, 3.9)); a(ax, (5.2, 2.4), (5.2, 1.7))
    ax.add_patch(FancyArrowPatch((0.4, 8.3), (0.4, 3.15), arrowstyle="-", lw=1.0, color="#777", ls="--"))
    ax.add_patch(FancyArrowPatch((0.4, 3.15), (3.0, 3.15), arrowstyle="-|>", mutation_scale=10, lw=1.0, color="#777", ls="--"))
    # dcnv3
    ax = axs[1, 0]
    stack(ax, [(0.2, 8.3, 2.6, 0.9, "input x (N,H,W,C)", IOCOL),
               (0.2, 6.7, 2.6, 1.2, "input_proj\nLinear C -> C", c_),
               (0.2, 4.7, 2.6, 1.5, "depth-wise 3x3 conv\nLayerNorm, GELU", c_),
               (3.4, 5.2, 3.2, 1.0, "offset  Linear\nC -> G x 9 x 2", "#cfeedb"), (3.4, 3.8, 3.2, 1.0, "mask  Linear C -> G x 9\nsoftmax over the 9 points", "#cfeedb"),
               (7.0, 3.4, 2.8, 2.2, "DCNv3 sampling core\nG = C/16 groups\n16 ch/group\nbilinear sampling,\nweighted sum", g["dcnv3"]),
               (7.0, 1.9, 2.8, 1.0, "output_proj\nLinear C -> C", c_), (7.0, 0.4, 2.8, 0.9, "output (H x W x C)", IOCOL)],
          "DCNv3: per-group offsets, softmax-normalised modulation", "Offsets/mask are predicted from a depth-wise-conv context branch.")
    a(ax, (1.5, 8.3), (1.5, 7.9)); a(ax, (1.5, 6.7), (1.5, 6.2)); a(ax, (2.8, 5.4), (3.4, 5.7)); a(ax, (2.8, 5.0), (3.4, 4.3))
    a(ax, (6.6, 5.7), (7.0, 5.2)); a(ax, (6.6, 4.3), (7.0, 4.2)); a(ax, (8.4, 3.4), (8.4, 2.9)); a(ax, (8.4, 1.9), (8.4, 1.3))
    ax.add_patch(FancyArrowPatch((2.8, 7.3), (8.4, 7.3), arrowstyle="-", lw=1.1, color="#333"))
    a(ax, (8.4, 7.3), (8.4, 5.6)); ax.text(4.2, 7.45, "value path: input_proj output", fontsize=7.3, color="#444")
    # dcnv4
    ax = axs[1, 1]
    stack(ax, [(0.2, 8.3, 2.6, 0.9, "input x (N,H,W,C)", IOCOL),
               (0.2, 6.2, 2.6, 1.3, "value_proj\nLinear C -> C", c_),
               (3.4, 6.2, 3.4, 1.3, "offset_mask  Linear (zero-init)\nC -> 3 x G x 9 (padded to x8)", "#e4d9f5"),
               (3.4, 4.5, 3.4, 1.1, "per group: 18 offsets + 9 mask\n(mask NOT softmax-normalised)", "#e4d9f5"),
               (2.0, 2.2, 4.8, 1.6, "DCN sampling core (shared with DCNv3)\nG = C/16 groups, 16 ch/group", g["dcnv4"]),
               (7.2, 2.2, 2.4, 1.6, "output_proj\nLinear C -> C", c_), (7.2, 0.7, 2.4, 0.9, "output (H x W x C)", IOCOL)],
          "DCNv4: no depth-wise branch, unnormalised modulation", "Reimplementation of the module semantics; the official CUDA op did not build here.")
    a(ax, (1.5, 8.3), (1.5, 7.5)); a(ax, (2.8, 8.7), (5.1, 7.5), None); a(ax, (5.1, 6.2), (5.1, 5.6)); a(ax, (5.1, 4.5), (5.1, 3.8))
    a(ax, (1.5, 6.2), (3.0, 3.8)); a(ax, (6.8, 3.0), (7.2, 3.0)); a(ax, (8.4, 2.2), (8.4, 1.6))
    fig.suptitle("Inside one operator block. The block is x + Op(GELU(GroupNorm(x))); C = 4w, 8w, 8w, 8w, 4w at the five positions "
                 "(e.g. C = 288 and G = 18 groups for w = 36)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "operator_blocks.png"), dpi=170, facecolor="white")
    fig.savefig(os.path.join(OUT, "operator_blocks.svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    rows = [unet_figure(cfg, op) for op in ("conv", "dcnv2", "dcnv3", "dcnv4")]
    operator_figure(cfg)
    with open(os.path.join(OUT, "architecture_table.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    for r in rows:
        print(r)
