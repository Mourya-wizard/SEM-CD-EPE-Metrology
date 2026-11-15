"""Paper-style architecture figures for the TSDF U-Net variants -> results/architecture/.

  python tools/make_architecture_diagrams.py

One drawing routine produces all four U-Net figures (conv / DCNv2 / DCNv3 / DCNv4), so layout, scales and symbols are
identical and only the operator colour changes. Shapes and parameter counts are read from the real models in
src/models.py and src/operators.py (forward hooks on a real forward pass), so the figures cannot drift from the code.

Outputs (PNG 300 dpi, SVG, PDF):  unet_<op>, operator_blocks;  tables: layer_tables.md, architecture_table.csv
"""
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data_lithobench import load_config  # noqa: E402
from models import build_model, n_params  # noqa: E402
from operators import make_op  # noqa: E402

OUT = os.path.join(ROOT, "results", "architecture")
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.unicode_minus": False})

NAMES = {"conv": "regular convolution", "dcnv2": "DCNv2", "dcnv3": "DCNv3", "dcnv4": "DCNv4 (PyTorch reimplementation)"}
SHORT = {"conv": "Conv", "dcnv2": "DCNv2", "dcnv3": "DCNv3", "dcnv4": "DCNv4"}
OPCOL = {"conv": "#8c9bab", "dcnv2": "#e69f00", "dcnv3": "#3aa67f", "dcnv4": "#cc79a7"}   # colour-blind-safe family
CB, UP, SKIP, HEAD, INP = "#c9dcee", "#e4e4e4", "#f3c9c4", "#e8d3ee", "#ffffff"
INK, RED, GREY = "#1f1f1f", "#c0392b", "#555555"
SCALE = 0.62                                  # inches per data unit in the U-Net figure


def shapes(model, size, in_ch):
    """Output (C, H, W) of every named block, recorded with forward hooks on a real forward pass."""
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


def blocks_of(model):
    """(name, module) in forward order, for the parameter table."""
    out = [("stem", model.stem)]
    for i in range(4):
        out.append((f"enc{i}", model.enc[i]))
        if str(i) in model.enc_op:
            out.append((f"encop{i}", model.enc_op[str(i)]))
        if i < 3:
            out.append((f"down{i}", model.down[i]))
    out += [("middown", model.mid[0]), ("midconv", model.mid[1]), ("midop", model.mid[2])]
    for j in range(4):
        out.append((f"dec{j}", model.dec[j]))
        if str(j) in model.dec_op:
            out.append((f"decop{j}", model.dec_op[str(j)]))
    out.append(("head", model.head))
    return out


def arrow(ax, pts, color=INK, lw=1.0, ls="-", head=True, ms=9, z=3):
    """Orthogonal polyline with an arrow head on the last segment."""
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=z)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=z,
                arrowprops=dict(arrowstyle="-|>" if head else "-", lw=lw, color=color, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=ms))


# ----------------------------------------------------------------------------------------------------------------------
# U-Net figure
# ----------------------------------------------------------------------------------------------------------------------
def slab_h(S):
    return 0.5 * (math.log2(S) - 3.5)         # 512 -> 2.75, 256 -> 2.25, ..., 32 -> 0.75 (height ~ log resolution)


def slab_w(c):
    return 0.22 + 0.30 * math.sqrt(c / 576.0)  # width grows with channel count


def unet_figure(cfg, op):
    mc, size = cfg["model"], cfg["lithobench"]["crop_px"]
    w = mc["width"][op]
    m2, m1 = build_model(mc, op, 2), build_model(mc, op, 1)
    r = shapes(m2, size, 2)
    C = lambda k: r[k][0]
    S = lambda k: r[k][1]
    oc = OPCOL[op]

    def slab(lab, kind, key=None, **kw):
        d = dict(lab=lab, kind=kind, key=key)
        if key:
            d.update(c=C(key), S=S(key))
        d.update(kw)
        return d

    stages = [
        dict(name="in", gap=0.0, arrow=None, slabs=[slab("In", "in", c=1, S=size, ctext="1 (+1)")]),
        dict(name="e0", gap=0.55, arrow="", slabs=[slab("CB", "conv", "stem"), slab("CB", "conv", "enc0", skip=0)]),
        dict(name="e1", gap=1.4, arrow="3×3 conv\nstride 2", slabs=[slab("CB", "conv", "enc1", skip=1)]),
        dict(name="e2", gap=1.4, arrow="3×3 conv\nstride 2", slabs=[slab("CB", "conv", "enc2"), slab("OB", "op", "encop2", skip=2)]),
        dict(name="e3", gap=1.4, arrow="3×3 conv\nstride 2", slabs=[slab("CB", "conv", "enc3"), slab("OB", "op", "encop3", skip=3)]),
        dict(name="b", gap=1.4, arrow="3×3 conv\nstride 2", slabs=[slab("CB", "conv", "midconv"), slab("OB", "op", "midop", gc="mid")]),
        dict(name="d0", gap=1.4, arrow="bilinear\n×2", slabs=[slab("Cat", "cat", "dec0", cu=C("midop"), cs=C("encop3"), skip=3, ckey="dec0"),
                                                              slab("CB", "conv", "dec0"), slab("OB", "op", "decop0")]),
        dict(name="d1", gap=1.4, arrow="bilinear\n×2", slabs=[slab("Cat", "cat", "dec1", cu=C("decop0"), cs=C("encop2"), skip=2),
                                                              slab("CB", "conv", "dec1"), slab("OB", "op", "decop1", gc="dec_op1")]),
        dict(name="d2", gap=1.4, arrow="bilinear\n×2", slabs=[slab("Cat", "cat", "dec2", cu=C("decop1"), cs=C("enc1"), skip=1),
                                                              slab("CB", "conv", "dec2")]),
        dict(name="d3", gap=1.4, arrow="bilinear\n×2", slabs=[slab("Cat", "cat", "dec3", cu=C("dec2"), cs=C("enc0"), skip=0),
                                                              slab("CB", "conv", "dec3", gc="last")]),
        dict(name="h", gap=1.4, arrow="conv 1×1", slabs=[slab("Head", "head", "head")]),
    ]
    # channel text / width / concat sizes
    for st in stages:
        for s in st["slabs"]:
            if s["kind"] == "cat":
                s["cw"], s["ctext"] = s["cu"] + s["cs"], str(s["cu"] + s["cs"])
            else:
                s["cw"] = s["c"]
                s.setdefault("ctext", str(s["c"]))
    # x positions
    x = 0.0
    for k, st in enumerate(stages):
        x += st["gap"] if k else 0.0
        for s in st["slabs"]:
            s["x"], s["w"], s["h"] = x, slab_w(s["cw"]), slab_h(s["S"])
            s["cx"] = x + s["w"] / 2
            x += s["w"] + 0.10
        x -= 0.10
        st["x0"], st["x1"] = st["slabs"][0]["x"], x
    xend = x

    DY = 2.15
    lvl = lambda Sx: int(round(math.log2(size / Sx)))
    for st in stages:
        for s in st["slabs"]:
            s["h"] = 0.30 * (math.log2(s["S"]) - 3.33)          # height ~ log resolution (512 -> 1.70, 32 -> 0.50)
            s["yc"] = -lvl(s["S"]) * DY
            s["cy_top"], s["cy_bot"] = s["yc"] + s["h"] / 2, s["yc"] - s["h"] / 2
    BLUE = "#1f5fa8"
    xmin, ymax = -2.3, 2.5
    ymin = -4 * DY - 3.1
    fig, ax = plt.subplots(figsize=((xend - xmin + 0.5) * SCALE, (ymax - ymin) * SCALE))
    ax.set_xlim(xmin, xend + 0.5); ax.set_ylim(ymin, ymax); ax.axis("off")

    # resolution guide lines and labels (one per level)
    for L in range(5):
        Sx = size // 2 ** L
        ax.plot([xmin + 0.9, xend + 0.4], [-L * DY] * 2, color="#c8c8c8", lw=0.6, ls=(0, (1, 3)), zorder=0)
        ax.text(xmin + 0.8, -L * DY + 0.1, f"{Sx}×{Sx}", ha="right", va="bottom", fontsize=8.5)
        ax.text(xmin + 0.8, -L * DY - 0.1, f"(1/{2 ** L})", ha="right", va="top", fontsize=7.5, color=GREY)

    # slabs
    for st in stages:
        for s in st["slabs"]:
            x0, hh, ww, y0 = s["x"], s["h"], s["w"], s["yc"]
            if s["kind"] == "cat":
                wu = ww * s["cu"] / (s["cu"] + s["cs"])
                s["wu"] = wu
                ax.add_patch(Rectangle((x0, y0 - hh / 2), wu, hh, fc=UP, ec=INK, lw=0.9, zorder=2))
                ax.add_patch(Rectangle((x0 + wu, y0 - hh / 2), ww - wu, hh, fc=SKIP, ec=INK, lw=0.9, zorder=2))
            else:
                fc = {"in": INP, "conv": CB, "op": oc, "head": HEAD}[s["kind"]]
                ax.add_patch(Rectangle((x0, y0 - hh / 2), ww, hh, fc=fc, ec=INK, lw=1.2 if s["kind"] == "op" else 0.9, zorder=2))
            ax.text(s["cx"], s["cy_top"] + 0.17, s["ctext"], ha="center", va="center", fontsize=8)
            ax.text(s["cx"], s["cy_bot"] - 0.2, s["lab"], ha="center", va="center", fontsize=8.5, weight="bold" if s["kind"] == "op" else "normal")
            if s.get("gc"):
                ax.add_patch(Rectangle((x0 - 0.05, y0 - hh / 2 - 0.05), ww + 0.10, hh + 0.10, fc="none", ec=RED, lw=1.8, ls=(0, (3, 1.6)), zorder=4))
                ax.text(s["cx"], s["cy_bot"] - 0.5, f"▲ {s['gc']}", ha="center", va="center", fontsize=7.5, color=RED)

    # arrows between stages: down = conv 3x3 stride 2 (black), up = bilinear x2 (blue), flat = in / head
    for k in range(1, len(stages)):
        a, b, nm = stages[k - 1]["slabs"][-1], stages[k]["slabs"][0], stages[k]["name"]
        if nm in ("e0", "h"):
            arrow(ax, [(a["x"] + a["w"], a["yc"]), (b["x"], b["yc"])])
            if nm == "h":
                ax.text((a["x"] + a["w"] + b["x"]) / 2, a["yc"] + 0.12, "1×1", ha="center", va="bottom", fontsize=7.5)
        elif nm in ("e1", "e2", "e3", "b"):
            arrow(ax, [(a["x"] + a["w"], a["yc"] - 0.36 * a["h"]), (b["x"], b["yc"] + 0.0)])
        else:
            arrow(ax, [(a["x"] + a["w"], a["yc"]), (b["x"], b["yc"] - 0.34 * b["h"])], color=BLUE)

    # skip connections (horizontal at each level)
    src = {s["skip"]: s for st in stages for s in st["slabs"] if s["kind"] != "cat" and "skip" in s}
    dst = {s["skip"]: s for st in stages for s in st["slabs"] if s["kind"] == "cat"}
    for k in range(4):
        a, b = src[k], dst[k]
        arrow(ax, [(a["x"] + a["w"], a["yc"]), (b["x"], b["yc"])], color=RED, lw=1.1, ls=(0, (4, 2)), ms=8)
        ax.text((a["x"] + a["w"] + b["x"]) / 2, a["yc"] + 0.07, f"skip {k}: concat", ha="center", va="bottom", fontsize=8, color=RED, style="italic")

    ax.text(0.0, 1.62, "Encoder", fontsize=9.5, style="italic", ha="left", va="center")
    ax.text(xend, 1.62, "Decoder", fontsize=9.5, style="italic", ha="right", va="center")
    ax.text((src[3]["x"] + dst[3]["x"]) / 2 + 0.0, -4 * DY - 1.15, "Bottleneck", fontsize=9.5, style="italic", ha="center", va="center")

    # legend (two rows)
    lx, ly = xmin + 0.1, -4 * DY - 1.75
    def swatch(cx, y, fc, special=None):
        if special:
            ax.add_patch(Rectangle((cx, y - 0.14), 0.15, 0.28, fc=UP, ec=INK, lw=0.8)); ax.add_patch(Rectangle((cx + 0.15, y - 0.14), 0.15, 0.28, fc=SKIP, ec=INK, lw=0.8))
        else:
            ax.add_patch(Rectangle((cx, y - 0.14), 0.3, 0.28, fc=fc, ec=INK, lw=0.8))
    cx = lx
    for txt, fc, sp in (("CB  ConvBlock: 3×3 conv, GroupNorm, GELU", CB, None), (f"OB  Operator block ({SHORT[op]}): x + Op(GELU(GN(x)))", oc, None),
                        ("Cat  Concat: upsampled | skip", None, "cat"), ("1×1 conv → TSDF head", HEAD, None)):
        swatch(cx, ly, fc, sp); ax.text(cx + 0.42, ly, txt, va="center", fontsize=8); cx += 0.42 + 0.108 * len(txt) + 0.5
    cx = lx; ly2 = ly - 0.5
    for col, ls_, txt in ((INK, "-", "3×3 conv, stride 2"), (BLUE, "-", "bilinear ×2"), (RED, (0, (4, 2)), "skip connection (channel concat)")):
        arrow(ax, [(cx, ly2), (cx + 0.7, ly2)], color=col, ls=ls_, ms=8)
        ax.text(cx + 0.85, ly2, txt, va="center", fontsize=8); cx += 0.85 + 0.108 * len(txt) + 0.5
    ax.add_patch(Rectangle((cx, ly2 - 0.14), 0.3, 0.28, fc="none", ec=RED, lw=1.6, ls=(0, (3, 1.6))))
    ax.text(cx + 0.42, ly2, "Grad-CAM layer (E11)", va="center", fontsize=8, color=RED)
    ax.text(lx, ly2 - 0.5, f"Slab height ~ log resolution, width ~ channels (numbers above slabs). Input 1 channel (SEM) or 2 channels (SEM + CAD); output: 1-channel TSDF at {size}×{size}.",
            fontsize=7.5, color=GREY, va="center")
    ax.text(xmin + 0.1, ymax - 0.05,
            f"{NAMES[op][0].upper() + NAMES[op][1:]} U-Net   ·   $w$ = {w}   ·   {n_params(m1) / 1e6:.2f} M parameters (SEM-only), {n_params(m2) / 1e6:.2f} M (SEM + CAD)",
            fontsize=11.5, weight="bold", va="top")
    for ext, kw in (("png", dict(dpi=300)), ("svg", {}), ("pdf", {})):
        fig.savefig(os.path.join(OUT, f"unet_{op}.{ext}"), bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)
    blocks = [(n, tuple(r[n]), sum(p.numel() for p in m.parameters())) for n, m in blocks_of(m2)]
    return dict(op=op, width=w, channels=[w, 2 * w, 4 * w, 8 * w], params_semonly=n_params(m1), params_cad=n_params(m2),
                op_blocks_params=sum(p.numel() for mm in [*m1.enc_op.values(), m1.mid[2], *m1.dec_op.values()] for p in mm.parameters())), blocks


# ----------------------------------------------------------------------------------------------------------------------
# Operator-block figure
# ----------------------------------------------------------------------------------------------------------------------
def node(ax, cx, cy, w, h, text, fc="#ffffff", bold=False, fs=8.6):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc=fc, ec=INK, lw=1.0, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", zorder=3, linespacing=1.15)
    return dict(t=(cx, cy + h / 2), b=(cx, cy - h / 2), l=(cx - w / 2, cy), r=(cx + w / 2, cy), cx=cx, cy=cy, w=w, h=h)


def operator_figure(cfg):
    w8 = cfg["model"]["width"]
    fig = plt.figure(figsize=(16, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.95, 3.9], hspace=0.02, wspace=0.06, left=0.02, right=0.98, top=0.95, bottom=0.03)
    # (a) residual wrapper
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 40); ax.set_ylim(0, 6.6); ax.set_aspect("equal"); ax.axis("off")
    ax.text(0, 6.5, "(a) Operator block used at the five positions (identical for all variants)", fontsize=11, weight="bold", va="top")
    cy, bh = 2.0, 1.5
    n_in = node(ax, 3.2, cy, 4.6, bh, "input $x$\n$H\\times W\\times C$", INP)
    n_gn = node(ax, 10.0, cy, 4.6, bh, "GroupNorm", CB)
    n_ge = node(ax, 16.4, cy, 4.0, bh, "GELU", CB)
    n_op = node(ax, 24.0, cy, 8.2, bh, "Op$(\\cdot)$\nConv | DCNv2 | DCNv3 | DCNv4", "#ffffff")
    for k, key in enumerate(OPCOL):
        ax.add_patch(Rectangle((19.9 + k * 2.05, cy - bh / 2 - 0.02), 2.05, 0.2, fc=OPCOL[key], ec=INK, lw=0.6, zorder=3))
    ax.add_patch(Circle((31.2, cy), 0.62, fc="#ffffff", ec=INK, lw=1.0, zorder=2))
    ax.text(31.2, cy, "+", ha="center", va="center", fontsize=13, zorder=3)
    n_out = node(ax, 36.4, cy, 4.6, bh, "output $y$\n$H\\times W\\times C$", INP)
    for a_, b_ in ((n_in["r"], n_gn["l"]), (n_gn["r"], n_ge["l"]), (n_ge["r"], n_op["l"])):
        arrow(ax, [a_, b_])
    arrow(ax, [n_op["r"], (30.58, cy)]); arrow(ax, [(31.82, cy), n_out["l"]])
    ax.plot([5.9], [cy], marker="o", ms=3.5, color=INK, zorder=4)
    arrow(ax, [(5.9, cy), (5.9, 4.0), (31.2, 4.0), (31.2, cy + 0.62)])
    ax.text(18.5, 4.12, "identity (residual)", ha="center", va="bottom", fontsize=8, style="italic")
    ax.text(0, 0.1, "$y = x + \\mathrm{Op}(\\mathrm{GELU}(\\mathrm{GN}(x)))$;  $C$ = 4$w$, 8$w$, 8$w$, 8$w$, 4$w$ at encoder levels 2 and 3, bottleneck, decoder levels 0 and 1.",
            fontsize=8.5, color=GREY, va="bottom")

    def panel(col, title):
        a = fig.add_subplot(gs[1, col]); a.set_xlim(0, 10); a.set_ylim(0, 14.4); a.set_aspect("equal"); a.axis("off")
        a.text(0.1, 14.3, title, fontsize=11, weight="bold", va="top")
        return a

    BH = 1.5
    def params_note(a, op):
        c = 8 * w8[op]
        n = sum(p.numel() for p in make_op(op, c).parameters())
        a.text(0.1, 0.0, f"{n / 1e3:.0f} k parameters at $C$ = {c} ($w$ = {w8[op]})", fontsize=8.5, color=GREY, va="bottom")

    cb = lambda op: OPCOL[op]
    # (b) regular convolution
    a = panel(0, "(b) Regular convolution")
    x_ = node(a, 5, 12.0, 4.4, 1.2, "input $x$ ($H\\times W\\times C$)", INP)
    c_ = node(a, 5, 8.0, 7.0, 1.9, "Conv 3×3, $C\\rightarrow C$\nstride 1, padding 1, bias", cb("conv"), bold=True)
    o_ = node(a, 5, 4.0, 4.4, 1.2, "output ($H\\times W\\times C$)", INP)
    arrow(a, [x_["b"], c_["t"]]); arrow(a, [c_["b"], o_["t"]])
    a.text(5.2, 10.1, "fixed 3×3 grid", fontsize=8, color=GREY, style="italic", va="center")
    params_note(a, "conv")
    # (c) DCNv2
    a = panel(1, "(c) DCNv2")
    x_ = node(a, 5, 12.3, 4.6, 1.2, "input $x$", INP)
    om = node(a, 7.5, 9.6, 4.6, BH, "Conv 3×3, $C\\rightarrow$ 27\n(zero-initialised)", "#ffffff")
    sp = node(a, 7.5, 7.0, 4.6, 1.9, "18 offsets ($\\Delta y,\\Delta x$ ×9)\n9 mask values\n$m=2\\sigma(\\cdot)$", "#f9e2b6")
    dc = node(a, 5, 4.2, 8.8, 1.7, "DeformConv2d 3×3, $C\\rightarrow C$\nsample $x$ at $p+\\Delta p$, weight by $m$", cb("dcnv2"), bold=True)
    o_ = node(a, 5, 1.9, 4.6, 1.0, "output", INP)
    arrow(a, [(6.6, x_["b"][1]), (6.6, om["t"][1])])
    arrow(a, [(3.6, x_["b"][1]), (3.6, dc["t"][1])]); a.text(3.35, 8.0, "value", fontsize=8, rotation=90, ha="right", va="center", style="italic")
    arrow(a, [om["b"], sp["t"]]); arrow(a, [(7.5, sp["b"][1]), (7.5, dc["t"][1])]); arrow(a, [dc["b"], o_["t"]])
    params_note(a, "dcnv2")
    # (d) DCNv3
    a = panel(2, "(d) DCNv3")
    x_ = node(a, 5, 12.3, 4.6, 1.2, "input $x$", INP)
    ip = node(a, 2.0, 9.9, 3.4, BH, "input_proj\nLinear $C\\rightarrow C$", "#ffffff")
    dw = node(a, 7.0, 9.9, 5.6, BH, "depth-wise Conv 3×3\nLayerNorm, GELU", "#ffffff")
    of = node(a, 5.6, 7.1, 2.6, 1.7, "offset\nLinear\n$C\\rightarrow 18G$", "#cfeedc", fs=8.2)
    mk = node(a, 8.5, 7.1, 2.6, 1.7, "mask\nLinear $C\\rightarrow 9G$\nsoftmax (9)", "#cfeedc", fs=8.2)
    dc = node(a, 5, 4.4, 9.0, 1.7, "DCNv3 sampling, $G$ = $C$/16 groups\nbilinear, 9 points per group", cb("dcnv3"), bold=True)
    op_ = node(a, 5, 2.4, 4.6, 1.0, "output_proj  Linear $C\\rightarrow C$", "#ffffff", fs=8.2)
    arrow(a, [(3.4, x_["b"][1]), (3.4, ip["t"][1])])
    arrow(a, [(6.6, x_["b"][1]), (6.6, dw["t"][1])])
    arrow(a, [(5.6, dw["b"][1]), (5.6, of["t"][1])]); arrow(a, [(8.5, dw["b"][1]), (8.5, mk["t"][1])])
    arrow(a, [(2.0, ip["b"][1]), (2.0, dc["t"][1])]); a.text(1.75, 7.1, "value", fontsize=8, rotation=90, ha="right", va="center", style="italic")
    arrow(a, [(5.6, of["b"][1]), (5.6, dc["t"][1])]); arrow(a, [(8.5, mk["b"][1]), (8.5, dc["t"][1])])
    arrow(a, [dc["b"], op_["t"]])
    o_ = node(a, 5, 0.95, 4.6, 0.8, "output", INP, fs=8.2)
    arrow(a, [op_["b"], o_["t"]])
    params_note(a, "dcnv3")
    # (e) DCNv4
    a = panel(3, "(e) DCNv4 (reimplementation)")
    x_ = node(a, 5, 12.3, 4.6, 1.2, "input $x$", INP)
    vp = node(a, 2.0, 9.9, 3.4, BH, "value_proj\nLinear $C\\rightarrow C$", "#ffffff")
    om = node(a, 7.0, 9.9, 5.6, BH, "offset_mask  Linear\n$C\\rightarrow 27G$ (zero-init)", "#ffffff")
    sp = node(a, 7.0, 7.1, 5.6, 1.7, "per group: 18 offsets, 9 mask\nmask not normalised (no softmax)", "#e8d9f3", fs=8.2)
    dc = node(a, 5, 4.4, 9.0, 1.7, "DCN sampling (shared core), $G$ = $C$/16\nbilinear, 9 points per group", cb("dcnv4"), bold=True)
    op_ = node(a, 5, 2.4, 4.6, 1.0, "output_proj  Linear $C\\rightarrow C$", "#ffffff", fs=8.2)
    arrow(a, [(3.4, x_["b"][1]), (3.4, vp["t"][1])]); arrow(a, [(6.6, x_["b"][1]), (6.6, om["t"][1])])
    arrow(a, [(7.0, om["b"][1]), (7.0, sp["t"][1])])
    arrow(a, [(2.0, vp["b"][1]), (2.0, dc["t"][1])]); a.text(1.75, 7.1, "value", fontsize=8, rotation=90, ha="right", va="center", style="italic")
    arrow(a, [(7.0, sp["b"][1]), (7.0, dc["t"][1])]); arrow(a, [dc["b"], op_["t"]])
    o_ = node(a, 5, 0.95, 4.6, 0.8, "output", INP, fs=8.2)
    arrow(a, [op_["b"], o_["t"]])
    params_note(a, "dcnv4")
    for ext, kw in (("png", dict(dpi=300)), ("svg", {}), ("pdf", {})):
        fig.savefig(os.path.join(OUT, f"operator_blocks.{ext}"), bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)


def write_tables(allblocks, rows):
    names = {"stem": "Stem ConvBlock", "enc0": "Encoder 0 ConvBlock", "enc1": "Encoder 1 ConvBlock", "enc2": "Encoder 2 ConvBlock",
             "enc3": "Encoder 3 ConvBlock", "encop2": "Encoder 2 operator block", "encop3": "Encoder 3 operator block",
             "down0": "Down conv 0 (3×3, stride 2)", "down1": "Down conv 1 (3×3, stride 2)", "down2": "Down conv 2 (3×3, stride 2)",
             "middown": "Bottleneck down conv (3×3, stride 2)", "midconv": "Bottleneck ConvBlock", "midop": "Bottleneck operator block",
             "dec0": "Decoder 0 ConvBlock (after concat)", "dec1": "Decoder 1 ConvBlock (after concat)", "dec2": "Decoder 2 ConvBlock (after concat)",
             "dec3": "Decoder 3 ConvBlock (after concat)", "decop0": "Decoder 0 operator block", "decop1": "Decoder 1 operator block",
             "head": "Head (1×1 conv)"}
    with open(os.path.join(OUT, "layer_tables.md"), "w", encoding="utf-8") as f:
        f.write("# Layer tables (generated by tools/make_architecture_diagrams.py)\n\nOutput shape is C×H×W for a 512×512 input (SEM + CAD model; the SEM-only "
                "model differs only in the 1-channel stem input). Parameters are for the SEM + CAD model.\n")
        for op, blocks in allblocks.items():
            f.write(f"\n## {NAMES[op]} (w = {[r for r in rows if r['op'] == op][0]['width']})\n\n| Block | Output (C×H×W) | Parameters |\n|---|---|--:|\n")
            for n, shp, p in blocks:
                f.write(f"| {names[n]} | {shp[0]}×{shp[1]}×{shp[2]} | {p:,} |\n")
            f.write(f"| **Total** | | **{[r for r in rows if r['op'] == op][0]['params_cad']:,}** |\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    rows, allblocks = [], {}
    for op in ("conv", "dcnv2", "dcnv3", "dcnv4"):
        row, blocks = unet_figure(cfg, op)
        rows.append(row); allblocks[op] = blocks
    operator_figure(cfg)
    write_tables(allblocks, rows)
    with open(os.path.join(OUT, "architecture_table.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print("done", [r["op"] for r in rows])
