"""Shared figure style for every README figure: serif (Times New Roman, STIX fallback), STIX math, sizes for ~880 px display."""
import matplotlib.pyplot as plt


def use(base=13):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": base, "axes.titlesize": base + 2, "axes.labelsize": base + 1, "xtick.labelsize": base, "ytick.labelsize": base,
        "legend.fontsize": base - 1, "figure.titlesize": base + 3, "axes.linewidth": 0.9, "pdf.fonttype": 42, "svg.fonttype": "path",
    })
