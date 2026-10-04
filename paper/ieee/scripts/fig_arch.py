"""Fig. 2: schematic of the encoder, the output heads and the SEIR decoder (no data needed)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import FIG  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PALE = {"in": "#DDEEF9", "enc": "#CDE8DD", "head": "#FBE5C0", "phys": "#F4D1C0"}


def draw() -> None:
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "mathtext.fontset": "dejavuserif"})
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=6.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec="k", lw=0.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5, lw=0.5, color="k"))

    box(0.1, 5.7, 2.3, 1.7, "Input window\n3 weeks of\nlog cases,\n25 districts", PALE["in"])
    box(3.0, 5.7, 2.6, 1.7, "Graph encoder\n(GCN, adaptive\nGCN, published)", PALE["enc"], 5.8)
    box(6.1, 5.7, 1.6, 1.7, "Read-out\n$r_{i,h}$", PALE["enc"])
    box(8.1, 5.7, 1.8, 1.7, "Persistence\nanchor $p$\n(last week)", PALE["in"])
    arrow(2.4, 6.55, 3.0, 6.55)
    arrow(5.6, 6.55, 6.1, 6.55)
    box(0.1, 2.9, 3.0, 2.1,
        "Direct: $r$\nResidual: $p+r$\nGated: $p+\\sigma(a)(r-p)$", PALE["head"], 5.8)
    box(3.5, 2.9, 3.3, 2.1,
        "SEIR decoder\n$\\lambda_{i,h}$ from $r$\n(free or anchored)\ndaily $S\\to E\\to I\\to R$\n"
        "counts $=\\rho N\\times$ onsets", PALE["phys"], 5.6)
    box(7.2, 2.9, 2.7, 2.1, "Gated SEIR head\n$p+\\sigma(a)(\\hat z_{\\mathrm{SEIR}}-p)$", PALE["head"], 5.8)
    arrow(6.9, 5.7, 5.15, 5.0)
    arrow(6.9, 5.7, 1.6, 5.0)
    arrow(6.8, 3.95, 7.2, 3.95)
    arrow(9.0, 5.7, 9.0, 5.0)
    box(2.2, 0.2, 5.6, 1.6, "Forecast of cases, $h=1,2,3$ weeks\n(squared error on $z$, or NB likelihood)",
        PALE["in"])
    arrow(1.6, 2.9, 3.4, 1.8)
    arrow(8.5, 2.9, 6.7, 1.8)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_arch.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    draw()
