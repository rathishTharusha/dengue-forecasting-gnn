"""Figure: what the two scoring approaches disagree about.

Panel (a) pairs each arm's all-windows RMSE against its artifact-free RMSE.
Panel (b) decomposes the gap by rolling origin, which is the point: the whole
disagreement is one fold. Week 395 lands only in the test split of origin 0.85,
so origins 0.55 and 0.70 score identically both ways.

AAGCN carries the smallest gap. That is not a modelling win -- it is the arm
that best predicted a reporting backlog.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "paper" / "_build"))
import figures as F  # noqa: E402


def fig_scoring_gap():
    F.style()
    df = pd.read_csv(REPO / "analysis" / "results" / "improved_sweep.csv")

    per_arch = df.groupby("arch")[["RMSE", "RMSE_clean"]].mean()
    per_arch["gap"] = per_arch["RMSE"] - per_arch["RMSE_clean"]
    per_arch = per_arch.sort_values("gap")

    per_origin = df.groupby("origin")[["RMSE", "RMSE_clean"]].mean()
    per_origin["gap"] = per_origin["RMSE"] - per_origin["RMSE_clean"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.35),
                                 gridspec_kw={"width_ratios": [1.5, 1]})

    # -- (a) paired dumbbell -------------------------------------------------
    y = np.arange(len(per_arch))
    for yi, (_, r) in zip(y, per_arch.iterrows()):
        ax.plot([r.RMSE_clean, r.RMSE], [yi, yi], color=F.GREY, lw=1.8,
                zorder=1, solid_capstyle="round")
    ax.scatter(per_arch.RMSE_clean, y, s=42, color=F.BLUE, zorder=3,
               label="artifact-free", edgecolor="white", linewidth=0.8)
    ax.scatter(per_arch.RMSE, y, s=42, color=F.ORANGE, zorder=3,
               label="all windows", edgecolor="white", linewidth=0.8)

    for yi, (_, r) in zip(y, per_arch.iterrows()):
        ax.text(r.RMSE + 1.0, yi, f"+{r.gap:.1f}", va="center", ha="left",
                fontsize=6.5, color=F.INK)

    ax.set_yticks(y)
    ax.set_yticklabels(list(per_arch.index))
    ax.set_ylim(-1.55, len(per_arch) - 0.35)
    ax.set_xlim(26, 53)
    ax.set_xlabel("pooled RMSE")
    ax.set_title("(a) the same arms, scored two ways", loc="left")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="lower center", handlelength=1.0, columnspacing=1.4,
              scatterpoints=1, borderpad=0.0)

    # -- (b) the gap, by origin ---------------------------------------------
    xs = np.arange(len(per_origin))
    bars = bx.bar(xs, per_origin.gap, width=0.5, color=F.GREY)
    bars[-1].set_color(F.ORANGE)
    for xi, g in zip(xs, per_origin.gap):
        bx.text(xi, g + 1.2, f"{g:.1f}" if g else "0", ha="center",
                va="bottom", fontsize=7, color=F.INK)
    bx.set_xticks(xs)
    bx.set_xticklabels([f"{o:.2f}" for o in per_origin.index])
    bx.set_xlabel("rolling origin")
    bx.set_ylabel("RMSE gap")
    bx.set_ylim(0, per_origin.gap.max() * 1.22)
    bx.set_title("(b) all of it is one fold", loc="left")
    bx.grid(axis="y")
    bx.set_axisbelow(True)
    bx.annotate("week 395 lands\nhere only", xy=(2, 30), xytext=(0.55, 30),
                fontsize=6.5, color=F.INK, va="center",
                arrowprops=dict(arrowstyle="->", color=F.INK, lw=0.8))

    fig.tight_layout()
    return F._save(fig, "fig_scoring_gap")


if __name__ == "__main__":
    fig_scoring_gap()
    print("wrote paper/figures/fig_scoring_gap.{pdf,png}")
