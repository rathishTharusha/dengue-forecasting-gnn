"""Vector figures for the IEEE paper, from saved files only.

Fig. 1  national weekly cases, legacy array against rebuilt series
Fig. 3  per-origin test RMSE minus persistence, nine-origin protocol
Fig. 4  validation RMSE against test RMSE for every three-origin configuration
(Fig. 2, the architecture diagram, is TikZ inside main.tex.)

Single-column width is 3.5 in. Colours are the Okabe-Ito palette.
"""
from __future__ import annotations

import collections

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import (FIG, KAGGLE, LEGACY, OI, P9_FILE, REBUILT, REBUILT_INDEX, RES,  # noqa: E402
                    load_json, load_jsonl)
from common import project_stats as ps  # noqa: E402

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 0.8,
                     "pdf.fonttype": 42, "font.family": "serif"})
W = 3.5
BACKLOG = 395


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_data() -> dict:
    legacy = np.load(LEGACY)[:, :, 5].astype(float).sum(axis=1)
    cases = np.load(REBUILT)[:, :, 0].astype(float)
    idx = pd.read_csv(REBUILT_INDEX, parse_dates=["week_start"])
    national = np.where(np.isnan(cases).all(axis=1), np.nan, np.nansum(cases, axis=1))
    fig, ax = plt.subplots(2, 1, figsize=(W, 2.45))
    ax[0].plot(np.arange(len(legacy)), legacy, color=OI["blue"])
    ax[0].axvline(BACKLOG, color=OI["vermillion"], lw=0.7, ls="--")
    ax[0].annotate("week 395\n(reporting backlog)", xy=(BACKLOG, legacy[BACKLOG]),
                   xytext=(228, 7700), fontsize=6.5, color=OI["vermillion"],
                   arrowprops=dict(arrowstyle="->", color=OI["vermillion"], lw=0.6))
    ax[0].set_xlabel("Week index in the legacy array (0 = first week)")
    ax[0].set_ylabel("Cases per week,\nall 25 districts")
    ax[0].set_title("(a) Legacy benchmark array", fontsize=7, loc="left", pad=2)
    ax[1].plot(idx["week_start"], national, color=OI["green"])
    ax[1].set_xlabel("Week start date")
    ax[1].set_ylabel("Cases per week,\nall 25 districts")
    ax[1].set_title("(b) Rebuilt series (missing weeks left blank)", fontsize=7, loc="left", pad=2)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig_data.pdf")
    return {"legacy_week395": float(legacy[BACKLOG]), "legacy_week394": float(legacy[BACKLOG - 1]),
            "legacy_week396": float(legacy[BACKLOG + 1]), "legacy_max_other": float(np.delete(legacy, BACKLOG).max()),
            "rebuilt_weeks": int(len(national)), "rebuilt_missing": int(np.isnan(national).sum()),
            "rebuilt_first": str(idx["week_start"].iloc[0].date()), "rebuilt_last": str(idx["week_start"].iloc[-1].date()),
            "rebuilt_max": float(np.nanmax(national))}


FOREST = [
    ("gcn+direct", "GCN, direct (baseline GNN)"), ("gcn+residual", "GCN, residual"),
    ("LSTM+direct", "LSTM, direct"), ("ASTGCN+direct", "ASTGCN, direct"), ("AAGCN+direct", "AAGCN, direct"),
    ("ASTGCN+residual", "ASTGCN, residual"), ("adaptive_gwn+residual", "Adaptive GCN, residual"),
    ("gcn+foi_res", "GCN, gated SEIR, free rate"),
    ("adaptive_gwn+foi anchor", "Adaptive GCN, SEIR, anchored"),
    ("adaptive_gwn+foi_res mass", "Adaptive GCN, gated SEIR,\nmass-action"),
    ("gcn+foi_res anchor", "GCN, gated SEIR, anchored"),
    ("adaptive_gwn+foi_res anchor", "Adaptive GCN, gated SEIR,\nanchored"),
    ("adaptive_gwn+foi_res anchor E0enc", "Adaptive GCN, gated SEIR,\nanchored, learned $E_0$"),
]


def fig_forest() -> dict:
    rows = load_json(P9_FILE)
    table = ps.cells(rows, "RMSE")
    ref = table["persistence"]
    out = {}
    fig, ax = plt.subplots(figsize=(W, 3.5))
    ys = np.arange(len(FOREST))[::-1]
    for y, (name, label) in zip(ys, FOREST):
        d = ps.paired(table[name], ref, "origin")           # one value per origin, seeds averaged
        ax.scatter(d, np.full(len(d), y), s=6, color=OI["sky"], zorder=2, linewidths=0)
        ax.scatter([d.mean()], [y], marker="D", s=16, color=OI["vermillion"], zorder=3, linewidths=0)
        out[name] = {"mean": float(d.mean()), "per_origin": [float(v) for v in d]}
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(ys, [lab for _, lab in FOREST], fontsize=5.8)
    ax.set_xlabel("Test RMSE minus persistence (cases)")
    ax.scatter([], [], s=6, color=OI["sky"], label="one origin (3 seeds averaged)")
    ax.scatter([], [], marker="D", s=16, color=OI["vermillion"], label="mean over 9 origins")
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=2, frameon=False, handletextpad=0.2, columnspacing=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", lw=0.3, color="0.85")
    fig.tight_layout()
    save(fig, "fig_forest.pdf")
    return out


def fig_valtest() -> dict:
    runs = [r for r in load_jsonl(KAGGLE / "runs.jsonl") if r["origin_set"] == "three"]
    res = load_json(KAGGLE / "results.json")["persistence_3"]
    by = collections.defaultdict(list)
    for r in runs:
        by[r["name"]].append(r)
    pts = {n: (np.mean([r["val_RMSE"] for r in rs]), np.mean([r["RMSE"] for r in rs])) for n, rs in by.items()}

    def family(n: str) -> str:
        if n.endswith("+foi") or n == "P4 metapopulation SEIR":
            return "SEIR head"
        if "+foi_res" in n:
            return "gated SEIR head"
        return "other"

    colours = {"SEIR head": OI["vermillion"], "gated SEIR head": OI["orange"], "other": OI["blue"]}
    fig, ax = plt.subplots(figsize=(W, 2.3))
    for fam, c in colours.items():
        xs = [pts[n][0] for n in pts if family(n) == fam]
        ys = [pts[n][1] for n in pts if family(n) == fam]
        ax.scatter(xs, ys, s=8, color=c, label=fam, linewidths=0, alpha=0.9)
    pick = min(pts, key=lambda n: pts[n][0])
    ax.scatter([pts[pick][0]], [pts[pick][1]], s=34, facecolors="none", edgecolors="k", linewidths=0.8)
    ax.annotate("lowest validation\nRMSE", xy=pts[pick], xytext=(pts[pick][0] + 3.5, pts[pick][1] + 14),
                fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.axhline(res["test"], color="k", ls="--", lw=0.6)
    ax.axvline(res["val"], color="k", ls="--", lw=0.6)
    ax.text(35.8, res["test"] - 1.2, "persistence (test)", ha="right", va="top", fontsize=6.5)
    ax.text(res["val"] + 0.3, 73.5, "persistence\n(validation)", fontsize=6.5, va="top")
    ax.set_xlim(14.5, 36)
    ax.set_ylim(32, 75)
    ax.set_xlabel("Validation RMSE (cases per district-week)")
    ax.set_ylabel("Test RMSE (cases per district-week)")
    ax.legend(loc="center right", frameon=False, handletextpad=0.2, borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_valtest.pdf")
    below = [n for n in pts if pts[n][0] < res["val"]]
    return {"n_configs": len(pts), "lowest_val": pick, "lowest_val_point": [float(v) for v in pts[pick]],
            "n_below_persistence_val": len(below),
            "of_those_test_above_persistence": int(sum(pts[n][1] > res["test"] for n in below))}


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    import fig_arch
    fig_arch.draw()
    info = {"fig_data": fig_data(), "fig_forest": fig_forest(), "fig_valtest": fig_valtest()}
    import json
    (RES / "figure_values.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "fig_forest"}, indent=2))
