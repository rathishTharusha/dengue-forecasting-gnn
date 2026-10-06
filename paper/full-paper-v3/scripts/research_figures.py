"""Fig. 5: three result panels drawn from saved outputs only (no training).

(a) RMSE by forecast horizon, nine-origin protocol
(b) test RMSE of the SEIR-head variants against persistence, nine-origin protocol
(c) persistence RMSE per legacy fold, all windows and without the windows touching the
    source-table error

Inputs: results/development_statistics.json (from publication.py) and
results/review_audit_results.json. Colours: Okabe-Ito.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from common import FIG, RES  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                     "pdf.fonttype": 42, "axes.linewidth": .6, "lines.linewidth": 1.1})
BLACK, ORANGE, SKY, GREEN, BLUE, VERM = "#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00"


def main() -> None:
    arms = json.loads((RES / "development_statistics.json").read_text(encoding="utf-8"))["arms"]
    audit = json.loads((RES / "review_audit_results.json").read_text(encoding="utf-8"))
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw={"width_ratios": [1, 1.35, 1]})
    fig.subplots_adjust(left=.065, right=.99, bottom=.2, top=.88, wspace=.55)

    # (a) horizon
    ax = axs[0]
    series = [("persistence", "Persistence", BLACK, "o", "--"),
              ("adaptive_gwn+foi_res anchor E0enc", "Adaptive SEIR-GNN", VERM, "D", "-")]
    for key, label, colour, marker, ls in series:
        s = arms[key]
        ax.plot([1, 2, 3], [s["rmse_h1"], s["rmse_h2"], s["rmse_h3"]], ls=ls, marker=marker, ms=3.5,
                color=colour, label=label, lw=1.6 if key.endswith("E0enc") else 1.1)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("Test RMSE (cases)")
    ax.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0)
    ax.set_ylim(21, 37)
    ax.set_title("(a) Error by horizon", fontsize=8.5, loc="left")

    # (b) SEIR-head variants
    ax = axs[1]
    rows = [("gcn+foi_res", "GCN, gated SEIR,\nfree rate"),
            ("gcn+foi_res anchor", "GCN, gated SEIR,\nanchored"),
            ("adaptive_gwn+foi anchor", "Adaptive GCN, bare\nSEIR, anchored"),
            ("adaptive_gwn+foi_res mass", "Adaptive GCN, gated\nSEIR, mass-action"),
            ("adaptive_gwn+foi_res anchor", "Adaptive GCN, gated\nSEIR, anchored"),
            ("adaptive_gwn+foi_res anchor E0enc", "Adaptive SEIR-GNN\n(+ learned $E_0$)")]
    y = np.arange(len(rows))[::-1]
    vals = [arms[k]["rmse"] for k, _ in rows]
    colours = [VERM if k.endswith("E0enc") else ORANGE for k, _ in rows]
    ax.barh(y, vals, color=colours, height=.62, edgecolor="none")
    ax.axvline(arms["persistence"]["rmse"], color=BLACK, ls="--", lw=.9)
    ax.text(arms["persistence"]["rmse"] + .08, y[0] + .55, "persistence", fontsize=7.5, va="center")
    for yi, v in zip(y, vals):
        ax.text(v + .08, yi, f"{v:.2f}", va="center", fontsize=7.3)
    ax.set_yticks(y, [lab for _, lab in rows], fontsize=7.2)
    ax.set_xlim(26.5, 30.6)
    ax.set_xlabel("Test RMSE (cases)")
    ax.set_title("(b) SEIR-head variants", fontsize=8.5, loc="left")

    # (c) data quality
    ax = axs[2]
    folds = audit["folds"]
    x = np.arange(len(folds))
    ax.bar(x - .19, [f["all_windows"] for f in folds], width=.36, color=BLUE, label="all windows")
    ax.bar(x + .19, [f["excluded_windows"] for f in folds], width=.36, color=SKY,
           label="error windows excluded")
    ax.set_xticks(x, [f"{f['origin']:.2f}" for f in folds])
    ax.set_xlabel("Legacy fold (origin)")
    ax.set_ylabel("Persistence RMSE (cases)")
    ax.legend(frameon=False, loc="upper left", borderaxespad=0)
    ax.set_ylim(0, 92)
    ax.set_title("(c) Effect of the source error", fontsize=8.5, loc="left")

    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "fig_results.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()


def origins_figure() -> None:
    """Fig. 6: seed-averaged test RMSE at each of the nine origins (nine-origin protocol)."""
    from common import P9_FILE

    rows = json.loads(P9_FILE.read_text(encoding="utf-8"))
    origins = sorted({r["origin"] for r in rows})

    def per_origin(name):
        return [np.mean([r["RMSE"] for r in rows if r["name"] == name and r["origin"] == o]) for o in origins]

    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    fig.subplots_adjust(left=.14, right=.98, bottom=.2, top=.97)
    for name, label, colour, marker, ls in [("persistence", "Persistence", BLACK, "o", "--"),
                                            ("adaptive_gwn+foi_res anchor E0enc", "Adaptive SEIR-GNN", VERM, "D", "-")]:
        ax.plot(origins, per_origin(name), ls=ls, marker=marker, ms=3.5, color=colour, label=label)
    ax.set_xticks(origins, [f"{o:.2f}" for o in origins], fontsize=7.5)
    ax.set_xlabel("Forecast origin (fraction of the series)")
    ax.set_ylabel("Test RMSE (cases)")
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "fig_origins.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    origins_figure()
