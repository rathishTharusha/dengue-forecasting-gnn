"""Build the paper's tables and figures from the reproduction notebook's own outputs.

Every number in ``full_paper/overleaf/tables_generated.tex`` and the ``fig_*`` files
comes from the files the Kaggle notebook (``full_paper/kaggle/dengue_physics_gnn.ipynb``)
wrote in its run, copied to ``full_paper/outputs/kaggle_run/``:

* ``runs.json``        one row per configuration x origin x seed (plus persistence)
* ``levers.csv``       every lever paired against its control, BH across the table
* ``rescue.csv``       gated SEIR vs its no-physics twin, per encoder
* ``comparisons.csv``  B vs persistence and SEIR-GNN vs SEIR-LSTM, three and nine origins
* ``results.json``     headline numbers

No statistic is recomputed here: the tables quote the notebook's values, so the paper
and the notebook cannot disagree.

    python scripts/build_paper_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "full_paper" / "outputs" / "kaggle_run"
PAPER = REPO / "full_paper" / "overleaf"
FIGS = PAPER / "figures"
ENC = ["AAGCN", "ASTGCN", "LSTM", "STGAT", "A3TGCN", "DCRNN"]
GROUPS = {"worked": "Worked", "graph/architecture": "Graph and architecture",
          "literature remedy": "Literature remedies", "data": "Data", "physics": "Physics"}


def fmt_p(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def tex(label: str) -> str:
    if not label.startswith("k-NN"):
        label = label[0].upper() + label[1:]
    return (label.replace("k-NN", "$k$-NN").replace(" B ", " $B$ ").replace("model B", "model $B$")
            .replace("lags 2-", "lags 2--"))


def load():
    runs = pd.DataFrame(json.loads((RUN / "runs.json").read_text(encoding="utf-8")))
    return (runs[runs.origin_set == "three"], runs[runs.origin_set == "nine"],
            pd.read_csv(RUN / "levers.csv"), pd.read_csv(RUN / "rescue.csv"),
            pd.read_csv(RUN / "comparisons.csv"))


def mean(df, name, metric="val_RMSE"):
    return float(df.loc[df.name == name, metric].mean())


def table_encoders(r3, out):
    out += ["\\begin{table}[t]", "\\centering",
            "\\caption{The six encoders on one harness (corrected data, 3 origins $\\times$ 3 seeds,"
            " squared error, no seasonal features). Mean validation / test RMSE. The pure SEIR"
            " decoder (\\texttt{foi}) is worse than persistence on every encoder; the gated SEIR head"
            " (\\texttt{foi\\_res}) brings all six close together, below persistence.}",
            "\\label{tab:encoders}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabular}{lccc}", "\\toprule",
            "encoder & direct & SEIR decoder & gated SEIR \\\\", "\\midrule"]
    for e in ENC:
        out.append(f"{e} & " + " & ".join(f"{mean(r3, f'{e}+{h}'):.2f} / {mean(r3, f'{e}+{h}', 'RMSE'):.2f}"
                                          for h in ("direct", "foi", "foi_res")) + " \\\\")
    out += ["\\midrule", f"persistence & \\multicolumn{{3}}{{c}}{{{mean(r3, 'persistence'):.2f} / "
                         f"{mean(r3, 'persistence', 'RMSE'):.2f}}} \\\\",
            "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]


def table_levers(lev, out):
    out += ["\\begin{table}[t]", "\\centering",
            "\\caption{Every lever, paired against its own control on matched (origin, seed) runs."
            " $\\Delta$ is validation RMSE (negative is better); wins count the 9 units where the"
            " arm is better; $p_{\\text{adj}}$ is BH across this table. Test $\\Delta$ is shown"
            " and decided nothing. With three origins these $p$-values describe run-to-run"
            " stability, not the series (\\S\\ref{sec:protocol}).}",
            "\\label{tab:levers}", "\\small", "\\setlength{\\tabcolsep}{1.8pt}",
            "\\begin{tabular}{lrcrr}", "\\toprule",
            "lever & $\\Delta$ val & wins & $p_{\\text{adj}}$ & $\\Delta$ test \\\\", "\\midrule"]
    for group, g in lev.groupby("group", sort=False):
        out.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{GROUPS[group]}}}}} \\\\")
        for r in g.to_dict("records"):
            out.append(f"\\quad {tex(r['lever'])} & {r['val delta']:+.2f} & {r['wins']} & {fmt_p(r['p_adj'])} & "
                       f"{r['test delta']:+.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]


def table_nine(cmp, out):
    out += ["\\begin{table*}[t]", "\\centering",
            "\\caption{The best model and the SEIR-GNN against their controls. Three-origin rows pair"
            " 9 (origin, seed) runs; nine-origin rows pair the 9 disjoint origins (seeds averaged),"
            " the unit a claim about the series rests on (attainable $p$ floor 0.004). Raw"
            " two-sided $p$.}",
            "\\label{tab:nine}", "\\small", "\\setlength{\\tabcolsep}{5pt}",
            "\\begin{tabular}{lrcrrcr}", "\\toprule",
            " & \\multicolumn{3}{c}{validation} & \\multicolumn{3}{c}{test} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
            "comparison & $\\Delta$ & wins & $p$ & $\\Delta$ & wins & $p$ \\\\", "\\midrule"]
    for r in cmp.to_dict("records"):
        label = r["comparison"].replace(" vs ", " vs ").replace("B ", "$B$ ", 1) if r["comparison"].startswith("B ") \
            else r["comparison"]
        out.append(f"{label} & {r['val delta']:+.2f} & {r['val wins']} & {fmt_p(r['val p'])} & "
                   f"{r['test delta']:+.2f} & {r['test wins']} & {fmt_p(r['test p'])} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]


def table_rescue(r3, res, out):
    out += ["\\begin{table}[t]", "\\centering",
            "\\caption{Anchor, gate or physics? Each encoder behind four heads that add, in turn, the"
            " persistence anchor, the learned gate and the SEIR simulator (mean validation RMSE;"
            " configuration of Table~\\ref{tab:encoders}). The last two columns pair the gated SEIR"
            " head with its no-physics twin (\\texttt{gated}); negative favours the physics, and the"
            " count is runs where it wins.}",
            "\\label{tab:rescue}", "\\small", "\\setlength{\\tabcolsep}{2.5pt}",
            "\\begin{tabular}{lrrrrrr}", "\\toprule",
            " & & & & gated & \\multicolumn{2}{c}{SEIR $-$ gated} \\\\", "\\cmidrule(lr){6-7}",
            "encoder & direct & resid. & gated & SEIR & val & test \\\\", "\\midrule"]
    res = res.set_index("encoder")
    for e in ENC:
        vals = " & ".join(f"{mean(r3, f'{e}+{h}'):.2f}" for h in ("direct", "residual", "gated", "foi_res"))
        r = res.loc[e]
        out.append(f"{e} & {vals} & {r['SEIR - gated (val)']:+.2f} ({r['wins']}) & "
                   f"{r['SEIR - gated (test)']:+.2f} ({r['test wins']}) \\\\")
    out += ["\\midrule", f"persistence & \\multicolumn{{4}}{{c}}{{{mean(r3, 'persistence'):.2f}}} & & \\\\",
            "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_encoders(r3):
    heads = (("direct", "direct", "#9aa5b1"), ("foi", "SEIR decoder", "#e0a458"), ("foi_res", "gated SEIR", "#3d7ea6"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    x = np.arange(len(ENC))
    for ax, metric, title in ((axes[0], "val_RMSE", "validation"), (axes[1], "RMSE", "test")):
        for k, (h, lab, col) in enumerate(heads):
            ax.bar(x + (k - 1) * 0.27, [mean(r3, f"{e}+{h}", metric) for e in ENC], 0.27, label=lab, color=col)
        ax.axhline(mean(r3, "persistence", metric), color="k", ls="--", lw=1, label="persistence")
        ax.set_xticks(x, ENC, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("RMSE", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "fig_encoders")


def fig_levers(lev):
    lv = lev[~lev.lever.str.contains("vs persistence")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.4, 4.6))
    y = np.arange(len(lv))[::-1]
    wins = lv.wins.str.split("/").str[0].astype(int)
    d = lv["val delta"]
    cols = np.where((d < 0) & (wins >= 7), "#2a9d8f", np.where(d < 0, "#9aa5b1", "#e76f51"))
    ax.barh(y, d, color=cols)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y, [lab.replace("k-NN", "k-NN") for lab in lv.lever], fontsize=6.3)
    ax.set_xlabel("$\\Delta$ validation RMSE vs own control", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    lo, hi = min(-1.0, d.min() - 0.1), max(1.2, d.max() + 0.45)
    ax.set_xlim(lo, hi)
    for yy, w in zip(y, lv.wins):
        ax.text(hi - 0.02, yy, w, va="center", ha="right", fontsize=6)
    fig.tight_layout()
    save(fig, "fig_levers")


def fig_nine(r9):
    arms = (("B", "$B$ (direct)", "#9aa5b1"), ("AAGCN+foi_res, NB+season", "gated SEIR-GNN", "#3d7ea6"),
            ("LSTM+foi_res, NB+season", "SEIR-LSTM", "#e0a458"), ("persistence", "persistence", "k"))
    origins = sorted(r9.origin.unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, metric, title in ((axes[0], "val_RMSE", "validation"), (axes[1], "RMSE", "test")):
        for name, lab, col in arms:
            ax.plot(range(len(origins)), [r9[(r9.name == name) & (r9.origin == o)][metric].mean() for o in origins],
                    marker="o", ms=3, lw=1, color=col, label=lab, ls="--" if name == "persistence" else "-")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([15, 20, 30, 50, 100, 150])
        ax.set_xticks(range(len(origins)), [f"{o:.2f}" for o in origins], fontsize=6)
        ax.set_xlabel("forecast origin (fraction of series)", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].set_ylabel("RMSE (log scale)", fontsize=8)
    axes[1].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, "fig_nine")


def main() -> None:
    import matplotlib.ticker  # noqa: F401

    r3, r9, lev, res, cmp = load()
    out = ["% Generated by scripts/build_paper_assets.py from the reproduction notebook's outputs"
           " (full_paper/outputs/kaggle_run) -- do not edit by hand.", ""]
    table_encoders(r3, out)
    table_levers(lev, out)
    table_nine(cmp, out)
    table_rescue(r3, res, out)
    (PAPER / "tables_generated.tex").write_text("\n".join(out), encoding="utf-8")
    fig_encoders(r3)
    fig_levers(lev)
    fig_nine(r9)
    print(f"tables and figures written from {RUN.relative_to(REPO)}")


if __name__ == "__main__":
    main()
