"""Build every number-bearing table and figure of the full paper from the result files.

Nothing in ``full_paper/overleaf/tables_generated.tex`` or the ``fig_*`` files it
references is typed by hand: each value is read from ``seirgnn2/results/*.json``
and paired with ``seirgnn2/stats.py``, the same code the experiment log used.
The result files the paper reads are also copied to ``full_paper/outputs/results``
so the standalone package carries its own evidence.

Usage::

    python scripts/build_paper_assets.py

Pairing: validation RMSE, matched (origin, seed) units, two-sided sign-flip
permutation test. BH adjustment is applied once across the whole levers table,
which is more conservative than the per-experiment families of
``docs/EXPERIMENT_LOG.md``; the log remains the record of each adoption decision.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "seirgnn2"))
import stats  # noqa: E402

RESULTS = REPO / "seirgnn2" / "results"
PAPER = REPO / "full_paper" / "overleaf"
FIGS = PAPER / "figures"
PKG = REPO / "full_paper" / "outputs" / "results"
GRIDS = ("screen", "real", "combo", "confirm", "remedies+ens", "curve", "augment",
         "climate", "arch", "physics", "physics9", "rescue")

ENCODERS = ("AAGCN", "ASTGCN", "LSTM", "STGAT", "A3TGCN", "DCRNN")

# One lever list for the paper and the notebook: defined in build_full_paper_notebook.py.
sys.path.insert(0, str(REPO / "scripts"))
from build_full_paper_notebook import LEVERS as _CANON  # noqa: E402

_GROUPS = {"worked": "Worked", "graph/architecture": "Graph and architecture",
           "literature remedy": "Literature remedies", "data": "Data", "physics": "Physics"}


def _tex(label: str) -> str:
    if not label.startswith("k-NN"):
        label = label[0].upper() + label[1:]
    return (label.replace("k-NN", "$k$-NN").replace(" B ", " $B$ ").replace("model B", "model $B$")
            .replace("lags 2-", "lags 2--"))


LEVERS = []
_seen: set[str] = set()
for _g, _label, _grid, _arm, _ref in _CANON:
    if _g not in _seen:
        _seen.add(_g)
        LEVERS.append(("\\emph{" + _GROUPS[_g] + "}", None, None, None))
    LEVERS.append((_tex(_label), _grid, _arm, _ref))


def rows(grid: str) -> list[dict]:
    out = json.loads((RESULTS / f"{grid}.json").read_text(encoding="utf-8"))
    for r in out:
        r.setdefault("name", f"{r.get('backbone')}+{r.get('head')}")
    return out


def mean(rs: list[dict], name: str, metric: str) -> float:
    v = [r[metric] for r in rs if r["name"] == name and r.get(metric) is not None]
    return float(np.mean(v)) if v else float("nan")


def pair(rs: list[dict], arm: str, ref: str, metric: str, unit: str) -> tuple[float, int, int, float]:
    c = stats.cells(rs, metric)
    d = stats.paired(c[arm], c[ref], unit)
    return float(d.mean()), int((d < 0).sum()), len(d), stats.sign_flip_p(d)


def fmt_p(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# --------------------------------------------------------------------- tables --
def table_encoders(out: list[str]) -> None:
    rs = rows("real")
    pv, pt = mean(rs, "persistence", "val_RMSE"), mean(rs, "persistence", "RMSE")
    out += [
        "\\begin{table}[t]", "\\centering",
        "\\caption{The six encoders on one harness (corrected data, 3 origins $\\times$ 3 seeds,"
        " squared error, no seasonal features). Mean validation / test RMSE. The pure SEIR"
        " decoder (\\texttt{foi}) is worse than persistence on every encoder; the gated SEIR head"
        " (\\texttt{foi\\_res}) brings all six to within 0.25 of each other, below persistence.}",
        "\\label{tab:encoders}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lccc}", "\\toprule",
        "encoder & direct & SEIR decoder & gated SEIR \\\\", "\\midrule",
    ]
    for bb in ENCODERS:
        cells = [f"{mean(rs, f'{bb}+{h}', 'val_RMSE'):.2f} / {mean(rs, f'{bb}+{h}', 'RMSE'):.2f}"
                 for h in ("direct", "foi", "foi_res")]
        out.append(f"{bb} & " + " & ".join(cells) + " \\\\")
    out += ["\\midrule", f"persistence & \\multicolumn{{3}}{{c}}{{{pv:.2f} / {pt:.2f}}} \\\\",
            "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]


def table_levers(out: list[str]) -> list[dict]:
    found = []
    for label, grid, arm, ref in LEVERS:
        if grid is None:
            found.append({"label": label, "section": True})
            continue
        rs = rows(grid)
        d, w, n, p = pair(rs, arm, ref, "val_RMSE", "origin_seed")
        dt = pair(rs, arm, ref, "RMSE", "origin_seed")[0]
        found.append({"label": label, "grid": grid, "d": d, "w": w, "n": n, "p": p, "dt": dt})
    real = [f for f in found if not f.get("section")]
    for f, a in zip(real, stats.benjamini_hochberg([f["p"] for f in real])):
        f["p_adj"] = a
    out += [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Every lever, paired against its own control on matched (origin, seed) runs."
        " $\\Delta$ is validation RMSE (negative is better); wins count the 9 units where the"
        " arm is better; $p_{\\text{adj}}$ is BH across this table. Test $\\Delta$ is shown"
        " and decided nothing. With three origins these $p$-values describe run-to-run"
        " stability, not the series (\\S\\ref{sec:protocol}).}",
        "\\label{tab:levers}", "\\small", "\\setlength{\\tabcolsep}{1.8pt}",
        "\\begin{tabular}{lrcrr}", "\\toprule",
        "lever & $\\Delta$ val & wins & $p_{\\text{adj}}$ & $\\Delta$ test \\\\", "\\midrule",
    ]
    for f in found:
        if f.get("section"):
            out.append(f"\\multicolumn{{5}}{{l}}{{{f['label']}}} \\\\")
            continue
        out.append(f"\\quad {f['label']} & {f['d']:+.2f} & {f['w']}/{f['n']} & "
                   f"{fmt_p(f['p_adj'])} & {f['dt']:+.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return real


def table_nine(out: list[str]) -> None:
    c, p = rows("confirm"), rows("physics9")
    comps = [
        ("EXP-038", c, "ASTGCN+foi_res", "LSTM+foi_res", "gated SEIR-GNN (ASTGCN) vs SEIR-LSTM"),
        ("EXP-047", p, "P3 gated SEIR-GNN", "P6 SEIR-LSTM", "gated SEIR-GNN (AAGCN) vs SEIR-LSTM"),
        ("EXP-047", p, "P4 metapopulation SEIR-GNN", "P6 SEIR-LSTM", "metapopulation SEIR-GNN vs SEIR-LSTM"),
        ("EXP-038", c, "AAGCN+direct", "LSTM+direct", "AAGCN vs LSTM, direct head"),
        ("EXP-038", c, "AAGCN+direct", "persistence", "$B$-type direct model vs persistence"),
        ("EXP-047", p, "P0 B", "persistence", "$B$ vs persistence"),
        ("EXP-047", p, "P3 gated SEIR-GNN", "persistence", "gated SEIR-GNN vs persistence"),
    ]
    out += [
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Nine disjoint origins, paired at the origin unit (the unit a claim about the"
        " series rests on; attainable $p$ floor 0.004). Raw two-sided $p$. The two runs share"
        " the same test spans and are not independent replications.}",
        "\\label{tab:nine}", "\\small", "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{llrcrrcr}", "\\toprule",
        " & & \\multicolumn{3}{c}{validation} & \\multicolumn{3}{c}{test} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
        "comparison & run & $\\Delta$ & wins & $p$ & $\\Delta$ & wins & $p$ \\\\", "\\midrule",
    ]
    for run, rs, a, b, label in comps:
        v, t = pair(rs, a, b, "val_RMSE", "origin"), pair(rs, a, b, "RMSE", "origin")
        out.append(f"{label} & {run} & {v[0]:+.2f} & {v[1]}/{v[2]} & {fmt_p(v[3])}"
                   f" & {t[0]:+.2f} & {t[1]}/{t[2]} & {fmt_p(t[3])} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]


def table_rescue(out: list[str]) -> bool:
    if not (RESULTS / "rescue.json").exists():
        out += ["% tab:rescue -- rescue.json not yet available (EXP-048 running on Kaggle).", ""]
        return False
    rs = rows("rescue")
    heads = ("direct", "residual", "gated", "foi_res")
    out += [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Anchor, gate or physics? Each encoder behind four heads that add, in turn, the"
        " persistence anchor, the learned gate and the SEIR simulator (mean validation RMSE;"
        " configuration of Table~\\ref{tab:encoders}, all rerun together on Kaggle). The last two"
        " columns pair the gated SEIR head against its no-physics twin (\\texttt{gated});"
        " negative favours the physics, and the count is units where it wins.}",
        "\\label{tab:rescue}", "\\small", "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        " & & & & gated & \\multicolumn{2}{c}{SEIR $-$ gated} \\\\",
        "\\cmidrule(lr){6-7}",
        "encoder & direct & resid. & gated & SEIR & val & test \\\\", "\\midrule",
    ]
    for bb in ENCODERS:
        vals = " & ".join(f"{mean(rs, f'{bb}+{h}', 'val_RMSE'):.2f}" for h in heads)
        d, w, n, _ = pair(rs, f"{bb}+foi_res", f"{bb}+gated", "val_RMSE", "origin_seed")
        dt, wt, nt, _ = pair(rs, f"{bb}+foi_res", f"{bb}+gated", "RMSE", "origin_seed")
        out.append(f"{bb} & {vals} & {d:+.2f} ({w}/{n}) & {dt:+.2f} ({wt}/{nt}) \\\\")
    out += ["\\midrule",
            f"persistence & \\multicolumn{{4}}{{c}}{{{mean(rs, 'persistence', 'val_RMSE'):.2f}}} & & \\\\",
            "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return True


# -------------------------------------------------------------------- figures --
def fig_encoders() -> None:
    rs = rows("real")
    heads = (("direct", "direct", "#9aa5b1"), ("foi", "SEIR decoder", "#e0a458"),
             ("foi_res", "gated SEIR", "#3d7ea6"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=False)
    x = np.arange(len(ENCODERS))
    for ax, metric, title in ((axes[0], "val_RMSE", "validation"), (axes[1], "RMSE", "test")):
        for k, (h, lab, col) in enumerate(heads):
            ax.bar(x + (k - 1) * 0.27, [mean(rs, f"{bb}+{h}", metric) for bb in ENCODERS],
                   0.27, label=lab, color=col)
        ax.axhline(mean(rs, "persistence", metric), color="k", ls="--", lw=1, label="persistence")
        ax.set_xticks(x, ENCODERS, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("RMSE", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, "fig_encoders")


def fig_levers(levers: list[dict]) -> None:
    lv = [f for f in levers if "vs persistence" not in f["label"]]
    fig, ax = plt.subplots(figsize=(3.4, 4.4))
    y = np.arange(len(lv))[::-1]
    cols = ["#2a9d8f" if f["d"] < 0 and f["w"] >= 7 else "#9aa5b1" if f["d"] < 0 else "#e76f51"
            for f in lv]
    ax.barh(y, [f["d"] for f in lv], color=cols)
    ax.axvline(0, color="k", lw=0.8)
    labels = [f["label"].replace("$", "").replace("\\", "") for f in lv]
    ax.set_yticks(y, labels, fontsize=6.5)
    ax.set_xlabel("$\\Delta$ validation RMSE vs own control", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(-1.0, 1.75)
    for yy, f in zip(y, lv):
        ax.text(1.72, yy, f"{f['w']}/{f['n']}", va="center", ha="right", fontsize=6)
    fig.tight_layout()
    _save(fig, "fig_levers")


def fig_nine() -> None:
    p = rows("physics9")
    arms = (("P0 B", "$B$ (direct)", "#9aa5b1"), ("P3 gated SEIR-GNN", "gated SEIR-GNN", "#3d7ea6"),
            ("P6 SEIR-LSTM", "SEIR-LSTM", "#e0a458"), ("persistence", "persistence", "k"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    origins = sorted({r["origin"] for r in p})
    for ax, metric, title in ((axes[0], "val_RMSE", "validation"), (axes[1], "RMSE", "test")):
        for name, lab, col in arms:
            ys = [np.mean([r[metric] for r in p if r["name"] == name and r["origin"] == o])
                  for o in origins]
            ax.plot(range(len(origins)), ys, marker="o", ms=3, lw=1, color=col, label=lab,
                    ls="--" if name == "persistence" else "-")
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
    _save(fig, "fig_nine")


def _save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    out = ["% Generated by scripts/build_paper_assets.py from seirgnn2/results -- do not edit by hand.", ""]
    table_encoders(out)
    levers = table_levers(out)
    table_nine(out)
    have_rescue = table_rescue(out)
    (PAPER / "tables_generated.tex").write_text("\n".join(out), encoding="utf-8")
    fig_encoders()
    fig_levers(levers)
    fig_nine()
    PKG.mkdir(parents=True, exist_ok=True)
    for g in GRIDS:
        if (RESULTS / f"{g}.json").exists():
            shutil.copy(RESULTS / f"{g}.json", PKG / f"{g}.json")
    for f in levers:
        print(f"{f['label'][:44]:44s} {f['d']:+.3f} {f['w']}/{f['n']} p={f['p']:.4f} "
              f"p_adj={f['p_adj']:.4f} test {f['dt']:+.2f}")
    print("rescue table:", "built" if have_rescue else "pending")


if __name__ == "__main__":
    main()
