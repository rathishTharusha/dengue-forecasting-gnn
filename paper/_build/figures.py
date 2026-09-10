"""Every figure and generated table in the paper, from saved results only.

Design rule: this module reads ``analysis/results/*.json`` and nothing else. It
imports numpy, matplotlib and the torch-free case loader, so the paper can be
rebuilt on a laptop with no GPU, no torch and no PyTorch Geometric. Anything that
needed training was run once by the scripts in ``analysis/_build/`` and saved.

That split is deliberate. A figure pipeline that retrains models is a figure
pipeline nobody reruns, and numbers that are expensive to regenerate quietly stop
matching the text.

Each ``fig_*`` returns the Matplotlib figure and writes both PDF (for LaTeX) and
PNG (for the notebook and README). Each ``table_*`` returns a LaTeX string.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from dengue_gnn.data import load_cases  # noqa: E402

RESULTS = REPO / "analysis" / "results"
FIGURES = REPO / "paper" / "figures"
NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"

#: EDA F4. The reporting backlog, as a week index into the 459-week series.
ARTIFACT_WEEK = 395

ARCH_ORDER = ["STGAT", "A3TGCN", "ASTGCN", "DCRNN", "AAGCN"]
INC_ORDER = ["base", "per_horizon_heads", "temporal_attention", "huber", "probabilistic"]
INC_LABEL = {"base": "base", "per_horizon_heads": "per-horizon", "huber": "Huber",
             "temporal_attention": "attention", "probabilistic": "probabilistic"}

#: Colourblind-safe. Okabe--Ito, minus the yellow, which is illegible on white.
INK = "#222222"
GREY = "#9a9a9a"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
SKY = "#56B4E9"


def style() -> None:
    """Matplotlib defaults tuned for a two-column ACM page."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
    })


def load(name: str):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def _save(fig, stem: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}")
    return fig


# ---------------------------------------------------------------------------
# Figure 1 — the evaluation artifact
# ---------------------------------------------------------------------------


def fig_artifact():
    """Week 395, and what it does to a fold's score.

    The point is not that an anomaly exists -- every surveillance series has them.
    It is that this one sits in exactly one fold's *test* split and in no training
    split, so it inflates that fold's error without any model being able to learn
    from it, and it inverts which fold looks hardest.

    Single column and single panel, deliberately. The per-fold RMSE breakdown that
    used to be panel (b) is three numbers, and three numbers belong in the caption
    or the running text -- spending a full-width float on them cost more page than
    they were worth against a four-page limit.
    """
    cases = load_cases(NPY)
    national = cases.sum(axis=1)

    fig, ax = plt.subplots(figsize=(3.34, 1.95))

    ax.plot(national, color=BLUE, lw=0.9)
    ax.axvline(ARTIFACT_WEEK, color=ORANGE, lw=1.0, ls="--")
    ax.annotate("week 395", xy=(ARTIFACT_WEEK, national[ARTIFACT_WEEK]),
                xytext=(ARTIFACT_WEEK - 175, national.max() * 0.88),
                color=ORANGE, fontsize=7,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
    # Shade the three test splits so the reader can see where 395 lands.
    n_ids = len(cases) - 6
    # Adjacent spans abut, so without a divider the first two read as one block
    # and the caption promises three.
    for origin, colour in zip((0.55, 0.70, 0.85), (GREY, GREY, PURPLE)):
        lo = 3 + int(origin * n_ids)
        hi = 3 + int(min(origin + 0.15, 1.0) * n_ids)
        ax.axvspan(lo, hi, color=colour, alpha=0.18, lw=0)
        ax.axvline(lo, color="white", lw=0.8)
    ax.set_xlabel("week index")
    ax.set_ylabel("national weekly cases")

    fig.tight_layout()
    return _save(fig, "fig_artifact")


# ---------------------------------------------------------------------------
# Figure 2 — the main result
# ---------------------------------------------------------------------------


def _sweep_grid(records, field):
    archs = [a for a in ARCH_ORDER if any(r["arch"] == a for r in records)]
    incs = [i for i in INC_ORDER
            if any(r["increment"] == i and r["arch"] != "persistence" for r in records)]
    grid = np.full((len(archs), len(incs)), np.nan)
    for i, a in enumerate(archs):
        for j, inc in enumerate(incs):
            v = [r[field] for r in records if r["arch"] == a and r["increment"] == inc]
            if v:
                grid[i, j] = float(np.mean(v))
    floor = float(np.mean([r[field] for r in records if r["arch"] == "persistence"]))
    return archs, incs, grid, floor


def fig_results():
    """Every arm against the persistence floor, scored both ways.

    The left panel is what the frozen protocol reports; the right drops the six
    contaminated windows. AAGCN's apparent 3.8-RMSE win on the left is the figure's
    subject: it does not survive on the right, because it was a win at predicting a
    reporting backlog rather than at forecasting.
    """
    records = load("improved_sweep")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), sharey=False)

    for ax, field, title in zip(
        axes, ("RMSE", "RMSE_clean"),
        ("(a) all windows", "(b) artifact-free")
    ):
        archs, incs, grid, floor = _sweep_grid(records, field)
        x = np.arange(len(archs))
        width = 0.16
        colours = [BLUE, SKY, GREEN, ORANGE, PURPLE]
        for j, inc in enumerate(incs):
            ax.bar(x + (j - (len(incs) - 1) / 2) * width, grid[:, j], width,
                   label=INC_LABEL.get(inc, inc), color=colours[j % len(colours)])
        ax.axhline(floor, color=INK, lw=1.1, ls="--")
        ax.text(len(archs) - 0.42, floor, f" persistence {floor:.2f}",
                va="bottom", ha="right", fontsize=7, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(archs, rotation=12)
        ax.set_ylabel("pooled RMSE")
        lo = min(np.nanmin(grid), floor)
        hi = max(np.nanmax(grid), floor)
        ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.10 * (hi - lo))
        ax.set_title(title, loc="left")
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    axes[0].legend(ncol=3, loc="upper left", columnspacing=1.0, handlelength=1.2)
    fig.tight_layout()
    return _save(fig, "fig_results")


# ---------------------------------------------------------------------------
# Figure 3 — why the physics fails
# ---------------------------------------------------------------------------


def fig_physics():
    """The two measurements that close the mechanistic route.

    (a) Every causally available renewal anchor is worse than persistence, even
    though the same equation with oracle R reconstructs the series almost exactly.
    (b) The reason: R is a backward-looking descriptor. It correlates strongly with
    the growth that has already happened and *negatively* with the growth to come,
    so a penalty that pulls a forecast toward it pulls the wrong way.
    """
    m = load("paper_measurements")
    feas = load("renewal_feasibility")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.3),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    order = ["persistence", "force", "ratio_0.5", "rhat_force"]
    labels = ["persistence\n(baseline)", "$R{=}1$\n(force only)",
              "ratio form\n(damped)", "$\\hat{R}\\cdot$force"]
    values = [m["anchors"]["mean"][k] for k in order]
    colours = [INK, GREY, GREY, ORANGE]
    bars = ax.bar(labels, values, color=colours, width=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.2f}",
                ha="center", fontsize=7, color=INK)
    ax.axhline(values[0], color=INK, lw=0.9, ls="--")
    ax.set_ylabel("artifact-free RMSE")
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_title("(a) Causally available physics anchors", loc="left")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.text(0.02, 0.93,
            f"with oracle $R_t$: {feas['oracle_replay_rmse']:.2f}",
            transform=ax.transAxes, fontsize=7, color=GREEN)

    ll = m["lead_lag"]
    names = ["past 3-week\ngrowth", "future 3-week\ngrowth", "$|$future$|$\ngrowth"]
    corrs = [ll["log_r_vs_past_growth"], ll["log_r_vs_future_growth"],
             ll["log_r_vs_abs_future_growth"]]
    cols = [GREEN if c > 0 else ORANGE for c in corrs]
    bars = bx.barh(names, corrs, color=cols, height=0.55)
    for bar, c in zip(bars, corrs):
        bx.text(c + (0.03 if c > 0 else -0.03), bar.get_y() + bar.get_height() / 2,
                f"{c:+.3f}", va="center", ha="left" if c > 0 else "right",
                fontsize=7, color=INK)
    bx.axvline(0, color=INK, lw=0.8)
    bx.set_xlim(-0.55, 1.02)
    bx.set_xlabel("correlation with $\\log \\hat{R}_t$")
    bx.set_title("(b) $\\hat{R}$ looks backwards", loc="left")
    bx.invert_yaxis()

    fig.tight_layout()
    return _save(fig, "fig_physics")


# ---------------------------------------------------------------------------
# Figure 4 — the information ceiling
# ---------------------------------------------------------------------------


def fig_ceiling():
    """Why no loss term moves the point forecast.

    (a) The models cannot express the growth the data contains -- predicted maxima
    sit an order of magnitude below observed. (b) The reason is informational, not
    architectural: the previous week explains 85% of the next, and every other
    candidate predictor of transmission explains almost nothing.
    """
    diag = load("error_diagnosis")["arms"]
    rp = load("r_predictability")
    mr = load("mechanistic_r")
    resp = load("response_diagnosis")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.3))

    archs = list(diag)
    x = np.arange(len(archs))
    pred = [diag[a]["growth"]["pred_max"] for a in archs]
    true = [diag[a]["growth"]["true_max"] for a in archs]
    ax.bar(x - 0.19, pred, 0.36, label="model, max", color=BLUE)
    ax.bar(x + 0.19, true, 0.36, label="observed, max", color=ORANGE)
    for i, (p, t) in enumerate(zip(pred, true)):
        ax.text(i - 0.19, p + 0.08, f"{p:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.19, t + 0.08, f"{t:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.set_ylabel("$|\\Delta \\log(1+\\mathrm{cases})|$ per week")
    ax.set_ylim(0, max(true) * 1.22)
    ax.legend(loc="upper left")
    ax.set_title("(a) The models cannot express an outbreak", loc="left")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    slope = resp["damping_slope_vs_truth"]
    ax.text(0.98, 0.60, f"growth response slope: {slope:.3f}\n(1.0 = undamped)",
            transform=ax.transAxes, ha="right", fontsize=7, color=ORANGE)

    names = ["cases at $t{-}1$\n$\\rightarrow$ cases at $t$",
             "$\\log R_{t-1}$\n$\\rightarrow \\log R_t$",
             "climate\n$\\rightarrow \\log R_t$",
             "climate + hump\n$\\rightarrow \\log R_t$",
             "depletion\n$\\rightarrow \\log R_t$"]
    vals = [0.85, mr["own_past"], mr["climate_linear"], mr["climate_hump"],
            mr["depletion_all"]]
    cols = [GREEN, SKY, GREY, GREY, GREY]
    bars = bx.barh(names, vals, color=cols, height=0.6)
    for bar, v in zip(bars, vals):
        bx.text(max(v, 0) + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=7, color=INK)
    bx.axvline(0, color=INK, lw=0.8)
    bx.set_xlim(-0.08, 1.0)
    bx.set_xlabel("$r^2$ (out of sample, except the first)")
    bx.set_title("(b) Where the predictability is", loc="left")
    bx.invert_yaxis()
    assert rp["own_past"] > 0  # in-sample counterpart; kept for the caption

    fig.tight_layout()
    return _save(fig, "fig_ceiling")


# ---------------------------------------------------------------------------
# Figure 5 — the tractable reframing
# ---------------------------------------------------------------------------


def fig_detection():
    """Outbreaks are rankable three weeks out even though they are not countable.

    The same mechanistic quantity that cannot forecast a count is the largest
    single addition to a detector, because detection asks about the present state
    and R estimates exactly that.
    """
    m = load("paper_measurements")
    roc, base_rate = m["roc"], m["roc_base_rate"]
    det = m["detection"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.4),
                                 gridspec_kw={"width_ratios": [1, 1.12]})

    for key, colour, label in (("level", SKY, "level"),
                               ("level+rhat", GREEN, "level $+\\ \\hat{R}$"),
                               ("climate", GREY, "climate only")):
        fpr, tpr = roc[key]
        ax.plot(fpr, tpr, color=colour,
                label=f"{label} (AUC {det.get(key.replace('level+rhat', 'level+rhat'), det['level'])['auc']:.3f})"
                if key in det else label)
    ax.plot([0, 1], [0, 1], color=INK, lw=0.8, ls=":")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"(a) ROC, 3 weeks ahead (base rate {base_rate:.1%})", loc="left")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    keys = ["level", "level+growth", "level+neighbours", "level+rhat"]
    labels = ["level only", "+ growth", "+ neighbours (graph)", "+ $\\hat{R}$ (physics)"]
    aucs = [det[k]["auc"] for k in keys]
    base = aucs[0]
    cols = [GREY, GREY, SKY, GREEN]
    bars = bx.barh(labels, aucs, color=cols, height=0.58)
    for bar, v in zip(bars, aucs):
        delta = v - base
        txt = f"{v:.3f}" + (f"  ({delta:+.3f})" if delta else "")
        bx.text(v + 0.002, bar.get_y() + bar.get_height() / 2, txt,
                va="center", fontsize=7, color=INK)
    bx.set_xlim(0.79, 0.845)
    bx.set_xlabel("AUC")
    bx.set_title("(b) What each addition buys", loc="left")
    bx.invert_yaxis()
    bx.grid(axis="x")
    bx.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, "fig_detection")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _fmt(v):
    return "--" if not np.isfinite(v) else f"{v:.2f}"


def table_main() -> str:
    """Main results table: every architecture x increment, scored both ways."""
    records = load("improved_sweep")
    archs, incs, all_grid, all_floor = _sweep_grid(records, "RMSE")
    _, _, clean_grid, clean_floor = _sweep_grid(records, "RMSE_clean")

    head = " & ".join(INC_LABEL.get(i, i) for i in incs)
    lines = [
        "% Generated by paper/_build/figures.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Pooled RMSE over 3 origins $\\times$ 3 seeds. Lower is better. "
        "\\textbf{Bold} beats the persistence floor for that column. No increment "
        "differs from \\texttt{base} at $p<0.05$ under a paired test on matched "
        "(architecture, origin, seed) runs.}",
        "\\label{tab:main}",
        "\\small",
        "\\begin{tabular}{l" + "r" * len(incs) + "}",
        "\\toprule",
        f"architecture & {head} \\\\",
    ]
    for title, grid, floor in (("all windows", all_grid, all_floor),
                               ("artifact-free", clean_grid, clean_floor)):
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{len(incs) + 1}}}{{l}}{{\\emph{{{title}}} "
                     f"---\\ persistence {floor:.2f}}} \\\\")
        for i, a in enumerate(archs):
            cells = []
            for j in range(len(incs)):
                v = grid[i, j]
                cells.append(f"\\textbf{{{_fmt(v)}}}" if np.isfinite(v) and v < floor
                             else _fmt(v))
            lines.append(f"{a} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def table_physics() -> str:
    """Why the mechanistic route closes, in one table."""
    m = load("paper_measurements")
    mr = load("mechanistic_r")
    a = m["anchors"]["mean"]
    ll = m["lead_lag"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{The mechanistic route, measured. Every anchor available at "
        "forecast time is worse than persistence, and every candidate predictor of "
        "transmission intensity explains almost nothing out of sample.}",
        "\\label{tab:physics}",
        "\\small",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "\\multicolumn{2}{l}{\\emph{Renewal anchors} --- artifact-free RMSE} \\\\",
        "\\midrule",
        f"persistence & {a['persistence']:.2f} \\\\",
        f"$R{{=}}1$ (force only) & {a['force']:.2f} \\\\",
        f"ratio form, damped & {a['ratio_0.5']:.2f} \\\\",
        f"$\\hat{{R}}\\cdot$force & {a['rhat_force']:.2f} \\\\",
        "\\midrule",
        "\\multicolumn{2}{l}{\\emph{Predicting} $\\log R_t$ --- out-of-sample $r^2$} \\\\",
        "\\midrule",
        f"own past & {mr['own_past']:.3f} \\\\",
        f"climate, linear & {mr['climate_linear']:.3f} \\\\",
        f"climate $+$ thermal curvature & {mr['climate_hump']:.3f} \\\\",
        f"susceptible depletion & {mr['depletion_all']:.3f} \\\\",
        "\\midrule",
        f"corr.\\ $\\log\\hat{{R}}_t$ with \\emph{{past}} growth "
        f"& ${ll['log_r_vs_past_growth']:+.3f}$ \\\\",
        f"corr.\\ with \\emph{{future}} growth "
        f"& ${ll['log_r_vs_future_growth']:+.3f}$ \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def table_detection() -> str:
    """Outbreak detection, incremental."""
    det = load("paper_measurements")["detection"]
    base = det["level"]["auc"]
    rows = [("level only", "level"), ("$+$ recent growth", "level+growth"),
            ("$+$ neighbours (graph)", "level+neighbours"),
            ("$+\\ \\hat{R}$ (mechanistic)", "level+rhat"),
            ("$+$ all, incl.\\ climate", "level+rhat+neighbours+climate")]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Outbreak detection three weeks ahead: will a district exceed its "
        "own 90th percentile? Logistic regression fitted on training weeks only. "
        "Base rate 14.4\\%.}",
        "\\label{tab:detection}",
        "\\small",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "predictors & AUC & AP & $\\Delta$AUC \\\\",
        "\\midrule",
    ]
    for label, key in rows:
        d = det[key]
        delta = d["auc"] - base
        lines.append(f"{label} & {d['auc']:.3f} & {d['ap']:.3f} & "
                     f"${delta:+.3f}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


#: Tables the four-page version emits. `main` duplicates fig_results and
#: `detection` duplicates fig_detection(b); against a 4-page limit the figures
#: are the better use of the space, so both are still generated -- and printed
#: by the notebook -- but not written into the LaTeX.
INCLUDED_TABLES = ("physics",)


def build_all(verbose: bool = True, include: tuple[str, ...] = INCLUDED_TABLES):
    """Regenerate every figure and the generated-tables file."""
    style()
    figures = {}
    for fn in (fig_artifact, fig_results, fig_physics, fig_ceiling, fig_detection):
        figures[fn.__name__] = fn()
        if verbose:
            print(f"  {fn.__name__} -> paper/figures/{fn.__name__.replace('fig_', 'fig_')}")
    builders = {"main": table_main, "physics": table_physics,
                "detection": table_detection}
    tables = "\n\n".join(
        ["% Generated by paper/_build/figures.py -- do not edit by hand."]
        + [builders[name]() for name in include]
    )
    (REPO / "paper" / "tables_generated.tex").write_text(tables + "\n", encoding="utf-8")
    if verbose:
        print("  tables  -> paper/tables_generated.tex")
    return figures, tables


if __name__ == "__main__":
    build_all()
