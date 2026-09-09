"""Generate the paper's result tables from committed data.

Every number the paper prints comes out of a results CSV through this script.
Hand-entry is what let the published lambda=1.00 row drift 0.88 RMSE from its
source and let the persistence row be filled from the wrong file entirely
(review findings F2, F3). Removing the keyboard from the path removes the class
of error.

Usage:
    python scripts/make_tables.py                          # Phase-3 runs
    python scripts/make_tables.py --runs results/phase2_runs.csv --prefix p2

Writes <prefix>_table_main.md, _lambda.md, _horizon.md, _compute.md under
results/, and paper/tables_generated.tex for the default (Phase-3) run.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Display names. Labels absent from a run are skipped; labels present but not
#: listed are appended, so a new config never silently vanishes from a table.
NAMES = {
    # naive
    "persistence": "Persistence floor",
    "seasonal_naive": "Seasonal naive (52 wk)",
    # non-graph learners
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "lstm": "LSTM (no graph)",
    # graph models
    "F_adaptive": "Adaptive graph (no temporal)",
    "A_dense_fixed": "Dense GCN, fixed graph",
    "A_adaptive": "Adaptive graph",
    "A_mech0.3": "+ growth constraint",
    "A_mech0.3_curr": "+ growth constraint + curriculum",
    "F_adaptive_shrunk": "Adaptive graph + shrinkage",
    "F_mech0.3_shrunk": "+ growth constraint + shrinkage",
    "F_gtcn": "+ gated temporal conv.",
    "F_gtcn_shrunk": "+ gated TCN + shrinkage",
    # augmentation (all unshrunk, no temporal)
    "F_aug_jitter": "aug: jitter",
    "F_aug_window_warp": "aug: window warp",
    "F_aug_gan": "aug: GAN (WGAN-GP)",
    # phase-2 legacy labels
    "dense_fixed": "Dense GCN, fixed graph",
    "adaptive": "+ adaptive graph",
    "adaptive_lam0.01": r"+ spatial reg. ($\lambda$=0.01)",
    "adaptive_lam0.1": r"+ spatial reg. ($\lambda$=0.1)",
    "adaptive_lam1.0": r"+ spatial reg. ($\lambda$=1.0)",
    "adaptive_d4": "adaptive, $d$=4",
    "adaptive_d20": "adaptive, $d$=20",
    "adaptive_h32": "adaptive, hidden=32",
    "adaptive_h128": "adaptive, hidden=128",
}
ORDER = list(NAMES)


def load(paths: list[Path]) -> list[dict]:
    """Read and concatenate result files.

    Multiple files are merged so the graph models and the non-graph comparators
    appear in one table -- they were produced by separate runs but under one
    protocol, and the paper needs them side by side. Schemas must agree; a
    mismatch means one file was produced by a different harness version and
    merging it would compare incomparable quantities.
    """
    rows: list[dict] = []
    fields: list[str] | None = None
    for path in paths:
        if not path.exists():
            raise SystemExit(f"no results at {path}; run scripts/run_phase3.py first")
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if fields is None:
                fields = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fields:
                raise SystemExit(
                    f"{path} has a different schema from {paths[0]}; refusing to merge"
                )
            rows.extend(reader)
    return rows


def labels_in(rows: list[dict]) -> list[str]:
    present = {r["label"] for r in rows}
    return [k for k in ORDER if k in present] + sorted(present - set(ORDER))


def name(label: str) -> str:
    return NAMES.get(label, label)


def pooled(rows: list[dict]) -> list[dict]:
    """horizon == 0 is the pooled-across-horizons record."""
    return [r for r in rows if r["horizon"] == "0"]


def by_label(rows: list[dict], field: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = r[field]
        if v not in ("", "nan"):
            out[r["label"]].append(float(v))
    return out


def mean_sd(values: list[float]) -> tuple[float, float]:
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def sign_test(rows: list[dict], a: str, b: str) -> tuple[int, int, float]:
    """Paired sign test on pooled RMSE over matched runs.

    Pairs on ``(fold, seed)``; falls back to matching on fold alone when the
    comparator is deterministic and has one record per fold, as persistence does.
    Returns ``(wins, n, p)`` two-sided.
    """
    A = {(r["fold"], r["seed"]): float(r["rmse"]) for r in rows if r["label"] == a}
    B = {(r["fold"], r["seed"]): float(r["rmse"]) for r in rows if r["label"] == b}
    by_fold = {f: v for (f, _), v in B.items()}

    wins = losses = 0
    for (fold, seed), va in A.items():
        vb = B.get((fold, seed), by_fold.get(fold))
        if vb is None:
            continue
        if va < vb:
            wins += 1
        elif va > vb:
            losses += 1

    n = wins + losses
    if n == 0:
        return 0, 0, 1.0
    k = min(wins, n - wins)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return wins, n, p


def md_main(rows: list[dict]) -> str:
    p = pooled(rows)
    rmse, mae, smape = (by_label(p, f) for f in ("rmse", "mae", "smape"))
    lines = [
        "# Main results — rolling-origin CV, pooled across horizons",
        "",
        "Mean ± SD over matched runs. `vs floor` is a two-sided paired sign test",
        "against persistence on matched (fold, seed) runs.",
        "",
        "| Model | n | RMSE | MAE | SMAPE | vs floor |",
        "|---|---|---|---|---|---|",
    ]
    for key in labels_in(p):
        rm, rs = mean_sd(rmse[key])
        am, _ = mean_sd(mae[key])
        sm, _ = mean_sd(smape[key])
        if key == "persistence":
            vs = "—"
        else:
            w, n, pv = sign_test(p, key, "persistence")
            vs = f"{w}/{n}, p={pv:.3f}"
        lines.append(
            f"| {name(key)} | {len(rmse[key])} | {rm:.2f} ± {rs:.2f} "
            f"| {am:.2f} | {sm:.1f}% | {vs} |"
        )

    if "adaptive" in rmse and "dense_fixed" in rmse:
        w, n, pv = sign_test(p, "adaptive", "dense_fixed")
        lines += [
            "",
            "## The contribution, against its own control",
            "",
            f"Adaptive graph vs the matched fixed-graph control: **{w}/{n} paired wins, "
            f"p={pv:.3f}**. This is the comparison that isolates the learned adjacency —",
            "same propagation path, same head, same optimiser, same target handling.",
        ]
    return "\n".join(lines) + "\n"


def md_lambda(rows: list[dict]) -> str:
    p = pooled(rows)
    rmse, gate = by_label(p, "rmse"), by_label(p, "gate_sigma")
    sweep = [
        ("adaptive", "0.00"),
        ("adaptive_lam0.01", "0.01"),
        ("adaptive_lam0.1", "0.10"),
        ("adaptive_lam1.0", "1.00"),
    ]
    lines = [
        "# Spatial regularisation sweep",
        "",
        "λ=0 is the adaptive graph with no regulariser. Applied to log1p(counts);",
        "see docs/PHASE2_REVIEW.md F5 for why not raw counts.",
        "",
        "| λ | RMSE | SD | learned gate σ(g) |",
        "|---|---|---|---|",
    ]
    for key, lam in sweep:
        if key not in rmse:
            continue
        rm, rs = mean_sd(rmse[key])
        gm, _ = mean_sd(gate[key]) if gate.get(key) else (float("nan"), 0.0)
        lines.append(f"| {lam} | {rm:.2f} | ± {rs:.2f} | {gm:.3f} |")
    return "\n".join(lines) + "\n"


def md_horizon(rows: list[dict]) -> str:
    lines = [
        "# Per-horizon breakdown",
        "",
        "Peak week error is the mean absolute displacement of the predicted outbreak",
        "peak, over districts reaching PEAK_FLOOR cases. Persistence is shifted h weeks",
        "by construction, which is the sanity check that the metric works.",
        "",
        "| Model | h | RMSE | MAE | Peak err (wks) |",
        "|---|---|---|---|---|",
    ]
    for key in labels_in(rows):
        for h in ("1", "2", "3"):
            sel = [r for r in rows if r["label"] == key and r["horizon"] == h]
            if not sel:
                continue
            rm, _ = mean_sd([float(r["rmse"]) for r in sel])
            am, _ = mean_sd([float(r["mae"]) for r in sel])
            pk = [float(r["peak_week_err"]) for r in sel if r["peak_week_err"] != "nan"]
            lines.append(
                f"| {name(key)} | {h} | {rm:.2f} | {am:.2f} | {st.mean(pk):.2f} |"
                if pk
                else f"| {name(key)} | {h} | {rm:.2f} | {am:.2f} | — |"
            )
    return "\n".join(lines) + "\n"


def md_compute(rows: list[dict]) -> str:
    """Computational Analysis — required by the handout's paper structure."""
    p = pooled(rows)
    if "n_params" not in (p[0] if p else {}):
        return "# Computational analysis\n\nNot instrumented in this run.\n"

    lines = [
        "# Computational analysis",
        "",
        "Training wall-clock is per fold, single-threaded. Inference is per",
        "25-district forward pass. Measured during the run, not estimated.",
        "",
        "| Model | Params | Train (s/fold) | Inference (ms/window) |",
        "|---|---|---|---|",
    ]
    for key in labels_in(p):
        sel = [r for r in p if r["label"] == key]
        params = int(float(sel[0]["n_params"]))
        tr = st.mean([float(r["train_seconds"]) for r in sel])
        inf = st.mean([float(r["infer_ms_per_window"]) for r in sel])
        lines.append(f"| {name(key)} | {params:,} | {tr:.1f} | {inf:.3f} |")
    return "\n".join(lines) + "\n"


def tex_main(rows: list[dict]) -> str:
    p = pooled(rows)
    rmse, mae, smape = (by_label(p, f) for f in ("rmse", "mae", "smape"))
    out = [
        "% GENERATED by scripts/make_tables.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Rolling-origin test performance, pooled across horizons. "
        "Mean $\\pm$ SD over matched runs.}",
        "  \\label{tab:main}",
        "  \\footnotesize",
        "  \\begin{tabular}{@{}lccc@{}}",
        "    \\toprule",
        "    Model & RMSE & MAE & SMAPE \\\\",
        "    \\midrule",
    ]
    for key in labels_in(p):
        rm, rs = mean_sd(rmse[key])
        am, _ = mean_sd(mae[key])
        sm, _ = mean_sd(smape[key])
        out.append(f"    {name(key)} & ${rm:.2f} \\pm {rs:.2f}$ & {am:.2f} & {sm:.1f}\\% \\\\")
    out += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        default=[
            REPO / "results" / "phase3_runs.csv",
            REPO / "results" / "phase3_baselines.csv",
        ],
    )
    ap.add_argument("--prefix", default="")
    ap.add_argument("--no-tex", action="store_true", help="skip paper/tables_generated.tex")
    args = ap.parse_args()

    rows = load([p for p in args.runs if p.exists()])
    pre = f"{args.prefix}_" if args.prefix else ""
    outputs = {
        REPO / "results" / f"{pre}table_main.md": md_main(rows),
        REPO / "results" / f"{pre}table_lambda.md": md_lambda(rows),
        REPO / "results" / f"{pre}table_horizon.md": md_horizon(rows),
        REPO / "results" / f"{pre}table_compute.md": md_compute(rows),
    }
    if not args.no_tex:
        outputs[REPO / "paper" / "tables_generated.tex"] = tex_main(rows)

    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")

    print()
    print(md_main(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
