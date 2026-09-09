"""Generate the Phase-2 ablation tables from the committed run CSV.

This is the script `docs/PHASE2_REVIEW.md` names as the fix for findings F2, F3,
F8 and F9: every number that reaches the paper is computed here from
`results/phase2_runs.csv`, so hand-transcription -- which is how the Phase-2
headline claim came to invert against the team's own data -- is no longer part of
the path from experiment to table.

Nothing is rounded until printing, and nothing is aggregated in the CSV, so the
choice of aggregation is visible in this file rather than baked into the data.

Two aggregation conventions coexisted in Phase 2 and disagreed by 0.81 RMSE --
more than the entire claimed improvement (review F9). They are both emitted here,
labelled:

    pooled          RMSE over all (window, node, horizon) elements at once.
                    Rows with `horizon == 0`. This is the default; it is what a
                    single headline RMSE should mean.
    mean-of-horizon Mean of the three per-horizon RMSEs. What the Phase-1
                    baseline CSV stores, so it is the one to use when quoting a
                    number alongside `results/baseline_rolling_origin.csv`.

Usage:
    python scripts/make_tables.py                       # markdown to stdout
    python scripts/make_tables.py --latex               # LaTeX tabular bodies
    python scripts/make_tables.py --runs path/to.csv
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEFAULT_RUNS = REPO / "results" / "phase2_runs.csv"

#: Display order and display names. Anything in the CSV but not here is appended
#: at the end, so a new configuration shows up rather than being silently dropped.
ORDER = [
    ("persistence", "Persistence floor"),
    ("dense_fixed", "Fixed geographic graph (control)"),
    ("adaptive", "+ adaptive graph"),
    ("adaptive_lam0.01", r"+ spatial reg. $\lambda=0.01$"),
    ("adaptive_lam0.1", r"+ spatial reg. $\lambda=0.1$"),
    ("adaptive_lam1.0", r"+ spatial reg. $\lambda=1.0$"),
]


def read_runs(path: Path) -> list[dict]:
    """Load the run CSV, coercing the numeric columns.

    Raises:
        SystemExit: If the file is missing -- with the command that produces it,
            because a missing results file is a workflow problem, not a bug.
    """
    import csv

    if not path.exists():
        raise SystemExit(
            f"no run data at {os.path.relpath(path, REPO)}\n"
            f"produce it first:  python scripts/run_phase2.py"
        )

    numeric = {
        "fold",
        "seed",
        "horizon",
        "rmse",
        "mae",
        "smape",
        "mape_masked",
        "peak_week_err",
        "gate_sigma",
        "lambda_phys",
        "n_obs",
        "n_test_weeks",
    }
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for k, v in raw.items():
                if k in numeric:
                    row[k] = float(v) if v not in ("", None) else math.nan
                else:
                    row[k] = v
            rows.append(row)
    return rows


def labels_in_order(rows: list[dict]) -> list[tuple[str, str]]:
    """Known labels first in ablation order, then anything unrecognised."""
    present = {r["label"] for r in rows}
    known = [(lab, name) for lab, name in ORDER if lab in present]
    extra = sorted(present - {lab for lab, _ in ORDER})
    return known + [(lab, lab) for lab in extra]


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.fmean(xs) if xs else math.nan


def _std(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def per_run(rows: list[dict], label: str, how: str) -> dict[tuple[int, int], dict]:
    """One aggregate score per ``(fold, seed)`` for one configuration.

    Args:
        rows: All run records.
        label: Configuration to select.
        how: ``"pooled"`` (the ``horizon == 0`` record) or ``"mean-of-horizon"``
            (mean over the per-horizon records).

    Returns:
        Mapping ``(fold, seed) -> {"rmse", "mae", "smape", "gate"}``.
    """
    out: dict[tuple[int, int], dict] = {}
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in rows:
        if r["label"] == label:
            grouped[(int(r["fold"]), int(r["seed"]))].append(r)

    for key, rs in grouped.items():
        if how == "pooled":
            pooled = [r for r in rs if int(r["horizon"]) == 0]
            if not pooled:
                continue
            r = pooled[0]
            out[key] = {
                "rmse": r["rmse"],
                "mae": r["mae"],
                "smape": r["smape"],
                "gate": r["gate_sigma"],
            }
        else:
            byh = [r for r in rs if int(r["horizon"]) > 0]
            if not byh:
                continue
            out[key] = {
                "rmse": _mean([r["rmse"] for r in byh]),
                "mae": _mean([r["mae"] for r in byh]),
                "smape": _mean([r["smape"] for r in byh]),
                "gate": _mean([r["gate_sigma"] for r in byh]),
            }
    return out


def peak_by_horizon(rows: list[dict], label: str) -> dict[int, float]:
    """Mean peak-week error per horizon. Not defined on the pooled record."""
    acc: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r["label"] == label and int(r["horizon"]) > 0:
            acc[int(r["horizon"])].append(r["peak_week_err"])
    return {h: _mean(v) for h, v in sorted(acc.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--latex", action="store_true", help="emit LaTeX tabular bodies")
    ap.add_argument(
        "--how",
        choices=("pooled", "mean-of-horizon"),
        default="pooled",
        help="aggregation across horizons for the headline number (review F9)",
    )
    args = ap.parse_args()

    rows = read_runs(args.runs)
    order = labels_in_order(rows)
    folds = sorted({int(r["fold"]) for r in rows})

    scores = {lab: per_run(rows, lab, args.how) for lab, _ in order}
    ref = scores.get("persistence", {})
    ref_by_fold = {f: _mean([v["rmse"] for (fo, _), v in ref.items() if fo == f]) for f in folds}

    sep = " & " if args.latex else " | "
    end = r" \\" if args.latex else " |"
    start = "" if args.latex else "| "

    print(f"# Phase-2 ablation  ({args.how} RMSE, {os.path.relpath(args.runs, REPO)})\n")
    head = ["Configuration", "RMSE", "MAE", "SMAPE %", "gate sigma", "n runs", "folds won"]
    print(start + sep.join(head) + end)
    if not args.latex:
        print("|" + "|".join(["---"] * len(head)) + "|")

    for lab, name in order:
        s = scores[lab]
        if not s:
            continue
        rmse = [v["rmse"] for v in s.values()]
        won = sum(
            1
            for f in folds
            if not math.isnan(ref_by_fold.get(f, math.nan))
            and _mean([v["rmse"] for (fo, _), v in s.items() if fo == f]) < ref_by_fold[f]
        )
        gate = _mean([v["gate"] for v in s.values()])
        cells = [
            name,
            f"{_mean(rmse):.2f} +/- {_std(rmse):.2f}",
            f"{_mean([v['mae'] for v in s.values()]):.2f}",
            f"{_mean([v['smape'] for v in s.values()]):.1f}",
            "--" if math.isnan(gate) else f"{gate:.3f}",
            str(len(s)),
            "--" if lab == "persistence" else f"{won}/{len(folds)}",
        ]
        print(start + sep.join(cells) + end)

    print(f"\n## Per-fold {args.how} RMSE (mean over seeds)\n")
    head = ["Configuration"] + [f"Fold {f}" for f in folds]
    print(start + sep.join(head) + end)
    if not args.latex:
        print("|" + "|".join(["---"] * len(head)) + "|")
    for lab, name in order:
        s = scores[lab]
        if not s:
            continue
        cells = [name]
        for f in folds:
            xs = [v["rmse"] for (fo, _), v in s.items() if fo == f]
            cells.append(f"{_mean(xs):.2f}" if xs else "--")
        print(start + sep.join(cells) + end)

    horizons = sorted({int(r["horizon"]) for r in rows if int(r["horizon"]) > 0})
    print("\n## Peak-week error (weeks), by horizon\n")
    head = ["Configuration"] + [f"h={h}" for h in horizons]
    print(start + sep.join(head) + end)
    if not args.latex:
        print("|" + "|".join(["---"] * len(head)) + "|")
    for lab, name in order:
        pw = peak_by_horizon(rows, lab)
        if not pw:
            continue
        print(start + sep.join([name] + [f"{pw.get(h, math.nan):.2f}" for h in horizons]) + end)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
