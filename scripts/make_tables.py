"""Generate the paper's result tables from committed data.

Every number the paper prints comes out of results/phase2_runs.csv through this
script. Hand-entry is what let the published lambda=1.00 row drift 0.88 RMSE from
its source and let the persistence row be filled from the wrong file entirely
(review findings F2, F3). Removing the keyboard from the path removes the class
of error.

Writes:
    results/table_main.md          headline comparison, mean +/- std
    results/table_lambda.md        regularisation sweep
    results/table_horizon.md       per-horizon breakdown
    paper/tables_generated.tex     the same tables as LaTeX, for \\input

Usage:
    python scripts/make_tables.py
"""

from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

RUNS = REPO / "results" / "phase2_runs.csv"

#: Display names, in the order the ablation table builds them up.
ORDER = [
    ("persistence", "Persistence floor"),
    ("dense_fixed", "Dense GCN, fixed graph"),
    ("adaptive", "+ adaptive graph"),
    ("adaptive_lam0.01", "+ spatial reg. ($\\lambda$=0.01)"),
    ("adaptive_lam0.1", "+ spatial reg. ($\\lambda$=0.1)"),
    ("adaptive_lam1.0", "+ spatial reg. ($\\lambda$=1.0)"),
]


def load(path: Path = RUNS) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no results at {path}; run scripts/run_phase2.py first")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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


def per_fold_seed(rows: list[dict], field: str) -> dict[str, dict[tuple, float]]:
    """Keyed by (fold, seed) so runs can be paired across configurations."""
    out: dict[str, dict[tuple, float]] = defaultdict(dict)
    for r in rows:
        out[r["label"]][(r["fold"], r["seed"])] = float(r[field])
    return out


def mean_sd(values: list[float]) -> tuple[float, float]:
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def wilcoxon_vs(a: dict, b: dict) -> tuple[int, int, str]:
    """Sign test over paired (fold, seed) runs: how often does `a` beat `b`?

    Persistence is deterministic and has one record per fold, so it is broadcast
    across seeds. Reported as a win count rather than a p-value: with 9 paired
    runs the exact sign test cannot reach p<0.05 unless the result is unanimous,
    and saying so plainly is more honest than quoting a borderline statistic.
    """
    wins = losses = 0
    for key, av in a.items():
        bv = b.get(key)
        if bv is None:  # persistence: match on fold only
            bv = next((v for (f, _), v in b.items() if f == key[0]), None)
        if bv is None:
            continue
        if av < bv:
            wins += 1
        elif av > bv:
            losses += 1
    n = wins + losses
    if n == 0:
        verdict = "no paired runs"
    elif wins == n:
        verdict = f"wins all {n}"
    elif losses == n:
        verdict = f"loses all {n}"
    else:
        verdict = f"{wins}/{n} — not significant"
    return wins, n, verdict


def md_main(rows: list[dict]) -> str:
    p = pooled(rows)
    rmse, mae, smape = (by_label(p, f) for f in ("rmse", "mae", "smape"))
    paired = per_fold_seed(p, "rmse")
    persist = paired["persistence"]

    lines = [
        "# Main results — rolling-origin CV, pooled across horizons",
        "",
        "Mean ± SD over 3 origins × 3 seeds (persistence is deterministic: 3 folds).",
        "`vs floor` is a paired sign test against persistence on matched (fold, seed) runs.",
        "",
        "| Model | RMSE | MAE | SMAPE | vs floor |",
        "|---|---|---|---|---|",
    ]
    for key, name in ORDER:
        if key not in rmse:
            continue
        rm, rs = mean_sd(rmse[key])
        am, _ = mean_sd(mae[key])
        sm, _ = mean_sd(smape[key])
        vs = "—" if key == "persistence" else wilcoxon_vs(paired[key], persist)[2]
        lines.append(f"| {name} | {rm:.2f} ± {rs:.2f} | {am:.2f} | {sm:.1f}% | {vs} |")
    return "\n".join(lines) + "\n"


def md_lambda(rows: list[dict]) -> str:
    p = pooled(rows)
    rmse = by_label(p, "rmse")
    gate = by_label(p, "gate_sigma")
    lines = [
        "# Spatial regularisation sweep",
        "",
        "λ=0 is the adaptive graph with no regulariser. Applied to log1p(counts);",
        "see docs/PHASE2_REVIEW.md F5 for why not raw counts.",
        "",
        "| λ | RMSE | SD | learned gate σ(g) |",
        "|---|---|---|---|",
    ]
    for key, lam in [
        ("adaptive", "0.00"),
        ("adaptive_lam0.01", "0.01"),
        ("adaptive_lam0.1", "0.10"),
        ("adaptive_lam1.0", "1.00"),
    ]:
        if key not in rmse:
            continue
        rm, rs = mean_sd(rmse[key])
        gm, _ = mean_sd(gate[key])
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
    for key, name in ORDER:
        for h in ("1", "2", "3"):
            sel = [r for r in rows if r["label"] == key and r["horizon"] == h]
            if not sel:
                continue
            rm, _ = mean_sd([float(r["rmse"]) for r in sel])
            am, _ = mean_sd([float(r["mae"]) for r in sel])
            pk = [float(r["peak_week_err"]) for r in sel if r["peak_week_err"] != "nan"]
            pks = f"{st.mean(pk):.2f}" if pk else "—"
            lines.append(f"| {name} | {h} | {rm:.2f} | {am:.2f} | {pks} |")
    return "\n".join(lines) + "\n"


def tex_main(rows: list[dict]) -> str:
    p = pooled(rows)
    rmse, mae, smape = (by_label(p, f) for f in ("rmse", "mae", "smape"))
    out = [
        "% GENERATED by scripts/make_tables.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Rolling-origin test performance, 3 origins $\\times$ 3 seeds, "
        "pooled across horizons. Mean $\\pm$ SD.}",
        "  \\label{tab:main}",
        "  \\footnotesize",
        "  \\begin{tabular}{@{}lccc@{}}",
        "    \\toprule",
        "    Model & RMSE & MAE & SMAPE \\\\",
        "    \\midrule",
    ]
    for key, name in ORDER:
        if key not in rmse:
            continue
        rm, rs = mean_sd(rmse[key])
        am, _ = mean_sd(mae[key])
        sm, _ = mean_sd(smape[key])
        out.append(f"    {name} & ${rm:.2f} \\pm {rs:.2f}$ & {am:.2f} & {sm:.1f}\\% \\\\")
    out += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def main() -> int:
    # Windows consoles default to cp1252, which cannot encode the lambda in the
    # sweep table. Files are written UTF-8 regardless; this is for stdout only.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = load()
    outputs = {
        REPO / "results" / "table_main.md": md_main(rows),
        REPO / "results" / "table_lambda.md": md_lambda(rows),
        REPO / "results" / "table_horizon.md": md_horizon(rows),
        REPO / "paper" / "tables_generated.tex": tex_main(rows),
    }
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")

    print()
    print(md_main(rows))
    print(md_lambda(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
