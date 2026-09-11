"""Consolidate every downloaded beat-the-floor run into one decision table.

Kaggle emits one JSON per (architecture, head, feature set). This merges them,
scores every arm against the persistence floor **clustered by origin**, applies a
multiple-comparison correction, and writes a Markdown table for the experiment
log.

Why clustering matters here
---------------------------
The existing sweep's ``n=9`` p-values treat (origin, seed) rows as independent.
They are not: within an origin, seed variance on this task is 0.02-0.05, while
between origins it is 22.07 / 37.74 / 29.35. Nine rows carry three observations,
and clustering turns A3TGCN's ``-0.467, p=0.10`` into ``p=0.436``. Everything
here averages seeds within an origin before testing, so ``n`` is the number of
independent folds.

Why the correction matters here
-------------------------------
This sweep evaluates on the order of a hundred arms. At alpha=0.05 that is
roughly five spurious "wins" expected by chance alone, so an uncorrected minimum
over the table is not a result. Benjamini-Hochberg controls the false discovery
rate across the family and is reported alongside the raw p-value; a row that
survives it is worth writing up, and a row that does not is worth repeating with
more origins before anyone believes it.

Run::

    python analysis/_build/merge_beat.py --dir analysis/results/beat_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = REPO / "analysis" / "results" / "beat_baseline"


def load_all(directory: Path) -> list[dict]:
    """Every ``beat_*.json`` in the directory, tagged with its source file."""
    records: list[dict] = []
    for path in sorted(directory.glob("beat_*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        stem = path.stem.lower()
        default_head = "gauss" if "_gauss" in stem else ("nb" if "_nb" in stem else "det")
        default_feats = "climate" if "_climate" in stem else ("causal" if "_causal" in stem else "cases")
        default_phys = "spatial" if "_spatial" in stem else "none"
        for r in rows:
            if not r.get("head"):
                r["head"] = default_head
            if not r.get("features"):
                r["features"] = default_feats
            if not r.get("physics") or r.get("physics") == "base":
                r["physics"] = default_phys
            r["source"] = path.stem
        records.extend(rows)
    return records


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return a boolean mask of discoveries at FDR ``alpha``."""
    n = len(pvalues)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresholds
    keep = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.max(np.flatnonzero(passed))
        keep[order[: cutoff + 1]] = True
    return keep


def summarise(records: list[dict], metric: str = "RMSE_clean") -> list[dict]:
    """One row per (arch, head, physics, features, arm), tested against the floor."""
    floor: dict[float, list[float]] = {}
    for r in records:
        if r["arch"] == "persistence":
            floor.setdefault(r["origin"], []).append(r[metric])
    floor_mean = {k: float(np.mean(v)) for k, v in floor.items()}
    if not floor_mean:
        raise SystemExit("no persistence rows found -- cannot score anything against the floor")

    groups: dict[tuple, dict[float, list[float]]] = {}
    for r in records:
        if r["arch"] == "persistence":
            continue
        key = (r["arch"], r["head"], r.get("physics", "none"), r["features"], r["arm"])
        groups.setdefault(key, {}).setdefault(r["origin"], []).append(r[metric])

    rows = []
    for (arch, head, phys, feats, arm), by_origin in groups.items():
        origins = sorted(o for o in by_origin if o in floor_mean)
        if len(origins) < 2:
            continue
        # Average seeds inside an origin first: one observation per independent fold.
        v = np.array([np.mean(by_origin[o]) for o in origins])
        f = np.array([floor_mean[o] for o in origins])
        d = v - f
        t, p = stats.ttest_rel(v, f)
        rows.append(dict(arch=arch, head=head, physics=phys, features=feats, arm=arm,
                         n_origins=len(v), n_runs=sum(len(x) for x in by_origin.values()),
                         RMSE=float(v.mean()), floor=float(f.mean()),
                         vs_floor=float(d.mean()), wins=int((d < 0).sum()),
                         t=float(t), p=float(p)))

    rows.sort(key=lambda r: r["vs_floor"])
    pv = np.array([r["p"] for r in rows])
    keep = benjamini_hochberg(pv)
    for r, k in zip(rows, keep, strict=True):
        r["bh_significant"] = bool(k)
    return rows


def to_markdown(rows: list[dict], limit: int = 30) -> str:
    """A table for the experiment log, best arms first."""
    head = ("| arch | head | phys | feat | arm | RMSE | vs floor | wins | p | BH |\n"
            "|---|---|---|---|---|---:|---:|---:|---:|:--:|\n")
    body = "".join(
        f"| {r['arch']} | {r['head']} | {r['physics']} | {r['features']} | `{r['arm']}` | "
        f"{r['RMSE']:.3f} | {r['vs_floor']:+.3f} | {r['wins']}/{r['n_origins']} | "
        f"{r['p']:.4f} | {'**yes**' if r['bh_significant'] else 'no'} |\n"
        for r in rows[:limit]
    )
    return head + body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--metric", default="RMSE_clean",
                    choices=["RMSE_clean", "RMSE", "MAE_clean", "MAE"])
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", default=None, help="Write the Markdown table here")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    directory = Path(args.dir)
    records = load_all(directory)
    if not records:
        raise SystemExit(f"no beat_*.json under {directory}")

    sources = sorted({r["source"] for r in records})
    print(f"{len(records)} records from {len(sources)} file(s):")
    for s in sources:
        print(f"  {s}")

    rows = summarise(records, args.metric)
    print(f"\n=== {args.metric}, clustered by origin, {len(rows)} arms ===")
    print(f"{'arch':20s}{'head':7s}{'phys':9s}{'feat':9s}{'arm':20s}"
          f"{'RMSE':>9s}{'vs floor':>10s}{'wins':>8s}{'p':>9s}  BH")
    print("-" * 105)
    for r in rows[: args.limit]:
        print(f"{r['arch']:20s}{r['head']:7s}{r['physics']:9s}{r['features']:9s}{r['arm']:20s}"
              f"{r['RMSE']:9.3f}{r['vs_floor']:+10.3f}"
              f"{r['wins']:5d}/{r['n_origins']:<3d}{r['p']:9.4f}"
              f"  {'YES' if r['bh_significant'] else '.'}")

    winners = [r for r in rows if r["vs_floor"] < 0 and r["bh_significant"]]
    print()
    if winners:
        b = winners[0]
        print(f"{len(winners)} arm(s) beat the floor and survive Benjamini-Hochberg.")
        print(f"Best: {b['arch']}/{b['head']}/{b['features']} {b['arm']} -> "
              f"{b['RMSE']:.3f} vs floor {b['floor']:.3f} "
              f"({b['vs_floor']:+.3f}, {b['wins']}/{b['n_origins']} origins, p={b['p']:.4f})")
    else:
        nominal = [r for r in rows if r["vs_floor"] < 0 and r["p"] < 0.05]
        print("No arm beats the floor after correcting for multiple comparisons.")
        if nominal:
            print(f"{len(nominal)} arm(s) are nominally significant but do not survive it; "
                  "treat them as candidates to re-run with more origins, not as results.")

    if args.out:
        Path(args.out).write_text(to_markdown(rows, args.limit), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
