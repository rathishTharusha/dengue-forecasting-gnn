"""Paired tests over the rows ``sweep.py`` writes.

``sweep.py`` saves one row per ``(config, origin, seed)`` and never aggregates,
which is what makes pairing possible: two arms that saw the same forecast origin
and the same initialisation seed differ only by the factor under test, so their
difference is a matched observation rather than two noisy means.

Two pairing units, both reported, because they answer different questions and
have very different power:

``origin``        seeds averaged first, one difference per forecast origin. The
                  origin is the only thing resampled from the *data*, so this is
                  the unit a claim about the series rests on. The frozen
                  protocol has three origins, and an exact sign-flip test on
                  three units cannot return a two-sided p below **0.25** --
                  no arm can be called significant at this unit until the
                  9-origin confirmatory grid is run.
``origin_seed``   one difference per (origin, seed). Nine units, so the floor
                  drops to 0.004, but the three seeds inside an origin share
                  their data and are replicates of *initialisation only*. A
                  p-value here is a statement about run-to-run stability, not
                  about the series. Reported second, and never on its own.

The test is the project's existing one (``run_s9_confirmatory.py``): exact
paired sign-flip permutation when the unit count allows, Benjamini-Hochberg FDR
across the family of arms compared in one call.

Usage::

    python seirgnn2/stats.py screen                       # every arm vs persistence
    python seirgnn2/stats.py screen --ref head=direct     # ... vs a named arm
    python seirgnn2/stats.py screen --metric RMSE         # test scores, not validation
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"

#: Below this many units the exact enumeration is used; above it, Monte Carlo.
EXACT_MAX = 16


def load(name: str) -> list[dict]:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"no such result file: {path}\nrun `python seirgnn2/sweep.py {name}` first")
    return json.loads(path.read_text(encoding="utf-8"))


def cells(rows: list[dict], metric: str) -> dict[str, dict[tuple[float, int], float]]:
    """``{arm: {(origin, seed): value}}``, averaging any accidental duplicates."""
    acc: dict[str, dict[tuple[float, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[r["name"]][(float(r["origin"]), int(r["seed"]))].append(float(r[metric]))
    return {arm: {k: float(np.mean(v)) for k, v in d.items()} for arm, d in acc.items()}


def paired(arm: dict[tuple[float, int], float], ref: dict[tuple[float, int], float],
           unit: str) -> np.ndarray:
    """Differences ``arm - ref``, one per unit.

    A reference with a single row per origin -- persistence, which is
    deterministic and is stored with ``seed = -1`` -- is broadcast across the
    arm's seeds rather than being dropped for failing to match on seed.
    """
    ref_by_origin = defaultdict(list)
    for (o, _), v in ref.items():
        ref_by_origin[o].append(v)
    ref_seeded = len({s for _, s in ref}) > 1 or all(s >= 0 for _, s in ref)

    diffs: dict[float, list[float]] = defaultdict(list)
    for (o, s), v in sorted(arm.items()):
        if ref_seeded and (o, s) in ref:
            base = ref[(o, s)]
        elif o in ref_by_origin:
            base = float(np.mean(ref_by_origin[o]))
        else:
            continue
        diffs[o].append(v - base)

    if unit == "origin":
        return np.array([np.mean(d) for _, d in sorted(diffs.items())])
    return np.array([v for _, d in sorted(diffs.items()) for v in d])


def sign_flip_p(diffs: np.ndarray, rng_seed: int = 42, num_perms: int = 100_000) -> float:
    """Two-sided paired sign-flip permutation p-value on the mean difference."""
    n = len(diffs)
    if n == 0:
        return float("nan")
    observed = abs(float(np.mean(diffs)))
    if n <= EXACT_MAX:
        flips = np.array(list(itertools.product((-1, 1), repeat=n)), dtype=float)
    else:
        flips = np.random.default_rng(rng_seed).choice((-1.0, 1.0), size=(num_perms, n))
    means = np.abs(flips @ diffs / n)
    # >= keeps the observed assignment in the null, so p is never 0.
    return float(np.mean(means >= observed - 1e-12))


def benjamini_hochberg(p: list[float]) -> list[float]:
    """Step-up BH adjusted p-values, in the order given."""
    m = len(p)
    order = np.argsort(p)
    adj, running = np.empty(m), 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * m / rank)
        adj[i] = min(running, 1.0)
    return adj.tolist()


def compare(rows: list[dict], metric: str, ref_name: str, unit: str) -> list[dict]:
    table = cells(rows, metric)
    if ref_name not in table:
        raise SystemExit(f"reference {ref_name!r} not in {sorted(table)}")
    ref = table[ref_name]

    out = []
    for arm, vals in table.items():
        if arm == ref_name:
            continue
        d = paired(vals, ref, unit)
        if len(d) == 0:
            continue
        out.append({
            "arm": arm,
            "n": len(d),
            "mean": float(np.mean(list(vals.values()))),
            "delta": float(np.mean(d)),
            "sd": float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"),
            "wins": int(np.sum(d < 0)),
            "p": sign_flip_p(d),
        })
    out.sort(key=lambda r: r["delta"])
    for row, padj in zip(out, benjamini_hochberg([r["p"] for r in out])):
        row["p_adj"] = padj
    return out


def report(rows: list[dict], metric: str, ref_name: str) -> None:
    ref_mean = float(np.mean(list(cells(rows, metric)[ref_name].values())))
    print(f"\n{'=' * 78}\n{metric}  vs  {ref_name}  (mean {ref_mean:.2f})"
          f"   -- negative delta = better than the reference\n{'=' * 78}")

    for unit in ("origin", "origin_seed"):
        table = compare(rows, metric, ref_name, unit)
        n = table[0]["n"] if table else 0
        floor = sign_flip_p(np.ones(n)) if n else float("nan")
        print(f"\n[unit = {unit}]  n = {n} paired units, smallest attainable p = {floor:.3f}")
        print(f"  {'arm':24s} {'mean':>7s} {'delta':>8s} {'sd':>7s} {'win':>5s} "
              f"{'p':>7s} {'p_adj':>7s}")
        for r in table:
            star = " *" if r["p_adj"] < 0.05 else ""
            print(f"  {r['arm']:24s} {r['mean']:7.2f} {r['delta']:+8.2f} {r['sd']:7.2f} "
                  f"{r['wins']:3d}/{r['n']:<2d} {r['p']:7.3f} {r['p_adj']:7.3f}{star}")
        if unit == "origin" and floor >= 0.05:
            print(f"  note: with {n} origins nothing can reach p < 0.05. Significance at this "
                  "unit needs the 9-origin grid (plan S9).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("grid", help="name of a results file under seirgnn2/results, without .json")
    ap.add_argument("--ref", default="persistence", help="arm every other arm is paired against")
    ap.add_argument("--metric", default=None,
                    help="val_RMSE and RMSE are both reported when this is omitted")
    args = ap.parse_args()

    rows = load(args.grid)
    for metric in ([args.metric] if args.metric else ["val_RMSE", "RMSE"]):
        report(rows, metric, args.ref)
    print("\nSelection is on val_RMSE (plan R6); RMSE is shown for the record only.\n")


if __name__ == "__main__":
    main()
