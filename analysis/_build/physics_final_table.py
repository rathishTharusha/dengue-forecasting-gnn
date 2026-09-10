"""The physics result, at the sample size that settles it.

The original sweep ran 3 origins x 3 seeds (n=9) per arm and reported unpaired
means beside cross-origin standard deviations of 6.6-7.9 -- fold difficulty, not
arm variance. At that n the spatial term sat at -0.082 with 9/9 folds better and
p=0.057: consistent in direction, short of significance, and exactly the case more
seeds decide.

This script scores the extended run (20-25 seeds per arm) the way the comparison
should be scored: paired on matched (origin, seed) against the *same
architecture's* base, artifact-free, with the persistence floor split the same way.

Run::

    python analysis/_build/physics_final_table.py
    python analysis/_build/physics_final_table.py --dir analysis/results/physics_extended
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = REPO / "analysis" / "results" / "physics_extended"
OUT = REPO / "analysis" / "results" / "physics_final.json"

#: Persistence, artifact-free, per origin. From adaptive.persistence_scores; the
#: floor must be split the same way the models are or the comparison is between
#: two different test sets.
FLOOR = {0.55: 26.882223151529477, 0.7: 38.88532915547214, 0.85: 22.79547681977414}


def load(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("physics_envelope_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no physics_envelope_*.json under {directory}")
    return [r for r in rows if r["arch"] != "persistence"]


def paired(rows: list[dict], arch: str, arm: str, field: str = "RMSE_clean"):
    """Arm minus its own architecture's base, on matched (origin, seed)."""
    def key(r):
        return (r["origin"], r["seed"])

    sub = [r for r in rows if r["arch"] == arch]
    base = {key(r): r[field] for r in sub if r["increment"] == "base"}
    pairs = [(r[field], base[key(r)]) for r in sub
             if r["increment"] == arm and key(r) in base]
    if len(pairs) < 3:
        return None
    d = np.array([a - b for a, b in pairs])
    if np.allclose(d, 0.0):
        return {"n": len(d), "delta": 0.0, "sd": 0.0, "better": 0, "p": float("nan"),
                "inert": True}
    _, p = stats.ttest_rel([a for a, _ in pairs], [b for _, b in pairs])
    return {"n": len(d), "delta": float(d.mean()), "sd": float(d.std(ddof=1)),
            "better": int((d < 0).sum()), "p": float(p), "inert": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = load(args.dir)
    archs = sorted({r["arch"] for r in rows})
    arms = [a for a in ("envelope", "spatial", "spatial_log", "composite",
                        "composite_log", "outbreak_aware")
            if any(r["increment"] == a for r in rows)]
    report: dict = {"paired": {}, "absolute": {}}

    print("Paired against the same architecture's `base`, artifact-free.")
    print("Negative dRMSE = the physics arm is better.\n")
    print(f"{'arch':8s}{'arm':16s}{'n':>4s}{'dRMSE':>10s}{'sd':>8s}{'better':>10s}{'p':>10s}")
    print("-" * 66)
    for arch in archs:
        for arm in arms:
            res = paired(rows, arch, arm)
            if res is None:
                continue
            report["paired"].setdefault(arch, {})[arm] = res
            mark = "  INERT" if res["inert"] else ""
            pstr = "   n/a" if np.isnan(res["p"]) else f"{res['p']:10.4f}"
            print(f"{arch:8s}{arm:16s}{res['n']:>4d}{res['delta']:+10.4f}"
                  f"{res['sd']:8.4f}{res['better']:>7d}/{res['n']:<3d}{pstr}{mark}")

    # Absolute standing against the floor, on the same windows.
    print(f"\n{'arch':8s}{'arm':16s}{'n':>4s}{'clean RMSE':>12s}{'vs floor':>10s}{'p':>10s}")
    print("-" * 62)
    for arch in archs:
        for arm in ["base", *arms]:
            sub = [r for r in rows if r["arch"] == arch and r["increment"] == arm]
            if not sub:
                continue
            # An arm whose origins are unevenly represented has a meaningless
            # absolute mean: origin 0.85 scores ~22.8 artifact-free against 38.9
            # for 0.70, so a partial run reads as a large spurious difference.
            counts = {o: sum(1 for r in sub if r["origin"] == o) for o in FLOOR}
            balanced = len(set(counts.values())) == 1
            vals = np.array([r["RMSE_clean"] for r in sub])
            floor = np.array([FLOOR[r["origin"]] for r in sub])
            _, p = stats.ttest_rel(vals, floor)
            report["absolute"].setdefault(arch, {})[arm] = {
                "n": len(vals), "mean": float(vals.mean()),
                "delta_floor": float((vals - floor).mean()), "p": float(p),
                "balanced": balanced}
            warn = "" if balanced else "   << origins unbalanced, mean not comparable"
            print(f"{arch:8s}{arm:16s}{len(vals):>4d}{vals.mean():12.3f}"
                  f"{(vals - floor).mean():+10.3f}{p:10.4f}{warn}")

    print(f"\nFloor, artifact-free: {np.mean(list(FLOOR.values())):.2f} "
          f"(per origin {', '.join(f'{k}: {v:.2f}' for k, v in sorted(FLOOR.items()))})")
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
