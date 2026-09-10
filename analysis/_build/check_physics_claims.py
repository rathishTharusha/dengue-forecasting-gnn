"""Paired tests for the physics-loss sweep, which the summary report omits.

``analysis/results/physics_vs_adaptive_report.md`` concludes that the
physics-informed loss "directly outperforms adaptive graph architectures and
successfully breaches the persistence benchmark floor", on differences of 0.08 to
0.60 RMSE. The arms it compares were run on the same origins and seeds, so the
comparison can and should be paired -- and the report gives unpaired means with
cross-origin standard deviations of 6.6 to 7.9 beside them, which is fold
difficulty rather than arm variance.

This script runs the missing test on the report's own saved records:

* each physics arm against its **own architecture's** ``base``, paired on
  (origin, seed);
* each arm against the persistence floor on the same windows;
* and a check of whether the envelope arm is numerically distinct from ``base``
  at all, since EXP-014 found the same ceiling inert against a model whose
  predicted growth sits far below it.

Run::

    python analysis/_build/check_physics_claims.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent.parent
SWEEP = REPO / "analysis" / "results" / "physics_sweep"
OUT = REPO / "analysis" / "results" / "physics_claims_check.json"

ARMS = ["base", "envelope", "spatial", "spatial_log", "composite",
        "composite_log", "outbreak_aware"]


def load_records(directory: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(directory.glob("physics_envelope_*.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise SystemExit(f"no physics_envelope_*.json under {directory}")
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=SWEEP,
                    help="directory of physics_envelope_*.json files")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records = load_records(args.dir)
    print(f"{len(records)} records from {args.dir}")
    model = [r for r in records if r["arch"] != "persistence"]
    floor_rows = [r for r in records if r["arch"] == "persistence"]
    archs = sorted({r["arch"] for r in model})
    report: dict = {"paired_vs_base": {}, "vs_floor": {}, "envelope_identical": {}}

    def key(r):
        return (r["origin"], r["seed"])

    print("Paired against the SAME architecture's `base`, on matched (origin, seed).")
    print("Positive dRMSE means the physics arm is WORSE.\n")
    print(f"{'arch':8s}{'arm':16s}{'n':>4s}{'mean dRMSE':>12s}{'sd':>8s}{'better':>9s}{'p':>9s}")
    print("-" * 66)
    for arch in archs:
        rows = [r for r in model if r["arch"] == arch]
        basis = {key(r): r["RMSE_clean"] for r in rows if r["increment"] == "base"}
        for arm in ARMS[1:]:
            pairs = [(r["RMSE_clean"], basis[key(r)]) for r in rows
                     if r["increment"] == arm and key(r) in basis]
            if len(pairs) < 3:
                continue
            d = np.array([a - b for a, b in pairs])
            if np.allclose(d, 0.0):
                p = float("nan")
            else:
                _, p = stats.ttest_rel([a for a, _ in pairs], [b for _, b in pairs])
            report["paired_vs_base"].setdefault(arch, {})[arm] = {
                "mean_delta": float(d.mean()), "sd": float(d.std(ddof=1)),
                "better": int((d < 0).sum()), "n": len(d), "p": float(p),
            }
            print(f"{arch:8s}{arm:16s}{len(d):>4d}{d.mean():+12.3f}{d.std(ddof=1):8.3f}"
                  f"{int((d < 0).sum()):6d}/{len(d):<3d}"
                  f"{'   n/a' if np.isnan(p) else f'{p:9.3f}'}")

    # Is the envelope arm distinct from base at all? EXP-014 found the same
    # SEIR-derived ceiling inert because predicted growth never approaches it.
    print("\nIs `envelope` numerically distinct from `base`?")
    for arch in archs:
        rows = [r for r in model if r["arch"] == arch]
        b = {key(r): r["RMSE_clean"] for r in rows if r["increment"] == "base"}
        e = {key(r): r["RMSE_clean"] for r in rows if r["increment"] == "envelope"}
        shared = sorted(set(b) & set(e))
        identical = sum(1 for k in shared if abs(b[k] - e[k]) < 1e-9)
        report["envelope_identical"][arch] = {"identical": identical, "n": len(shared)}
        print(f"  {arch:8s}{identical}/{len(shared)} runs bit-identical to base")

    # Why it is inert on some architectures and not others: the hinge only acts
    # where predicted growth reaches the ceiling. composite = envelope + spatial,
    # so composite == spatial exactly wherever the envelope contributes nothing.
    print("\nDoes the envelope bind? (composite == spatial iff it does not)")
    for arch in archs:
        rows = [r for r in model if r["arch"] == arch]
        sp = {key(r): r["RMSE_clean"] for r in rows if r["increment"] == "spatial"}
        co = {key(r): r["RMSE_clean"] for r in rows if r["increment"] == "composite"}
        shared = sorted(set(sp) & set(co))
        same = sum(1 for k in shared if abs(sp[k] - co[k]) < 1e-9)
        gmax = [r.get("growth_max") for r in rows if r.get("growth_max") is not None]
        peak = max(gmax) if gmax else float("nan")
        report.setdefault("envelope_binds", {})[arch] = {
            "composite_equals_spatial": same, "n": len(shared), "max_growth": peak}
        verdict = "INERT" if same == len(shared) else "binds"
        print(f"  {arch:8s}{verdict:6s} composite==spatial {same}/{len(shared)}, "
              f"max predicted growth {peak:.2f} vs ceiling 2.3884")

    # Against the floor, on the same windows, paired by (origin, seed).
    floor = {r["origin"]: r["RMSE_clean"] for r in floor_rows}
    if not floor:
        print("\n(no persistence rows in these files; floor comparison skipped)")
    print(f"\n{'arch':8s}{'arm':16s}{'mean clean':>12s}{'vs floor':>10s}{'p':>9s}")
    print("-" * 55)
    for arch in archs:
        for arm in ARMS:
            rows = [r for r in model if r["arch"] == arch and r["increment"] == arm]
            if not rows:
                continue
            vals = np.array([r["RMSE_clean"] for r in rows])
            if floor:
                base_vals = np.array([floor[r["origin"]] for r in rows])
                _, p = stats.ttest_rel(vals, base_vals)
            else:
                base_vals, p = np.full(len(vals), np.nan), float("nan")
            delta = float(np.nanmean(vals - base_vals))
            report["vs_floor"].setdefault(arch, {})[arm] = {
                "mean": float(vals.mean()), "delta": delta, "p": float(p)}
            print(f"{arch:8s}{arm:16s}{vals.mean():12.2f}{delta:+10.3f}"
                  f"{'   n/a' if np.isnan(p) else f'{p:9.3f}'}")

    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
