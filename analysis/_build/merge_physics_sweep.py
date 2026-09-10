"""Join individual per-architecture physics sweep results into a single table.

Usage:
    python analysis/_build/merge_physics_sweep.py
    python analysis/_build/merge_physics_sweep.py --in-dir analysis/results/sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent.parent
IN_DIR = REPO / "analysis" / "results" / "physics_sweep"
OUT_JSON = REPO / "analysis" / "results" / "physics_sweep_summary.json"
OUT_CSV = REPO / "analysis" / "results" / "physics_sweep_summary.csv"

ARMS = ["base", "envelope", "spatial", "composite", "outbreak_aware"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, default=str(IN_DIR))
    args = ap.parse_args()

    in_path = Path(args.in_dir)
    files = sorted(list(in_path.glob("physics_envelope_*.json")))
    if not files:
        print(f"No physics_envelope_*.json found in {in_path}")
        return 1

    records = []
    seen = set()

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for r in data:
            key = (r.get("arch"), r.get("increment"), r.get("origin"), r.get("seed"))
            if key not in seen:
                seen.add(key)
                records.append(r)

    df = pd.DataFrame(records)
    print(f"Loaded {len(records)} unique records from {len(files)} files: {[f.name for f in files]}")

    df.to_json(OUT_JSON, indent=2, orient="records")
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved merged results to {OUT_JSON} and {OUT_CSV}")

    # Print summary tables
    p_sub = df[df["arch"] == "persistence"]
    if len(p_sub) > 0:
        p_floor = p_sub["RMSE_clean"].mean()
        p_all = p_sub["RMSE"].mean()
        print("\n" + "=" * 85)
        print(f"PERSISTENCE BENCHMARK FLOOR: All RMSE = {p_all:.2f} | Clean RMSE = {p_floor:.2f}")
        print("=" * 85)

    archs = sorted([a for a in df["arch"].unique() if a != "persistence"])
    for arch in archs:
        sub = df[df["arch"] == arch]
        n_runs = len(sub[sub["increment"] == "base"])
        print(f"\n--- Architecture: {arch} (n={n_runs} runs per arm) ---")
        summary = sub.groupby("increment").agg({
            "RMSE": ["mean", "std"],
            "RMSE_clean": ["mean", "std"],
            "MAE_clean": ["mean"],
            "growth_max": ["mean"]
        })
        print(summary)

        # Paired test against base
        base_sub = sub[sub["increment"] == "base"].drop_duplicates(subset=["origin", "seed"])
        base_clean = base_sub.set_index(["origin", "seed"])["RMSE_clean"]
        for arm in [a for a in ARMS if a != "base"]:
            arm_sub = sub[sub["increment"] == arm].drop_duplicates(subset=["origin", "seed"])
            arm_clean = arm_sub.set_index(["origin", "seed"])["RMSE_clean"]
            common = sorted(set(base_clean.index) & set(arm_clean.index))
            if len(common) >= 3:
                b_vals = np.array([base_clean.loc[c] for c in common], dtype=float)
                a_vals = np.array([arm_clean.loc[c] for c in common], dtype=float)
                diffs = a_vals - b_vals
                _, pval = stats.ttest_rel(a_vals, b_vals)
                better = (diffs < 0).sum()
                print(f"  vs base: {arm:14s} dRMSE={diffs.mean():+6.2f} (better: {better}/{len(common)}, p={pval:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
