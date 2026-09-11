#!/usr/bin/env python3
"""Run Phase 3: Physics-Informed Loss (Contribution a).

Evaluates:
1. Lambda regularizer sweep: lambda_phys in {0.0, 0.01, 0.10, 1.0}
2. Unified PIAG-Net: Adaptive Graph + optimal Physics-informed loss (lambda_phys=0.10, lambda_smooth=0.01)

Benchmarks against Phase 1 and Phase 2 baselines across 3 expanding-window
chronological cutoffs (0.55, 0.70, 0.85) and 3 random seeds.
Results are saved to results/phase3_physics_results.csv.
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch

# Add src to python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dengue_gnn.data import load_dataset
from dengue_gnn.runner import TrainConfig, run_rolling_origin_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 Physics-Informed Loss Benchmarks.")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated random seeds (default: 0,1,2).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="Maximum training epochs per fold (default: 120).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience (default: 25).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick sanity check (epochs=5, patience=3, seed 0).",
    )
    parser.add_argument(
        "--sweep-only",
        action="store_true",
        help="Run only the lambda sweep on GCN.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(ROOT_DIR, "results", "phase3_physics_results.csv"),
        help="Output CSV path for fold-level metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    npy_path = os.path.join(ROOT_DIR, "notebooks", "baseline", "sri_lanka_2013-2022_shifted.npy")
    adj_path = os.path.join(ROOT_DIR, "notebooks", "baseline", "sri_lanka_adj_list.json")

    if not os.path.exists(npy_path) or not os.path.exists(adj_path):
        print(f"Error: Required baseline data missing.\nChecked: {npy_path}\nChecked: {adj_path}")
        sys.exit(1)

    print("=" * 65)
    print("PHASE 3: PHYSICS-INFORMED LOSS (CONTRIBUTION a)")
    print("=" * 65)
    raw, edge_index, districts = load_dataset(npy_path, adj_path, n_nodes=25, self_loops=True)
    print(f"Dataset: raw shape {raw.shape} | graph: {edge_index.shape[1]} edges across {len(districts)} districts.")

    seeds = [0] if args.quick else [int(s.strip()) for s in args.seeds.split(",")]
    epochs = 5 if args.quick else args.epochs
    patience = 3 if args.quick else args.patience

    # Experiments to run:
    # 1. Lambda sweep on GCN: lambda_phys in {0.01, 0.10, 1.0}
    # 2. Combined PIAG-Net: AdaptiveGCN + lambda_phys=0.10 + lambda_smooth=0.01
    experiments = []

    if args.quick:
        # Quick check: 2 configs
        experiments.append({
            "name": "GCN (lambda_phys=0.10)",
            "use_adaptive": False,
            "lambda_phys": 0.10,
            "lambda_smooth": 0.0,
        })
        experiments.append({
            "name": "PIAG-Net (Combined)",
            "use_adaptive": True,
            "lambda_phys": 0.10,
            "lambda_smooth": 0.01,
        })
    else:
        # Full evaluation
        for lam in [0.01, 0.10, 1.0]:
            experiments.append({
                "name": f"GCN (lambda_phys={lam})",
                "use_adaptive": False,
                "lambda_phys": lam,
                "lambda_smooth": 0.0,
            })
        if not args.sweep_only:
            experiments.append({
                "name": "PIAG-Net (Combined)",
                "use_adaptive": True,
                "lambda_phys": 0.10,
                "lambda_smooth": 0.01,
            })

    all_dfs = []

    for exp in experiments:
        name = exp["name"]
        use_adaptive = exp["use_adaptive"]
        lam_phys = exp["lambda_phys"]
        lam_smooth = exp["lambda_smooth"]

        print("\n" + "-" * 55)
        print(f"Running: {name} (seeds: {seeds}, epochs: {epochs}, patience: {patience})")
        print("-" * 55)

        cfg = TrainConfig(
            model_kind="AdaptiveGCN" if use_adaptive else "GCN",
            use_adaptive=use_adaptive,
            use_fixed=True,
            emb_dim=10,
            n_nodes=25,
            hidden=64,
            dropout=0.1,
            lr=1e-3,
            weight_decay=5e-4,
            epochs=epochs,
            patience=patience,
            grad_clip=5.0,
            residual=True,
            log_transform=True,
            window=3,
            horizon=3,
            cases_idx=5,
            use_all_feats=True,
            lambda_phys=lam_phys,
            lambda_smooth=lam_smooth,
        )

        df_res, summary = run_rolling_origin_cv(
            raw=raw,
            edge_index=edge_index,
            config=cfg,
            origins=(0.55, 0.70, 0.85),
            test_frac=0.15,
            val_weeks=30,
            seeds=seeds,
            verbose=True,
        )
        df_res["experiment"] = name
        all_dfs.append(df_res)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    combined_df.to_csv(args.out, index=False)
    print(f"\nSaved detailed fold metrics to: {args.out}")

    print("\n" + "=" * 65)
    print("PHASE 3 BENCHMARK & COMPARISON TABLE")
    print("=" * 65)
    summary_table = combined_df.groupby("experiment").agg(
        Mean_RMSE=("model_RMSE", "mean"),
        Mean_MAE=("model_MAE", "mean"),
        Mean_SMAPE=("model_SMAPE", "mean"),
        Persist_RMSE=("persist_RMSE", "mean"),
        Persist_MAE=("persist_MAE", "mean"),
    ).reset_index()
    print(summary_table.to_string(index=False))

    # Cross-phase unified leaderboard
    print("\n" + "=" * 65)
    print("UNIFIED CROSS-PHASE LEADERBOARD (PHASES 1, 2, 3)")
    print("=" * 65)
    records = []
    # Add persistence
    p_rmse = summary_table["Persist_RMSE"].iloc[0]
    p_mae = summary_table["Persist_MAE"].iloc[0]
    records.append({"Model / Phase": "Persistence Floor", "Phase": "Baseline", "Mean_RMSE": p_rmse, "Mean_MAE": p_mae})

    # Add Phase 1
    p1_csv = os.path.join(ROOT_DIR, "results", "baseline_rolling_origin.csv")
    if os.path.exists(p1_csv):
        df_p1 = pd.read_csv(p1_csv)
        for m, grp in df_p1.groupby("model"):
            records.append({"Model / Phase": f"{m} (Baseline)", "Phase": "Phase 1", "Mean_RMSE": grp["model_RMSE"].mean(), "Mean_MAE": grp["model_MAE"].mean()})

    # Add Phase 2
    p2_csv = os.path.join(ROOT_DIR, "results", "phase2_adaptive_results.csv")
    if os.path.exists(p2_csv):
        df_p2 = pd.read_csv(p2_csv)
        for m, grp in df_p2.groupby("model"):
            records.append({"Model / Phase": f"{m} (Adaptive)", "Phase": "Phase 2", "Mean_RMSE": grp["model_RMSE"].mean(), "Mean_MAE": grp["model_MAE"].mean()})

    # Add Phase 3
    for exp, grp in combined_df.groupby("experiment"):
        records.append({"Model / Phase": exp, "Phase": "Phase 3", "Mean_RMSE": grp["model_RMSE"].mean(), "Mean_MAE": grp["model_MAE"].mean()})

    df_unified = pd.DataFrame(records).sort_values(by="Mean_RMSE").reset_index(drop=True)
    print(df_unified.to_string(index=False))


if __name__ == "__main__":
    main()
