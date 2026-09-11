#!/usr/bin/env python3
"""Run Phase 1 baseline reproduction under the frozen rolling-origin protocol.

Evaluates Naive Persistence, Spatio-Temporal GCN, and Spatio-Temporal GAT across
3 expanding-window chronological cutoffs (0.55, 0.70, 0.85) and 3 random seeds.
Results are saved to results/baseline_rolling_origin.csv.
"""

from __future__ import annotations

import argparse
import os
import sys
import pandas as pd

# Add src to python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dengue_gnn.data import load_dataset
from dengue_gnn.runner import TrainConfig, run_rolling_origin_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 Baseline Models.")
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["GCN", "GAT", "both"],
        help="Which baseline GNN backbone to benchmark (default: both).",
    )
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
        "--out",
        type=str,
        default=os.path.join(ROOT_DIR, "results", "baseline_rolling_origin.csv"),
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

    print("=" * 60)
    print("PHASE 1: BASELINE & PROTOCOL REPRODUCTION")
    print("=" * 60)
    print(f"Loading data from: {npy_path}")
    raw, edge_index, districts = load_dataset(npy_path, adj_path, n_nodes=25, self_loops=True)
    print(f"Loaded: raw shape {raw.shape} | graph: {edge_index.shape[1]} directed edges across {len(districts)} districts.")

    models_to_run = ["GCN", "GAT"] if args.model == "both" else [args.model]
    seeds = [0] if args.quick else [int(s.strip()) for s in args.seeds.split(",")]
    epochs = 5 if args.quick else args.epochs
    patience = 3 if args.quick else args.patience

    all_dfs = []

    for m in models_to_run:
        print("\n" + "-" * 50)
        print(f"Running Baseline: {m} (seeds: {seeds}, epochs: {epochs}, patience: {patience})")
        print("-" * 50)

        cfg = TrainConfig(
            model_kind=m,
            hidden=64,
            gat_heads=8,
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
        all_dfs.append(df_res)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    combined_df.to_csv(args.out, index=False)
    print(f"\nSaved detailed fold metrics to: {args.out}")

    print("\n" + "=" * 60)
    print("PHASE 1 BENCHMARK SUMMARY TABLE")
    print("=" * 60)
    summary_table = combined_df.groupby("model").agg(
        Mean_RMSE=("model_RMSE", "mean"),
        Mean_MAE=("model_MAE", "mean"),
        Mean_SMAPE=("model_SMAPE", "mean"),
        Persist_RMSE=("persist_RMSE", "mean"),
        Persist_MAE=("persist_MAE", "mean"),
    ).reset_index()
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
