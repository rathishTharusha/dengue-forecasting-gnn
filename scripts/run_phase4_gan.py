#!/usr/bin/env python3
"""Run Phase 4: GAN Augmentation (Contribution b).

Evaluates:
1. No Augmentation baseline (standard GCN)
2. Mandatory Comparator: Heuristic Jittering & Magnitude Warping
3. Conditional WGAN-GP Augmentation:
   Synthesizes realistic (X -> Y) sequence augmentations to address dataset scarcity.

Evaluated across 3 expanding-window chronological cutoffs (0.55, 0.70, 0.85)
and 3 random seeds under the frozen rolling-origin protocol.
Results are saved to results/phase4_gan_results.csv.
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

from dengue_gnn.data import (
    build_fold_tensors,
    get_persistence_forecast,
    get_rolling_origin_splits,
    load_dataset,
)
from dengue_gnn.gan import (
    heuristic_augmentation,
    synthesize_gan_augmentation,
    train_wgan_gp,
)
from dengue_gnn.metrics import metrics
from dengue_gnn.models import create_baseline_model
from dengue_gnn.runner import TrainConfig, evaluate_fold, set_seed, train_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 GAN Augmentation Benchmarks.")
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
        help="Maximum training epochs per fold for downstream GNN (default: 120).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience (default: 25).",
    )
    parser.add_argument(
        "--gan-epochs",
        type=int,
        default=40,
        help="Number of epochs to train WGAN-GP on each fold (default: 40).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick sanity check (downstream epochs=5, GAN epochs=5, seed 0).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(ROOT_DIR, "results", "phase4_gan_results.csv"),
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
    print("PHASE 4: GAN AUGMENTATION (CONTRIBUTION b)")
    print("=" * 65)
    raw, edge_index, districts = load_dataset(npy_path, adj_path, n_nodes=25, self_loops=True)
    print(f"Dataset: raw shape {raw.shape} | graph: {edge_index.shape[1]} edges across {len(districts)} districts.")

    seeds = [0] if args.quick else [int(s.strip()) for s in args.seeds.split(",")]
    epochs = 5 if args.quick else args.epochs
    patience = 3 if args.quick else args.patience
    gan_epochs = 5 if args.quick else args.gan_epochs

    splits = get_rolling_origin_splits(
        total_timesteps=raw.shape[0],
        window=3,
        horizon=3,
        origins=(0.55, 0.70, 0.85),
        test_frac=0.15,
        val_weeks=30,
    )

    T, N, Fd = raw.shape
    in_dim = Fd * 3

    aug_strategies = ["No Augmentation", "Heuristic (Jitter+Warp)", "WGAN-GP Augmentation"]
    records = []

    for s in splits:
        fold_idx = s["fold"]
        origin = s["origin"]
        train_ids = s["train_ids"]
        val_ids = s["val_ids"]
        test_ids = s["test_ids"]

        print(f"\n>>> Processing Fold {fold_idx} (origin {origin:.2f}, {len(train_ids)} train wks, {len(test_ids)} test wks) <<<")

        # Prepare base fold tensors
        tr_base, _ = build_fold_tensors(raw, train_ids, train_ids, window=3, horizon=3, log_transform=True)
        va_base, _ = build_fold_tensors(raw, train_ids, val_ids, window=3, horizon=3, log_transform=True)
        te_base, _ = build_fold_tensors(raw, train_ids, test_ids, window=3, horizon=3, log_transform=True)

        # Baseline persistence
        p_pred, p_truth = get_persistence_forecast(raw, test_ids, horizon=3, cases_idx=5)
        p_metrics = metrics(p_pred, p_truth)["overall"]

        # 1. Heuristic augmented data
        tr_heuristic = heuristic_augmentation(tr_base, noise_std=0.03, scale_range=(0.95, 1.05))

        # 2. Train WGAN-GP on training pool
        print(f"  Training Conditional WGAN-GP for Fold {fold_idx} ({gan_epochs} epochs)...")
        set_seed(0)
        generator, _ = train_wgan_gp(
            train_data=tr_base,
            in_dim=in_dim,
            latent_dim=16,
            horizon=3,
            epochs=gan_epochs,
            batch_size=32,
            n_critic=5,
            lr=1e-4,
            verbose=False,
        )
        tr_gan = synthesize_gan_augmentation(generator, tr_base, ratio=0.5, latent_dim=16)

        strategy_packs = {
            "No Augmentation": tr_base,
            "Heuristic (Jitter+Warp)": tr_heuristic,
            "WGAN-GP Augmentation": tr_gan,
        }

        for strat_name, tr_data in strategy_packs.items():
            rmse_runs, mae_runs, smape_runs = [], [], []

            for seed in seeds:
                set_seed(seed)
                cfg = TrainConfig(
                    model_kind="GCN",
                    hidden=64,
                    dropout=0.1,
                    lr=1e-3,
                    weight_decay=5e-4,
                    epochs=epochs,
                    patience=patience,
                    grad_clip=5.0,
                    residual=True,
                    log_transform=True,
                )
                model = create_baseline_model(in_dim=in_dim, hidden=64, horizon=3, kind="GCN")
                trained_model, _ = train_fold(model, tr_data, va_base, edge_index, cfg, verbose=False)
                sc = evaluate_fold(trained_model, te_base, edge_index, residual=True, device=cfg.device)["overall"]

                rmse_runs.append(sc["RMSE"])
                mae_runs.append(sc["MAE"])
                smape_runs.append(sc["SMAPE"])

            row = {
                "fold": fold_idx,
                "origin": origin,
                "strategy": strat_name,
                "n_train_samples": tr_data.X.shape[0],
                "RMSE": float(np.mean(rmse_runs)),
                "RMSE_std": float(np.std(rmse_runs)),
                "MAE": float(np.mean(mae_runs)),
                "SMAPE": float(np.mean(smape_runs)),
                "persist_RMSE": p_metrics["RMSE"],
                "persist_MAE": p_metrics["MAE"],
            }
            records.append(row)
            print(
                f"  [{strat_name:23s}] (N={row['n_train_samples']}) | "
                f"RMSE: {row['RMSE']:.2f} +/- {row['RMSE_std']:.2f} | MAE: {row['MAE']:.2f}"
            )

    df_results = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_results.to_csv(args.out, index=False)
    print(f"\nSaved detailed metrics to: {args.out}")

    print("\n" + "=" * 65)
    print("PHASE 4: GAN AUGMENTATION BENCHMARK SUMMARY")
    print("=" * 65)
    summary_table = df_results.groupby("strategy").agg(
        Mean_RMSE=("RMSE", "mean"),
        Mean_MAE=("MAE", "mean"),
        Mean_SMAPE=("SMAPE", "mean"),
        Persist_RMSE=("persist_RMSE", "mean"),
        Persist_MAE=("persist_MAE", "mean"),
    ).reset_index().sort_values(by="Mean_RMSE").reset_index(drop=True)
    print(summary_table.to_string(index=False))

    # Cross-phase leaderboard
    print("\n" + "=" * 65)
    print("UNIFIED CROSS-PHASE LEADERBOARD (PHASES 1, 2, 3, 4)")
    print("=" * 65)
    lb_rows = []
    p_rmse = summary_table["Persist_RMSE"].iloc[0]
    p_mae = summary_table["Persist_MAE"].iloc[0]
    lb_rows.append({"Model / Strategy": "Persistence Floor", "Phase": "Baseline", "Mean_RMSE": p_rmse, "Mean_MAE": p_mae})

    # Phase 1
    p1_csv = os.path.join(ROOT_DIR, "results", "baseline_rolling_origin.csv")
    if os.path.exists(p1_csv):
        for m, grp in pd.read_csv(p1_csv).groupby("model"):
            lb_rows.append({"Model / Strategy": f"{m} (Baseline)", "Phase": "Phase 1", "Mean_RMSE": grp["model_RMSE"].mean(), "Mean_MAE": grp["model_MAE"].mean()})

    # Phase 2
    p2_csv = os.path.join(ROOT_DIR, "results", "phase2_adaptive_results.csv")
    if os.path.exists(p2_csv):
        for m, grp in pd.read_csv(p2_csv).groupby("model"):
            lb_rows.append({"Model / Strategy": f"{m} (Adaptive)", "Phase": "Phase 2", "Mean_RMSE": grp["model_RMSE"].mean(), "Mean_MAE": grp["model_MAE"].mean()})

    # Phase 3
    p3_csv = os.path.join(ROOT_DIR, "results", "phase3_physics_results.csv")
    if os.path.exists(p3_csv):
        for m, grp in pd.read_csv(p3_csv).groupby("experiment"):
            lb_rows.append({"Model / Strategy": m, "Phase": "Phase 3", "Mean_RMSE": grp["model_RMSE"].mean(), "Mean_MAE": grp["model_MAE"].mean()})

    # Phase 4
    for strat, grp in df_results.groupby("strategy"):
        lb_rows.append({"Model / Strategy": f"GCN + {strat}", "Phase": "Phase 4", "Mean_RMSE": grp["RMSE"].mean(), "Mean_MAE": grp["MAE"].mean()})

    df_lb = pd.DataFrame(lb_rows).sort_values(by="Mean_RMSE").reset_index(drop=True)
    print(df_lb.to_string(index=False))


if __name__ == "__main__":
    main()
