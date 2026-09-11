#!/usr/bin/env python3
"""Run Phase 2: Adaptive Graph Learning (Contribution c).

Evaluates:
1. Gated Adaptive GCN: A_blend = sigma(g) * A_fixed + (1 - sigma(g)) * A_adp
2. Pure Adaptive GCN: A = A_adp (isolated ablation without fixed geographic prior)

Benchmarks both against the Phase 1 Baseline GCN and Persistence floor across
3 expanding-window chronological cutoffs (0.55, 0.70, 0.85) and 3 random seeds.
Results are saved to results/phase2_adaptive_results.csv.
"""

from __future__ import annotations

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add src to python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dengue_gnn.data import load_dataset
from dengue_gnn.models import create_adaptive_model, edge_index_to_dense_adj
from dengue_gnn.runner import TrainConfig, run_rolling_origin_cv, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 Adaptive Graph Models.")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["gated", "pure", "both"],
        help="Which adaptive architecture to benchmark (default: both).",
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
        "--emb-dim",
        type=int,
        default=10,
        help="Node embedding dimension d for adaptive graph (default: 10).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick sanity check (epochs=5, patience=3, seed 0).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(ROOT_DIR, "results", "phase2_adaptive_results.csv"),
        help="Output CSV path for fold-level metrics.",
    )
    parser.add_argument(
        "--plot-adj",
        action="store_true",
        help="Generate and save adaptive adjacency matrix heatmap.",
    )
    return parser.parse_args()


def plot_adjacency_comparison(
    model,
    edge_index: torch.Tensor,
    districts: list[str],
    out_path: str,
) -> None:
    """Generate and save side-by-side heatmaps of Fixed, Adaptive, and Blended Adjacency."""
    fixed_adj = edge_index_to_dense_adj(edge_index, n_nodes=len(districts)).cpu().numpy()
    adp_adj = model.get_adaptive_adj().detach().cpu().numpy()
    blend_adj = model.get_effective_adj().detach().cpu().numpy()
    gate = model.get_gate_weight()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(fixed_adj, cmap="viridis", interpolation="nearest")
    axes[0].set_title("1. Fixed Geographic Adjacency", fontsize=12)
    axes[0].set_xticks(range(len(districts)))
    axes[0].set_xticklabels(districts, rotation=90, fontsize=7)
    axes[0].set_yticks(range(len(districts)))
    axes[0].set_yticklabels(districts, fontsize=7)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(adp_adj, cmap="magma", interpolation="nearest")
    axes[1].set_title("2. Learned Self-Adaptive A_adp", fontsize=12)
    axes[1].set_xticks(range(len(districts)))
    axes[1].set_xticklabels(districts, rotation=90, fontsize=7)
    axes[1].set_yticks(range(len(districts)))
    axes[1].set_yticklabels(districts, fontsize=7)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(blend_adj, cmap="plasma", interpolation="nearest")
    axes[2].set_title(f"3. Gated Blended A_blend (sigma(g)={gate:.3f})", fontsize=12)
    axes[2].set_xticks(range(len(districts)))
    axes[2].set_xticklabels(districts, rotation=90, fontsize=7)
    axes[2].set_yticks(range(len(districts)))
    axes[2].set_yticklabels(districts, fontsize=7)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved adjacency heatmaps to: {out_path}")


def main() -> None:
    args = parse_args()

    npy_path = os.path.join(ROOT_DIR, "notebooks", "baseline", "sri_lanka_2013-2022_shifted.npy")
    adj_path = os.path.join(ROOT_DIR, "notebooks", "baseline", "sri_lanka_adj_list.json")

    if not os.path.exists(npy_path) or not os.path.exists(adj_path):
        print(f"Error: Required baseline data missing.\nChecked: {npy_path}\nChecked: {adj_path}")
        sys.exit(1)

    print("=" * 65)
    print("PHASE 2: ADAPTIVE GRAPH LEARNING (CONTRIBUTION c)")
    print("=" * 65)
    raw, edge_index, districts = load_dataset(npy_path, adj_path, n_nodes=25, self_loops=True)
    print(f"Dataset: raw shape {raw.shape} | graph: {edge_index.shape[1]} edges across {len(districts)} districts.")

    modes_to_run = []
    if args.mode in ("gated", "both"):
        modes_to_run.append({"name": "AdaptiveGCN (Gated Blend)", "use_fixed": True})
    if args.mode in ("pure", "both"):
        modes_to_run.append({"name": "PureAdaptiveGCN (No Prior)", "use_fixed": False})

    seeds = [0] if args.quick else [int(s.strip()) for s in args.seeds.split(",")]
    epochs = 5 if args.quick else args.epochs
    patience = 3 if args.quick else args.patience

    all_dfs = []

    for m in modes_to_run:
        name = m["name"]
        use_fixed = m["use_fixed"]

        print("\n" + "-" * 55)
        print(f"Running: {name} (seeds: {seeds}, epochs: {epochs}, patience: {patience})")
        print("-" * 55)

        cfg = TrainConfig(
            model_kind="AdaptiveGCN",
            use_adaptive=True,
            use_fixed=use_fixed,
            emb_dim=args.emb_dim,
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

    print("\n" + "=" * 65)
    print("PHASE 2 BENCHMARK & COMPARISON TABLE")
    print("=" * 65)
    summary_table = combined_df.groupby("model").agg(
        Mean_RMSE=("model_RMSE", "mean"),
        Mean_MAE=("model_MAE", "mean"),
        Mean_SMAPE=("model_SMAPE", "mean"),
        Mean_Gate_Weight=("gate_weight", "mean"),
        Persist_RMSE=("persist_RMSE", "mean"),
        Persist_MAE=("persist_MAE", "mean"),
    ).reset_index()
    print(summary_table.to_string(index=False))

    # Also display comparison with Phase 1 baseline if available
    baseline_csv = os.path.join(ROOT_DIR, "results", "baseline_rolling_origin.csv")
    if os.path.exists(baseline_csv):
        print("\n--- Comparative Leaderboard (vs Phase 1) ---")
        df_base = pd.read_csv(baseline_csv)
        base_summary = df_base.groupby("model").agg(
            Mean_RMSE=("model_RMSE", "mean"),
            Mean_MAE=("model_MAE", "mean"),
            Mean_SMAPE=("model_SMAPE", "mean"),
        ).reset_index()
        phase2_summary = summary_table[["model", "Mean_RMSE", "Mean_MAE", "Mean_SMAPE"]]
        leaderboard = pd.concat([phase2_summary, base_summary], ignore_index=True)
        leaderboard = leaderboard.sort_values(by="Mean_RMSE").reset_index(drop=True)
        print(leaderboard.to_string(index=False))

    # Optional plot
    if args.plot_adj:
        print("\nGenerating adaptive adjacency visualization...")
        plot_out = os.path.join(ROOT_DIR, "results", "adaptive_adj_heatmap.png")
        # Build quick model to plot
        fixed_adj = edge_index_to_dense_adj(edge_index, n_nodes=25)
        model = create_adaptive_model(
            in_dim=33, hidden=64, horizon=3, n_nodes=25, emb_dim=args.emb_dim,
            use_adaptive=True, use_fixed=True, fixed_adj=fixed_adj,
        )
        plot_adjacency_comparison(model, edge_index, districts, plot_out)


if __name__ == "__main__":
    main()
