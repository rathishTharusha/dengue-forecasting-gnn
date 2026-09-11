#!/usr/bin/env python3
"""Run Phase 5: Full Master Ablation & Synthesis.

Executes the definitive 6-row Master Ablation benchmark under the frozen
rolling-origin protocol (3 folds x 3 seeds):
Row 0: Persistence Floor
Row 1: GCN Baseline (Fixed graph, data MSE)
Row 2: + Adaptive Graph (Contribution c: Gated blend)
Row 3: + Physics-Informed Loss (Contribution a: lambda_phys = 1.0)
Row 4: + Data Augmentation (Contribution b: Jittering & Warping)
Row 5: All Combined (Unified Architecture: Adaptive + Physics + Augmentation)

Computes point metrics (RMSE, MAE, SMAPE), spatial autocorrelation (Moran's I),
and temporal peak timing error (PTE).
Exports results to results/master_ablation_table.csv and outputs publication-ready tables.
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
from dengue_gnn.gan import heuristic_augmentation
from dengue_gnn.metrics import (
    compute_morans_i,
    compute_peak_timing_error,
    metrics,
)
from dengue_gnn.models import (
    create_adaptive_model,
    create_baseline_model,
    edge_index_to_dense_adj,
)
from dengue_gnn.runner import TrainConfig, evaluate_fold, set_seed, train_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 5 Master Ablation Benchmark.")
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
        default=os.path.join(ROOT_DIR, "results", "master_ablation_table.csv"),
        help="Output CSV path for master ablation results.",
    )
    return parser.parse_args()


def evaluate_predictions_full(
    model,
    eval_data,
    edge_index: torch.Tensor,
    dense_adj: np.ndarray,
    residual: bool = True,
    device: str = "cpu",
) -> tuple[dict[str, float], float, float]:
    """Evaluate model and return overall point metrics, Moran's I, and Peak Timing Error."""
    model.eval()
    dev = torch.device(device)
    ei = edge_index.to(dev)

    Xe, Ye, Pe = eval_data.X, eval_data.Y, eval_data.P
    preds_raw = []
    truths_raw = []

    with torch.no_grad():
        for i in range(Xe.shape[0]):
            p = model(Xe[i].to(dev), ei).cpu()
            if residual:
                p = p + Pe[i]
            preds_raw.append(eval_data.inv_fn(p))
            truths_raw.append(eval_data.inv_fn(Ye[i]))

    pred_arr = np.stack(preds_raw)
    truth_arr = np.stack(truths_raw)

    score_dict = metrics(pred_arr, truth_arr)["overall"]
    resids = pred_arr - truth_arr
    moran_i = compute_morans_i(resids, dense_adj)
    pte = compute_peak_timing_error(pred_arr, truth_arr)

    return score_dict, moran_i, pte


def main() -> None:
    args = parse_args()

    npy_path = os.path.join(ROOT_DIR, "notebooks", "baseline", "sri_lanka_2013-2022_shifted.npy")
    adj_path = os.path.join(ROOT_DIR, "notebooks", "baseline", "sri_lanka_adj_list.json")

    if not os.path.exists(npy_path) or not os.path.exists(adj_path):
        print(f"Error: Required baseline data missing.\nChecked: {npy_path}\nChecked: {adj_path}")
        sys.exit(1)

    print("=" * 70)
    print("PHASE 5: MASTER ABLATION & PROJECT SYNTHESIS")
    print("=" * 70)
    raw, edge_index, districts = load_dataset(npy_path, adj_path, n_nodes=25, self_loops=True)
    dense_adj_np = edge_index_to_dense_adj(edge_index, n_nodes=25, self_loops=False).cpu().numpy()
    print(f"Dataset: raw shape {raw.shape} | graph: {edge_index.shape[1]} edges across {len(districts)} districts.")

    seeds = [0] if args.quick else [int(s.strip()) for s in args.seeds.split(",")]
    epochs = 5 if args.quick else args.epochs
    patience = 3 if args.quick else args.patience

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

    # Define the 6 canonical ablation configurations
    ablation_configs = [
        {
            "row_id": 0,
            "name": "Persistence Floor",
            "type": "persistence",
        },
        {
            "row_id": 1,
            "name": "GCN Baseline (residual + log)",
            "type": "gnn",
            "model_kind": "GCN",
            "use_adaptive": False,
            "lambda_phys": 0.0,
            "lambda_smooth": 0.0,
            "augmented": False,
        },
        {
            "row_id": 2,
            "name": "+ Adaptive Graph (Contrib c)",
            "type": "gnn",
            "model_kind": "AdaptiveGCN",
            "use_adaptive": True,
            "lambda_phys": 0.0,
            "lambda_smooth": 0.0,
            "augmented": False,
        },
        {
            "row_id": 3,
            "name": "+ Physics Loss (Contrib a)",
            "type": "gnn",
            "model_kind": "GCN",
            "use_adaptive": False,
            "lambda_phys": 1.0,
            "lambda_smooth": 0.0,
            "augmented": False,
        },
        {
            "row_id": 4,
            "name": "+ Data Augmentation (Contrib b)",
            "type": "gnn",
            "model_kind": "GCN",
            "use_adaptive": False,
            "lambda_phys": 0.0,
            "lambda_smooth": 0.0,
            "augmented": True,
        },
        {
            "row_id": 5,
            "name": "All Three Combined (PIAG-Net+Aug)",
            "type": "gnn",
            "model_kind": "AdaptiveGCN",
            "use_adaptive": True,
            "lambda_phys": 0.10,
            "lambda_smooth": 0.01,
            "augmented": True,
        },
    ]

    master_results = []

    for cfg_desc in ablation_configs:
        row_id = cfg_desc["row_id"]
        name = cfg_desc["name"]
        cfg_type = cfg_desc["type"]

        print(f"\n--- Evaluating Row {row_id}: {name} ---")

        all_fold_rmse = []
        all_fold_mae = []
        all_fold_smape = []
        all_fold_moran = []
        all_fold_pte = []

        for s in splits:
            fold_idx = s["fold"]
            train_ids = s["train_ids"]
            val_ids = s["val_ids"]
            test_ids = s["test_ids"]

            tr_data, _ = build_fold_tensors(raw, train_ids, train_ids, window=3, horizon=3, log_transform=True)
            va_data, _ = build_fold_tensors(raw, train_ids, val_ids, window=3, horizon=3, log_transform=True)
            te_data, _ = build_fold_tensors(raw, train_ids, test_ids, window=3, horizon=3, log_transform=True)

            if cfg_type == "persistence":
                p_pred, p_truth = get_persistence_forecast(raw, test_ids, horizon=3, cases_idx=5)
                sc = metrics(p_pred, p_truth)["overall"]
                p_resids = p_pred - p_truth
                moran_i = compute_morans_i(p_resids, dense_adj_np)
                pte = compute_peak_timing_error(p_pred, p_truth)

                all_fold_rmse.append(sc["RMSE"])
                all_fold_mae.append(sc["MAE"])
                all_fold_smape.append(sc["SMAPE"])
                all_fold_moran.append(moran_i)
                all_fold_pte.append(pte)
            else:
                train_input = tr_data
                if cfg_desc["augmented"]:
                    train_input = heuristic_augmentation(tr_data, noise_std=0.03, scale_range=(0.95, 1.05))

                seed_rmse, seed_mae, seed_smape, seed_moran, seed_pte = [], [], [], [], []

                for seed in seeds:
                    set_seed(seed)
                    t_cfg = TrainConfig(
                        model_kind=cfg_desc["model_kind"],
                        use_adaptive=cfg_desc["use_adaptive"],
                        use_fixed=True,
                        hidden=64,
                        dropout=0.1,
                        lr=1e-3,
                        weight_decay=5e-4,
                        epochs=epochs,
                        patience=patience,
                        grad_clip=5.0,
                        residual=True,
                        log_transform=True,
                        lambda_phys=cfg_desc["lambda_phys"],
                        lambda_smooth=cfg_desc["lambda_smooth"],
                    )

                    if cfg_desc["use_adaptive"]:
                        fixed_adj = edge_index_to_dense_adj(edge_index, n_nodes=25)
                        model = create_adaptive_model(
                            in_dim=in_dim, hidden=64, horizon=3, n_nodes=25,
                            use_adaptive=True, use_fixed=True, fixed_adj=fixed_adj,
                        )
                    else:
                        model = create_baseline_model(in_dim=in_dim, hidden=64, horizon=3, kind="GCN")

                    trained_model, _ = train_fold(model, train_input, va_data, edge_index, t_cfg, verbose=False)
                    sc, m_i, pte_val = evaluate_predictions_full(
                        trained_model, te_data, edge_index, dense_adj_np, residual=True, device=t_cfg.device,
                    )

                    seed_rmse.append(sc["RMSE"])
                    seed_mae.append(sc["MAE"])
                    seed_smape.append(sc["SMAPE"])
                    seed_moran.append(m_i)
                    seed_pte.append(pte_val)

                all_fold_rmse.append(float(np.mean(seed_rmse)))
                all_fold_mae.append(float(np.mean(seed_mae)))
                all_fold_smape.append(float(np.mean(seed_smape)))
                all_fold_moran.append(float(np.mean(seed_moran)))
                all_fold_pte.append(float(np.mean(seed_pte)))

        res_row = {
            "Row": row_id,
            "Configuration": name,
            "RMSE": float(np.mean(all_fold_rmse)),
            "MAE": float(np.mean(all_fold_mae)),
            "SMAPE": float(np.mean(all_fold_smape)),
            "Moran_I": float(np.mean(all_fold_moran)),
            "Peak_Timing_Err_wks": float(np.mean(all_fold_pte)),
        }
        master_results.append(res_row)
        print(
            f"  Result: RMSE = {res_row['RMSE']:.2f} | MAE = {res_row['MAE']:.2f} | "
            f"SMAPE = {res_row['SMAPE']:.1f}% | Moran's I = {res_row['Moran_I']:.3f} | PTE = {res_row['Peak_Timing_Err_wks']:.2f} wks"
        )

    df_master = pd.DataFrame(master_results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_master.to_csv(args.out, index=False)
    print(f"\nSaved Master Ablation Table to: {args.out}")

    print("\n" + "=" * 75)
    print("MASTER ABLATION BENCHMARK TABLE (DELIVERABLE §5)")
    print("=" * 75)
    print(df_master.to_string(index=False))

    print("\n" + "=" * 75)
    print("LATEX TABLE FOR REPORT / PAPER WRITE-UP")
    print("=" * 75)
    latex_str = df_master.to_latex(
        index=False,
        float_format="%.2f",
        caption="Ablation of model contributions under rolling-origin CV (3 origins x 3 seeds).",
        label="tab:master_ablation",
    )
    print(latex_str)


if __name__ == "__main__":
    main()
