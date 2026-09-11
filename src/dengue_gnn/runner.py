"""Training, evaluation, and rolling-origin cross-validation runners.

Implements the frozen rolling-origin cross-validation protocol:
- 3 chronological origins (0.55, 0.70, 0.85)
- Expanding training pool with 30-week validation slice
- Multi-seed averaging
- Residual-over-persistence reconstruction and original-scale evaluation
- Supports both static baseline GNNs and Phase 2 Adaptive Graph models
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dengue_gnn.data import (
    FoldData,
    build_fold_tensors,
    get_persistence_forecast,
    get_rolling_origin_splits,
)
from dengue_gnn.losses import PhysicsInformedLoss, compute_normalized_laplacian
from dengue_gnn.metrics import metrics
from dengue_gnn.models import (
    AdaptiveGCN,
    GNNBaseline,
    create_adaptive_model,
    create_baseline_model,
    edge_index_to_dense_adj,
)


@dataclass
class TrainConfig:
    """Hyperparameter and runtime configuration."""
    model_kind: str = "GCN"  # "GCN", "GAT", or "AdaptiveGCN"
    use_adaptive: bool = False
    use_fixed: bool = True
    emb_dim: int = 10
    n_nodes: int = 25
    hidden: int = 64
    gat_heads: int = 8
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 120
    patience: int = 25
    grad_clip: float = 5.0
    residual: bool = True
    log_transform: bool = True
    window: int = 3
    horizon: int = 3
    cases_idx: int = 5
    use_all_feats: bool = True
    lambda_smooth: float = 0.0
    lambda_phys: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_fold(
    model: nn.Module,
    eval_data: FoldData,
    edge_index: torch.Tensor,
    residual: bool = True,
    device: torch.device | str = "cpu",
) -> dict[str, dict[str, float]]:
    """Evaluate a trained model on a fold, returning metrics on the raw case count scale.

    Args:
        model: Trained GNN model (GNNBaseline or AdaptiveGCN).
        eval_data: FoldData object with evaluation tensors and inverse mapping.
        edge_index: Adjacency edge index tensor.
        residual: Whether predictions are residual-over-persistence.
        device: PyTorch device.

    Returns:
        Dictionary of metrics per horizon ("h1", "h2", "h3") and "overall".
    """
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
    return metrics(pred_arr, truth_arr)


def train_fold(
    model: nn.Module,
    train_data: FoldData,
    val_data: FoldData,
    edge_index: torch.Tensor,
    config: TrainConfig,
    verbose: bool = False,
) -> tuple[nn.Module, float]:
    """Train a model on a single fold with early stopping on validation RMSE.

    Args:
        model: GNN model instance.
        train_data: FoldData for training.
        val_data: FoldData for validation.
        edge_index: Adjacency edge index tensor.
        config: Training configuration dataclass.
        verbose: Whether to print periodic training progress.

    Returns:
        model: Best model with restored weights.
        best_val_rmse: Lowest validation overall RMSE achieved.
    """
    dev = torch.device(config.device)
    model = model.to(dev)
    ei = edge_index.to(dev)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    loss_fn = PhysicsInformedLoss(
        lambda_smooth=config.lambda_smooth,
        lambda_phys=config.lambda_phys,
    )
    laplacian = None
    if config.lambda_smooth > 0.0:
        adj = edge_index_to_dense_adj(edge_index, n_nodes=config.n_nodes, self_loops=True)
        laplacian = compute_normalized_laplacian(adj).to(dev)

    Xtr, Ytr, Ptr = train_data.X, train_data.Y, train_data.P

    best_val_rmse = float("inf")
    best_state = None
    wait = 0

    for epoch in range(config.epochs):
        model.train()
        perm = torch.randperm(Xtr.shape[0])

        for i in perm:
            optimizer.zero_grad()
            pred = model(Xtr[i].to(dev), ei)
            target = (Ytr[i] - Ptr[i]) if config.residual else Ytr[i]
            loss_dict = loss_fn(pred, target.to(dev), laplacian=laplacian)
            loss = loss_dict["loss"]
            loss.backward()

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()

        # Validation step
        val_scores = evaluate_fold(model, val_data, edge_index, residual=config.residual, device=dev)
        val_rmse = val_scores["overall"]["RMSE"]

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (epoch % 20 == 0 or epoch == config.epochs - 1):
            print(f"  epoch {epoch:3d} | val RMSE: {val_rmse:.2f} | best: {best_val_rmse:.2f}")

        if wait >= config.patience:
            if verbose:
                print(f"  early stopped at epoch {epoch} (patience {config.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_rmse


def build_model_from_config(config: TrainConfig, in_dim: int, edge_index: torch.Tensor) -> nn.Module:
    """Instantiate appropriate model based on configuration."""
    if config.use_adaptive or config.model_kind.upper() == "ADAPTIVEGCN":
        fixed_adj = None
        if config.use_fixed:
            fixed_adj = edge_index_to_dense_adj(
                edge_index, n_nodes=config.n_nodes, self_loops=True, row_normalize=True
            )
        return create_adaptive_model(
            in_dim=in_dim,
            hidden=config.hidden,
            horizon=config.horizon,
            n_nodes=config.n_nodes,
            emb_dim=config.emb_dim,
            use_adaptive=True,
            use_fixed=config.use_fixed,
            fixed_adj=fixed_adj,
            dropout=config.dropout,
        )
    else:
        return create_baseline_model(
            in_dim=in_dim,
            hidden=config.hidden,
            horizon=config.horizon,
            kind=config.model_kind,
            heads=config.gat_heads,
            dropout=config.dropout,
        )


def run_rolling_origin_cv(
    raw: np.ndarray,
    edge_index: torch.Tensor,
    config: TrainConfig,
    origins: Sequence[float] = (0.55, 0.70, 0.85),
    test_frac: float = 0.15,
    val_weeks: int = 30,
    seeds: Sequence[int] = (0, 1, 2),
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run full rolling-origin cross validation across folds and random seeds."""
    splits = get_rolling_origin_splits(
        total_timesteps=raw.shape[0],
        window=config.window,
        horizon=config.horizon,
        origins=origins,
        test_frac=test_frac,
        val_weeks=val_weeks,
    )

    T, N, Fd = raw.shape
    in_dim = (Fd * config.window) if config.use_all_feats else config.window

    fold_records = []
    model_name = (
        "AdaptiveGCN"
        if (config.use_adaptive or config.model_kind.upper() == "ADAPTIVEGCN")
        else config.model_kind
    )
    if model_name == "AdaptiveGCN" and not config.use_fixed:
        model_name = "PureAdaptiveGCN"

    for s in splits:
        fold_idx = s["fold"]
        origin = s["origin"]
        train_ids = s["train_ids"]
        val_ids = s["val_ids"]
        test_ids = s["test_ids"]

        # Build folds with train-only normalization
        tr_data, _ = build_fold_tensors(
            raw, train_ids, train_ids,
            window=config.window, horizon=config.horizon,
            cases_idx=config.cases_idx, use_all_feats=config.use_all_feats,
            log_transform=config.log_transform,
        )
        va_data, _ = build_fold_tensors(
            raw, train_ids, val_ids,
            window=config.window, horizon=config.horizon,
            cases_idx=config.cases_idx, use_all_feats=config.use_all_feats,
            log_transform=config.log_transform,
        )
        te_data, _ = build_fold_tensors(
            raw, train_ids, test_ids,
            window=config.window, horizon=config.horizon,
            cases_idx=config.cases_idx, use_all_feats=config.use_all_feats,
            log_transform=config.log_transform,
        )

        # Baseline: Naive Persistence on test slice
        p_pred, p_truth = get_persistence_forecast(
            raw, test_ids, horizon=config.horizon, cases_idx=config.cases_idx,
        )
        persist_metrics = metrics(p_pred, p_truth)["overall"]

        rmse_runs, mae_runs, smape_runs, mape_runs = [], [], [], []
        gate_weights = []

        for seed in seeds:
            set_seed(seed)
            model = build_model_from_config(config, in_dim=in_dim, edge_index=edge_index)
            trained_model, _ = train_fold(model, tr_data, va_data, edge_index, config, verbose=False)
            scores = evaluate_fold(trained_model, te_data, edge_index, residual=config.residual, device=config.device)
            overall = scores["overall"]

            rmse_runs.append(overall["RMSE"])
            mae_runs.append(overall["MAE"])
            smape_runs.append(overall["SMAPE"])
            mape_runs.append(overall["MAPE"])

            if isinstance(trained_model, AdaptiveGCN):
                gate_weights.append(trained_model.get_gate_weight())

        record = {
            "fold": fold_idx,
            "origin": origin,
            "n_test": len(test_ids),
            "model": model_name,
            "model_RMSE": float(np.mean(rmse_runs)),
            "model_RMSE_std": float(np.std(rmse_runs)),
            "model_MAE": float(np.mean(mae_runs)),
            "model_SMAPE": float(np.mean(smape_runs)),
            "persist_RMSE": persist_metrics["RMSE"],
            "persist_MAE": persist_metrics["MAE"],
            "persist_SMAPE": persist_metrics["SMAPE"],
            "gate_weight": float(np.mean(gate_weights)) if gate_weights else 1.0,
        }
        fold_records.append(record)

        if verbose:
            gate_str = f" | gate sigma(g): {record['gate_weight']:.3f}" if gate_weights else ""
            print(
                f"Fold {fold_idx} (origin {origin:.2f}, {len(test_ids)} test wks) | "
                f"{model_name} RMSE: {record['model_RMSE']:.2f} +/- {record['model_RMSE_std']:.2f}, "
                f"MAE: {record['model_MAE']:.2f}{gate_str} | "
                f"Persistence RMSE: {record['persist_RMSE']:.2f}, MAE: {record['persist_MAE']:.2f}"
            )

    df_results = pd.DataFrame(fold_records)
    summary = {
        "model": model_name,
        "mean_model_RMSE": float(df_results["model_RMSE"].mean()),
        "mean_model_MAE": float(df_results["model_MAE"].mean()),
        "mean_model_SMAPE": float(df_results["model_SMAPE"].mean()),
        "mean_persist_RMSE": float(df_results["persist_RMSE"].mean()),
        "mean_persist_MAE": float(df_results["persist_MAE"].mean()),
        "mean_persist_SMAPE": float(df_results["persist_SMAPE"].mean()),
        "mean_gate_weight": float(df_results["gate_weight"].mean()),
    }

    if verbose:
        print(f"\n=== {model_name} Rolling-Origin CV Summary ===")
        gate_info = f" | Mean Gate sigma(g): {summary['mean_gate_weight']:.3f}" if config.use_adaptive else ""
        print(
            f"Mean Model RMSE: {summary['mean_model_RMSE']:.2f} | MAE: {summary['mean_model_MAE']:.2f} | SMAPE: {summary['mean_model_SMAPE']:.1f}%{gate_info}\n"
            f"Persistence RMSE: {summary['mean_persist_RMSE']:.2f} | MAE: {summary['mean_persist_MAE']:.2f} | SMAPE: {summary['mean_persist_SMAPE']:.1f}%"
        )

    return df_results, summary
