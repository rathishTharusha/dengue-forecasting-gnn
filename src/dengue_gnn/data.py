"""Data loading, pre-processing, and rolling-origin split utilities.

Follows the protocol established in notebooks/baseline/dengue_baseline_GNN_v2.ipynb:
- 3-week window -> 3-week horizon
- Rolling-origin expanding window splits (0.55, 0.70, 0.85)
- Train-only normalization (preventing temporal data leakage)
- Log1p transformation and residual-over-persistence target representations
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch


@dataclass
class FoldData:
    """Pack of features, targets, and persistence baseline values for an evaluation split."""
    X: torch.Tensor  # shape: (n_samples, n_nodes, in_dim)
    Y: torch.Tensor  # shape: (n_samples, n_nodes, horizon)
    P: torch.Tensor  # shape: (n_samples, n_nodes, horizon)
    inv_fn: Callable[[np.ndarray | torch.Tensor], np.ndarray]


def load_adjacency(adj_path: str, n_nodes: int = 25, self_loops: bool = True) -> tuple[torch.Tensor, list[str]]:
    """Load district adjacency list JSON and convert to PyG edge_index tensor.

    Args:
        adj_path: Path to adjacency JSON file.
        n_nodes: Number of nodes / districts (default 25).
        self_loops: Whether to include self-loops on each node.

    Returns:
        edge_index: PyG edge tensor of shape (2, num_edges), dtype torch.long.
        district_names: Sorted list of district names matching node indices.
    """
    with open(adj_path, "r", encoding="utf-8") as f:
        adj = json.load(f)

    district_names = sorted(adj.keys())
    idx_map = {name: i for i, name in enumerate(district_names)}

    adj_mat = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    if self_loops:
        np.fill_diagonal(adj_mat, 1.0)

    for dist, neighbors in adj.items():
        for nb in neighbors:
            if nb in idx_map:
                adj_mat[idx_map[dist], idx_map[nb]] = 1.0

    edges = np.array([[i, j] for i in range(n_nodes) for j in range(n_nodes) if adj_mat[i, j]]).T
    edge_index = torch.tensor(edges, dtype=torch.long)
    return edge_index, district_names


def load_dataset(
    npy_path: str,
    adj_path: str,
    n_nodes: int = 25,
    self_loops: bool = True,
) -> tuple[np.ndarray, torch.Tensor, list[str]]:
    """Load the raw feature array and adjacency graph.

    Args:
        npy_path: Path to .npy file (T, N, Fd).
        adj_path: Path to adjacency JSON.
        n_nodes: Number of districts.
        self_loops: Whether to add self-loops to adjacency.

    Returns:
        raw: Array of shape (T, N, Fd), float32.
        edge_index: PyG edge index tensor (2, num_edges).
        district_names: List of district names.
    """
    raw = np.nan_to_num(np.load(npy_path, allow_pickle=True)).astype(np.float32)
    edge_index, district_names = load_adjacency(adj_path, n_nodes=n_nodes, self_loops=self_loops)
    return raw, edge_index, district_names


def get_rolling_origin_splits(
    total_timesteps: int,
    window: int = 3,
    horizon: int = 3,
    origins: Sequence[float] = (0.55, 0.70, 0.85),
    test_frac: float = 0.15,
    val_weeks: int = 30,
) -> list[dict]:
    """Generate chronological rolling-origin train, val, and test index sets.

    Args:
        total_timesteps: Total number of time steps (T).
        window: Input window length in weeks.
        horizon: Forecast horizon in weeks.
        origins: Chronological cutoff fractions for expanding window.
        test_frac: Test window fraction per fold.
        val_weeks: Number of validation weeks carved from the end of the train pool.

    Returns:
        List of dicts containing fold metadata and split index lists.
    """
    valid_ids = list(range(window, total_timesteps - horizon))
    n = len(valid_ids)

    splits = []
    for k, a in enumerate(origins):
        cut = int(a * n)
        tend = int(min(a + test_frac, 1.0) * n)

        train_pool = valid_ids[:cut]
        test_ids = valid_ids[cut:tend]

        if len(test_ids) == 0:
            continue

        train_ids = train_pool[:-val_weeks]
        val_ids = train_pool[-val_weeks:]

        splits.append({
            "fold": k + 1,
            "origin": a,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
        })

    return splits


def build_fold_tensors(
    raw: np.ndarray,
    train_ids: list[int],
    eval_ids: list[int],
    window: int = 3,
    horizon: int = 3,
    cases_idx: int = 5,
    use_all_feats: bool = True,
    log_transform: bool = True,
) -> tuple[FoldData, Callable[[np.ndarray | torch.Tensor], np.ndarray]]:
    """Build normalized input features X, normalized target Y, and persistence reference P.

    Crucially, normalisation statistics (mean and std) are fitted exclusively on the
    training pool (`raw[:train_ids[-1]]`), guaranteeing zero lookahead leakage into
    validation or test evaluation sets.

    Args:
        raw: Full data array (T, N, Fd).
        train_ids: Week indices for training.
        eval_ids: Week indices for evaluation (train, val, or test).
        window: Window size (weeks).
        horizon: Forecast horizon (weeks).
        cases_idx: Channel index of dengue cases.
        use_all_feats: Whether to flatten all feature channels or use target cases only.
        log_transform: Whether to model targets in log1p space.

    Returns:
        FoldData containing X, Y, P tensors and the inverse transformation function.
    """
    T, N, Fd = raw.shape
    last_train = train_ids[-1]
    tw = raw[:last_train]  # weeks strictly before eval region

    # Feature standardization from training data only
    fmean = tw.reshape(-1, Fd).mean(0)
    fstd = tw.reshape(-1, Fd).std(0) + 1e-6
    feat = (raw - fmean) / fstd

    cases = raw[..., cases_idx]

    if log_transform:
        base = np.log1p(cases)
        tbase = np.log1p(tw[..., cases_idx])
        tm = float(tbase.mean())
        ts = float(tbase.std() + 1e-6)

        def inv_fn(x: np.ndarray | torch.Tensor) -> np.ndarray:
            arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            return np.expm1(np.clip(arr * ts + tm, 0, 12))
    else:
        tm = float(tw[..., cases_idx].mean())
        ts = float(tw[..., cases_idx].std() + 1e-6)
        base = cases

        def inv_fn(x: np.ndarray | torch.Tensor) -> np.ndarray:
            arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            return arr * ts + tm

    tgt = (base - tm) / ts

    X_list, Y_list, P_list = [], [], []
    for i in eval_ids:
        if use_all_feats:
            xi = np.transpose(feat[i - window : i], (1, 2, 0)).reshape(N, Fd * window)
        else:
            xi = tgt[i - window : i].T

        yi = tgt[i : i + horizon].T
        pi = tgt[i - 1].reshape(N, 1).repeat(horizon, axis=1)

        X_list.append(xi)
        Y_list.append(yi)
        P_list.append(pi)

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y_list), dtype=torch.float32)
    P = torch.tensor(np.stack(P_list), dtype=torch.float32)

    return FoldData(X=X, Y=Y, P=P, inv_fn=inv_fn), inv_fn


def get_persistence_forecast(
    raw: np.ndarray,
    eval_ids: list[int],
    horizon: int = 3,
    cases_idx: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate naive persistence forecasts and true observations on the raw case scale.

    Persistence repeats the most recent observed case count (week i-1) across all
    forecast horizons h=1..horizon.

    Args:
        raw: Full data array (T, N, Fd).
        eval_ids: List of evaluation week indices.
        horizon: Forecast horizon length.
        cases_idx: Channel index of dengue cases.

    Returns:
        preds: Array of shape (len(eval_ids), N, horizon).
        truths: Array of shape (len(eval_ids), N, horizon).
    """
    preds = np.stack([
        np.repeat(raw[i - 1, :, cases_idx : cases_idx + 1], horizon, axis=1)
        for i in eval_ids
    ])
    truths = np.stack([
        raw[i : i + horizon, :, cases_idx].T
        for i in eval_ids
    ])
    return preds, truths
