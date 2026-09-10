"""A spatio-temporal GNN in which the adjacency is the only variable.

The point of this module is a controlled comparison. Four graph modes share one
encoder, one training loop, one normalisation and one protocol, so a difference
between them is attributable to the graph and nothing else:

``none``
    Identity adjacency -- 25 independent time series, no message passing. The
    control that says whether a graph earns its keep at all.
``fixed``
    The authors' hand-built district adjacency, symmetrically normalised. What
    every model in Weng et al. uses.
``adaptive``
    ``softmax(ReLU(E1 @ E2.T))`` with ``E1``, ``E2`` learned node embeddings --
    the Graph WaveNet construction ``docs/ROADMAP.md`` cites for Contribution (c).
``hybrid``
    ``0.5 * fixed + 0.5 * adaptive``, which is how Graph WaveNet actually uses it:
    the learned graph augments the prior rather than replacing it.

Modelling choices other than the graph follow ``docs/decisions/0001``: predict
the residual over persistence, in log1p space, with normalisation from training
weeks only. Those are the choices that made this project's own baseline
competitive, and holding them fixed keeps the comparison about the graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

__all__ = [
    "GRAPH_MODES",
    "STGNN",
    "Fold",
    "build_folds",
    "evaluate",
    "load_dataset",
    "persistence_scores",
    "pooled_scores",
    "train_one",
]

GRAPH_MODES = ("none", "fixed", "adaptive", "hybrid")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_dataset(npy_path: str | Path, adj_path: str | Path, cases_idx: int = 5):
    """Return ``(cases, adjacency, district_names)``.

    ``cases`` is ``(weeks, districts)`` raw counts; ``adjacency`` is the
    symmetrically normalised binary graph with self-loops.
    """
    raw = np.nan_to_num(np.load(Path(npy_path), allow_pickle=True)).astype(np.float64)
    cases = raw[..., cases_idx]

    adj = json.loads(Path(adj_path).read_text(encoding="utf-8"))
    names = sorted(adj)
    index = {name: i for i, name in enumerate(names)}
    n = len(names)

    a = np.eye(n)
    for district, neighbours in adj.items():
        for neighbour in neighbours:
            a[index[district], index[neighbour]] = 1.0
    # Symmetrise: the released list has two one-way edges (EDA F7), and an
    # adjacency relation that is not symmetric is a data defect, not a modelling
    # choice. Both arms get the same corrected graph.
    a = np.maximum(a, a.T)

    deg = a.sum(1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    return cases, d_inv_sqrt @ a @ d_inv_sqrt, names


@dataclass
class Fold:
    """One rolling-origin fold, with train-only normalisation baked in."""

    origin: float
    x_train: torch.Tensor
    y_train: torch.Tensor
    p_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    p_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    p_test: torch.Tensor
    mean: float
    std: float
    test_index: np.ndarray = field(default_factory=lambda: np.array([]))
    train_index: np.ndarray = field(default_factory=lambda: np.array([]))
    val_index: np.ndarray = field(default_factory=lambda: np.array([]))

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Undo log1p + z-score, returning raw counts."""
        return np.expm1(np.clip(values * self.std + self.mean, 0, 12))


def build_folds(
    cases: np.ndarray,
    window: int = 3,
    horizon: int = 3,
    origins: tuple[float, ...] = (0.55, 0.70, 0.85),
    test_frac: float = 0.15,
    val_weeks: int = 30,
) -> list[Fold]:
    """Rolling-origin folds, following ``docs/ROADMAP.md``'s frozen protocol.

    Targets are ``log1p`` z-scored with statistics from training weeks only, and
    the persistence term ``p`` is carried alongside so the model can predict a
    residual over it.
    """
    n_weeks = cases.shape[0]
    ids = list(range(window, n_weeks - horizon))
    folds: list[Fold] = []

    for origin in origins:
        cut = int(origin * len(ids))
        end = int(min(origin + test_frac, 1.0) * len(ids))
        if end <= cut:
            continue
        train_ids = ids[: cut - val_weeks]
        val_ids = ids[cut - val_weeks : cut]
        test_ids = ids[cut:end]

        train_weeks = np.log1p(cases[: train_ids[-1] + 1])
        mean, std = float(train_weeks.mean()), float(train_weeks.std() + 1e-8)
        z = (np.log1p(cases) - mean) / std

        def pack(chosen, scaled=z):
            # `scaled` is bound at definition time on purpose: closing over the
            # loop's `z` would silently use the last origin's scaling if this
            # were ever called outside the iteration that created it.
            x = np.stack([scaled[i - window : i].T for i in chosen])
            y = np.stack([scaled[i : i + horizon].T for i in chosen])
            p = np.stack([np.repeat(scaled[i - 1][:, None], horizon, axis=1) for i in chosen])
            to = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
            return to(x), to(y), to(p)

        xtr, ytr, ptr = pack(train_ids)
        xva, yva, pva = pack(val_ids)
        xte, yte, pte = pack(test_ids)
        folds.append(
            Fold(
                origin, xtr, ytr, ptr, xva, yva, pva, xte, yte, pte, mean, std,
                np.asarray(test_ids), np.asarray(train_ids), np.asarray(val_ids),
            )
        )
    return folds


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


class STGNN(nn.Module):
    """Two graph-convolution layers over a window, predicting a horizon.

    Args:
        n_nodes: Number of districts.
        window: Input weeks.
        horizon: Output weeks.
        graph_mode: One of :data:`GRAPH_MODES`.
        hidden: Width of the hidden representation.
        emb_dim: Node-embedding width for the learned adjacency.
        dropout: Dropout between the two graph layers.
        alpha: Weight on the fixed graph in ``hybrid`` mode.
    """

    def __init__(
        self,
        n_nodes: int,
        window: int = 3,
        horizon: int = 3,
        graph_mode: str = "fixed",
        hidden: int = 64,
        emb_dim: int = 16,
        dropout: float = 0.1,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if graph_mode not in GRAPH_MODES:
            raise ValueError(f"graph_mode must be one of {GRAPH_MODES}, got {graph_mode!r}")
        self.graph_mode = graph_mode
        self.alpha = alpha
        self.n_nodes = n_nodes

        self.lin_in = nn.Linear(window, hidden)
        self.gc1 = nn.Linear(hidden, hidden)
        self.gc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

        # Learned node embeddings. Only used by 'adaptive' and 'hybrid', but
        # always created so parameter initialisation consumes the same RNG draws
        # in every arm -- otherwise the arms differ by more than the graph.
        self.emb_src = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.emb_dst = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)

    def adjacency(self, fixed: torch.Tensor) -> torch.Tensor:
        """Return the adjacency this arm uses, shape ``(n_nodes, n_nodes)``."""
        if self.graph_mode == "none":
            return torch.eye(self.n_nodes, device=fixed.device)
        if self.graph_mode == "fixed":
            return fixed
        learned = torch.softmax(torch.relu(self.emb_src @ self.emb_dst.T), dim=-1)
        if self.graph_mode == "adaptive":
            return learned
        return self.alpha * fixed + (1.0 - self.alpha) * learned

    def forward(self, x: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        """``(batch, nodes, window)`` -> ``(batch, nodes, horizon)``."""
        a = self.adjacency(fixed)
        h = torch.relu(self.lin_in(x))
        h = torch.relu(a @ self.gc1(h))
        h = self.drop(h)
        h = torch.relu(a @ self.gc2(h))
        return self.head(h)


# --------------------------------------------------------------------------
# train / evaluate
# --------------------------------------------------------------------------


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    """Pooled RMSE on raw counts -- not the per-window mean Weng et al. report."""
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    """Pooled MAE on raw counts."""
    return float(np.mean(np.abs(pred - truth)))


def pooled_scores(
    pred: np.ndarray, truth: np.ndarray, artifact: np.ndarray | None = None
) -> dict:
    """Pooled RMSE/MAE, reported twice when an artifact mask is supplied.

    Args:
        pred: Predictions, ``(windows, nodes, horizon)`` raw counts.
        truth: Ground truth, same shape.
        artifact: Optional boolean mask over the window axis, ``True`` for
            windows contaminated by the week-395 reporting spike
            (:func:`improved.artifact_windows`).

    Returns:
        ``RMSE``/``MAE`` over all windows, plus ``RMSE_clean``/``MAE_clean`` over
        the artifact-free windows and the counts, when a mask is given.

    Both numbers are reported because neither alone is honest. Including week 395
    measures a reporting backlog -- on the origin-0.85 fold, 6 of 68 windows carry
    90% of the squared error. Excluding it silently would flatter every arm
    equally and hide that the fold's difficulty is one week of bad data.
    """
    out = {"RMSE": rmse(pred, truth), "MAE": mae(pred, truth)}
    if artifact is None:
        return out
    keep = ~np.asarray(artifact, dtype=bool)
    out["n_windows"] = int(keep.size)
    out["n_artifact"] = int(keep.size - keep.sum())
    out["RMSE_clean"] = rmse(pred[keep], truth[keep]) if keep.any() else float("nan")
    out["MAE_clean"] = mae(pred[keep], truth[keep]) if keep.any() else float("nan")
    return out


@torch.no_grad()
def evaluate(model: STGNN, fold: Fold, fixed: torch.Tensor, split: str = "test") -> dict:
    """Score one split of a fold on the raw case scale."""
    model.eval()
    x = getattr(fold, f"x_{split}")
    y = getattr(fold, f"y_{split}")
    p = getattr(fold, f"p_{split}")
    pred = model(x, fixed) + p  # residual over persistence
    pred_raw = fold.inverse(pred.numpy())
    truth_raw = fold.inverse(y.numpy())
    return pooled_scores(pred_raw, truth_raw)


def train_one(
    fold: Fold,
    fixed: torch.Tensor,
    graph_mode: str,
    seed: int = 0,
    epochs: int = 150,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    patience: int = 30,
    batch_size: int = 32,
    **model_kwargs,
) -> tuple[STGNN, dict]:
    """Train one arm on one fold, selecting on validation RMSE.

    Returns the best model and its held-out test scores. Early stopping is on a
    genuine validation split -- unlike Weng et al.'s loop, which keeps the final
    epoch's weights.
    """
    torch.manual_seed(seed)
    # Legacy global seed on purpose: it mirrors how evaluation.py seeds, and any
    # numpy randomness reached from here should follow the same seed.
    np.random.seed(seed)  # noqa: NPY002

    model = STGNN(fixed.shape[0], graph_mode=graph_mode, **model_kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    n = fold.x_train.shape[0]
    best_state, best_val, waited = None, float("inf"), 0

    for _ in range(epochs):
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            opt.zero_grad()
            pred = model(fold.x_train[idx], fixed)
            target = fold.y_train[idx] - fold.p_train[idx]  # residual
            loss = loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        val = evaluate(model, fold, fixed, "val")["RMSE"]
        if val < best_val - 1e-6:
            best_val, waited = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, evaluate(model, fold, fixed, "test")


def persistence_scores(
    fold: Fold, cases: np.ndarray, horizon: int = 3, artifact: np.ndarray | None = None
) -> dict:
    """Last-value-carried-forward on the same test windows -- the floor to beat.

    Pass ``artifact`` to get the floor reported with and without the week-395
    windows; the floor has to be split the same way the models are, or the
    comparison is between two different test sets.
    """
    pred = np.stack([np.repeat(cases[i - 1][:, None], horizon, axis=1) for i in fold.test_index])
    truth = np.stack([cases[i : i + horizon].T for i in fold.test_index])
    return pooled_scores(pred, truth, artifact)
