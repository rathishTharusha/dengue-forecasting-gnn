"""Leakage-controlled harness for the SEIR-GNN re-run.

Everything the re-run reports comes through this module. It exists because the
original ``analysis/_build/run_s5_seir_gnn.py`` mixed four separate defects --
one gradient step per epoch, a SMAPE objective scored by RMSE, a force of
infection held constant across the horizon, and all input channels collapsed to
a single scalar before the backbone -- so its numbers say nothing about whether
physics-informed graph networks work on this series.

Protocol is the project's frozen one, unchanged: rolling origins 0.55/0.70/0.85,
window 3 -> horizon 3, normalisation from training weeks only, no shuffling.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
import corrected_data as cd  # noqa: E402

WINDOW, HORIZON = 3, 3
ORIGINS = (0.55, 0.70, 0.85)


@dataclass
class Fold:
    origin: float
    idx: dict[str, np.ndarray]
    mean: float
    std: float


def build_folds(cases: np.ndarray, missing: np.ndarray) -> list[Fold]:
    """The frozen protocol's split boundaries, computed on the full window list.

    Boundaries come from ``len(ids)`` before any missing-week filtering, so a
    gap never shifts a fold; windows touching a gap are then dropped.
    """
    bad = {int(m) for m in np.where(missing)[0]}
    ids = list(range(WINDOW, cases.shape[0] - HORIZON))

    def clean(i: int) -> bool:
        return not any(t in bad for t in range(i - WINDOW, i + HORIZON))

    folds = []
    for origin in ORIGINS:
        cut = int(origin * len(ids))
        end = int(min(origin + 0.15, 1.0) * len(ids))
        tr = [i for i in ids[: cut - 30] if clean(i)]
        va = [i for i in ids[cut - 30 : cut] if clean(i)]
        te = [i for i in ids[cut:end] if clean(i)]
        history = np.log1p(cases[: ids[: cut - 30][-1] + 1])
        mean = float(np.nanmean(history))
        std = float(np.nanstd(history) + 1e-8)
        folds.append(Fold(origin, {"train": np.array(tr), "val": np.array(va),
                                   "test": np.array(te)}, mean, std))
    return folds


def seasonal_features(week_start, idx: np.ndarray) -> np.ndarray:
    """``(K, 4)`` sin/cos of week-of-year at 1 and 2 cycles per year.

    EDA finding F9: the series is bimodally seasonal with a 2.9x peak-to-trough
    ratio and no model in this project has ever seen week-of-year. A 3-week
    window cannot represent a 52-week cycle, so it is supplied directly.
    """
    doy = week_start.dt.dayofyear.to_numpy()[idx]
    ang = 2 * np.pi * doy / 365.25
    return np.stack([np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)], -1)


def build_tensors(data: cd.CorrectedData, fold: Fold, split: str, use_climate: bool,
                  use_ndvi: bool, use_season: bool) -> dict[str, torch.Tensor]:
    """Slice one split under ``cd.LAGS`` and z-score with this fold's statistics.

    Case history and target are returned both as raw counts and in fold-scaled
    log1p space; ``p`` is last-observed-week carried across the horizon, the
    anchor every residual model predicts a correction to.
    """
    idx = fold.idx[split]
    cases = data.cases
    x_raw = np.stack([cases[i - WINDOW : i].T for i in idx])          # (K,N,W)
    y_raw = np.stack([cases[i : i + HORIZON].T for i in idx])         # (K,N,H)
    p_raw = np.repeat(cases[idx - 1][:, :, None], HORIZON, axis=2)    # (K,N,H)

    z = lambda a: (np.log1p(a) - fold.mean) / fold.std  # noqa: E731
    feats = [z(x_raw)]

    if use_climate:
        cl = np.stack([np.moveaxis(data.climate[i - cd.LAGS["climate"] - 2 : i - cd.LAGS["climate"] + 1],
                                   0, 1) for i in idx])               # (K,N,3,6)
        tr = fold.idx["train"]
        ref = np.stack([np.moveaxis(data.climate[i - cd.LAGS["climate"] - 2 : i - cd.LAGS["climate"] + 1],
                                    0, 1) for i in tr])
        m, s = ref.mean((0, 1, 2)), ref.std((0, 1, 2)) + 1e-8
        feats.append(((cl - m) / s).reshape(len(idx), cl.shape[1], -1))

    if use_ndvi:
        nd = np.stack([data.ndvi[i - 2 : i + 1].T for i in idx])       # (K,N,3)
        ref = np.stack([data.ndvi[i - 2 : i + 1].T for i in fold.idx["train"]])
        feats.append((nd - ref.mean()) / (ref.std() + 1e-8))

    if use_season:
        se = seasonal_features(data.week_start, idx)                   # (K,4)
        feats.append(np.repeat(se[:, None, :], x_raw.shape[1], axis=1))

    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    return {"x": t(np.concatenate(feats, axis=-1)), "y_raw": t(y_raw), "p_raw": t(p_raw),
            "y_z": t(z(y_raw)), "p_z": t(z(p_raw)), "x_raw": t(x_raw),
            "pop": t(data.population[idx - 1]), "idx": idx}


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - truth)))


def score(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """RMSE/MAE overall and per horizon step, on raw counts."""
    out = {"RMSE": rmse(pred, truth), "MAE": mae(pred, truth)}
    for h in range(truth.shape[-1]):
        out[f"RMSE_h{h + 1}"] = rmse(pred[..., h], truth[..., h])
    return out


def adjacency(names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """``(edge_index, row-normalised dense adjacency with self-loops)``."""
    import json
    adj = json.loads((REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json").read_text())
    n = len(names)
    pos = {d: k for k, d in enumerate(names)}
    a = np.eye(n, dtype=np.float32)
    for d, nb in adj.items():
        for o in nb:
            if o in pos:
                a[pos[d], pos[o]] = 1.0
    src, dst = np.nonzero(a)
    return (torch.tensor(np.stack([src, dst]), dtype=torch.long),
            torch.tensor(a / a.sum(1, keepdims=True), dtype=torch.float32))


class EarlyStop:
    """Track the best validation RMSE and hold a copy of the weights that got it."""

    def __init__(self, model: nn.Module, patience: int = 40):
        self.model, self.patience = model, patience
        self.best, self.best_state, self.waited = float("inf"), None, 0

    def step(self, value: float) -> bool:
        if value < self.best - 1e-6:
            self.best, self.waited = value, 0
            self.best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            return False
        self.waited += 1
        return self.waited >= self.patience

    def restore(self) -> None:
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
