"""Forecast metrics, on the original case scale.

Extracted verbatim (numerically) from the ``metrics`` function in
``notebooks/baseline/dengue_baseline_GNN_v2.ipynb`` section 6, so that every phase
of the project is scored by one implementation instead of a copy per notebook.

Why these four:

* ``RMSE`` / ``MAE`` -- reported on raw counts, comparable to Weng et al. (2024).
* ``SMAPE`` -- bounded and symmetric; the primary percentage metric, because 9.7%
  of district-weeks are zero and plain MAPE explodes on them.
* ``MAPE`` -- computed only over weeks with ``truth >= 1`` (zero-masked), kept for
  comparability with papers that report it. ``nan`` when no week qualifies.

Predictions and targets must already be inverse-transformed out of log1p /
residual space before they get here: these are raw-count metrics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["MAPE_FLOOR", "SMAPE_EPS", "metrics", "score"]

#: Denominator stabilizer in SMAPE, matching the notebook implementation.
SMAPE_EPS = 1e-6

#: Weeks with ``truth`` below this are excluded from MAPE.
MAPE_FLOOR = 1.0


def score(pred: ArrayLike, truth: ArrayLike) -> dict[str, float]:
    """Return RMSE, MAE, SMAPE and zero-masked MAPE for one set of forecasts.

    Args:
        pred: Predicted case counts, any shape.
        truth: Observed case counts, same shape as ``pred``.

    Returns:
        Mapping with keys ``RMSE``, ``MAE``, ``SMAPE``, ``MAPE``. SMAPE and MAPE
        are percentages. ``MAPE`` is ``nan`` when every observation is below
        ``MAPE_FLOOR``.

    Raises:
        ValueError: If ``pred`` and ``truth`` have different shapes, or are empty.
    """
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)

    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs truth {y.shape}")
    if p.size == 0:
        raise ValueError("cannot score empty arrays")

    err = p - y
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    smape = float(np.mean(2 * np.abs(err) / (np.abs(p) + np.abs(y) + SMAPE_EPS)) * 100)

    mask = y >= MAPE_FLOOR
    mape = float(np.mean(np.abs(err[mask]) / y[mask]) * 100) if mask.any() else float("nan")

    return {"RMSE": rmse, "MAE": mae, "SMAPE": smape, "MAPE": mape}


def metrics(pred: ArrayLike, truth: ArrayLike) -> dict[str, dict[str, float]]:
    """Score multi-horizon forecasts per horizon and overall.

    The last axis is the forecast horizon, so ``(n_windows, n_nodes, H)`` yields
    ``h1 .. hH`` plus ``overall``. Per-horizon reporting is required by the
    evaluation protocol -- a model can look fine on average while degrading
    sharply at 3 weeks ahead, which is the horizon that matters operationally.

    Args:
        pred: Predicted case counts, horizon on the last axis.
        truth: Observed case counts, same shape as ``pred``.

    Returns:
        Mapping from ``"h1"``..``"hH"`` and ``"overall"`` to the dict returned by
        :func:`score`.
    """
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)

    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs truth {y.shape}")
    if p.ndim == 0:
        raise ValueError("expected at least one axis, with horizon last")

    out = {f"h{h + 1}": score(p[..., h], y[..., h]) for h in range(p.shape[-1])}
    out["overall"] = score(p, y)
    return out
