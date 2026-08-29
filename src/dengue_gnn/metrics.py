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

__all__ = ["MAPE_FLOOR", "SMAPE_EPS", "metrics", "score", "smape", "peak_timing_error"]

#: Denominator stabilizer in SMAPE, matching the notebook implementation.
SMAPE_EPS = 1e-6

#: Weeks with ``truth`` below this are excluded from MAPE.
MAPE_FLOOR = 1.0


def smape(pred: ArrayLike, truth: ArrayLike) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE)."""
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs truth {y.shape}")
    if p.size == 0:
        raise ValueError("cannot score empty arrays")
    err = p - y
    return float(np.mean(2 * np.abs(err) / (np.abs(p) + np.abs(y) + SMAPE_EPS)) * 100)


def peak_timing_error(pred: ArrayLike, truth: ArrayLike) -> float:
    """Calculate the mean peak timing error (in time steps) between predicted and true series.

    For 1D arrays or ND arrays with the time dimension along the last axis,
    computes the absolute difference in peak (argmax) index between pred and truth.
    """
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs truth {y.shape}")
    if p.size == 0:
        raise ValueError("cannot score empty arrays")

    if p.ndim == 1:
        peak_p = int(np.argmax(p))
        peak_y = int(np.argmax(y))
        return float(abs(peak_p - peak_y))
    else:
        peak_p = np.argmax(p, axis=-1)
        peak_y = np.argmax(y, axis=-1)
        return float(np.mean(np.abs(peak_p - peak_y)))


def score(pred: ArrayLike, truth: ArrayLike) -> dict[str, float]:
    """Return RMSE, MAE, SMAPE, zero-masked MAPE, and PeakTimingErr for forecasts.

    Args:
        pred: Predicted case counts, any shape.
        truth: Observed case counts, same shape as ``pred``.

    Returns:
        Mapping with keys ``RMSE``, ``MAE``, ``SMAPE``, ``MAPE``, ``PeakTimingErr``.
        SMAPE and MAPE are percentages. ``MAPE`` is ``nan`` when every observation is
        below ``MAPE_FLOOR``.

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
    smape_val = smape(p, y)

    mask = y >= MAPE_FLOOR
    mape = float(np.mean(np.abs(err[mask]) / y[mask]) * 100) if mask.any() else float("nan")
    pte = peak_timing_error(p, y)

    return {"RMSE": rmse, "MAE": mae, "SMAPE": smape_val, "MAPE": mape, "PeakTimingErr": pte}


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
