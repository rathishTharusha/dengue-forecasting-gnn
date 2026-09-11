"""Forecast metrics, on the original case scale.

Extracted verbatim (numerically) from the ``metrics`` function in
``notebooks/baseline/dengue_baseline_GNN_v2.ipynb`` section 6, with additions
for Moran's I spatial residual autocorrelation and Peak Timing Error.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "MAPE_FLOOR",
    "SMAPE_EPS",
    "metrics",
    "score",
    "compute_morans_i",
    "compute_peak_timing_error",
]

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
        Mapping with keys ``RMSE``, ``MAE``, ``SMAPE``, ``MAPE``.
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
    """Score multi-horizon forecasts per horizon and overall."""
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)

    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs truth {y.shape}")
    if p.ndim == 0:
        raise ValueError("expected at least one axis, with horizon last")

    out = {f"h{h + 1}": score(p[..., h], y[..., h]) for h in range(p.shape[-1])}
    out["overall"] = score(p, y)
    return out


def compute_morans_i(residuals: np.ndarray, adj: np.ndarray) -> float:
    """Compute Moran's I spatial autocorrelation on district residuals.

    Args:
        residuals: Spatial residuals of shape (N,) or (T, N).
        adj: Binary or spatial weight adjacency matrix of shape (N, N).

    Returns:
        Moran's I statistic between -1 (perfect dispersion) and +1 (perfect clustering).
    """
    res = np.asarray(residuals, dtype=np.float64)
    if res.ndim == 3:
        # (T, N, H) -> average over time and horizon to get (N,) per-district residual
        res = np.mean(res, axis=(0, -1))
    elif res.ndim == 2:
        # (T, N) -> average over time
        res = np.mean(res, axis=0)

    N = len(res)
    # Zero diagonal for spatial neighbors
    W = np.array(adj, dtype=np.float64, copy=True)
    np.fill_diagonal(W, 0.0)

    s0 = np.sum(W)
    if s0 == 0:
        return 0.0

    res_dev = res - np.mean(res)
    denom = np.sum(res_dev**2)
    if denom == 0:
        return 0.0

    numer = np.sum(W * np.outer(res_dev, res_dev))
    moran_i = float((N / s0) * (numer / denom))
    return moran_i


def compute_peak_timing_error(pred: np.ndarray, truth: np.ndarray) -> float:
    """Compute mean Peak Timing Error (PTE) in weeks.

    For each district, calculates the absolute difference in peak week index
    between observed cases and predicted cases.

    Args:
        pred: Predicted counts of shape (T, N, H) or (T, N).
        truth: Observed counts of shape (T, N, H) or (T, N).

    Returns:
        Mean absolute peak timing error across districts in weeks.
    """
    p = np.asarray(pred)
    y = np.asarray(truth)

    # If multi-horizon, take 1-step ahead (h=0) for peak trajectory
    if p.ndim == 3:
        p = p[..., 0]
        y = y[..., 0]

    T, N = p.shape
    errors = []
    for n in range(N):
        peak_true = int(np.argmax(y[:, n]))
        peak_pred = int(np.argmax(p[:, n]))
        errors.append(abs(peak_true - peak_pred))

    return float(np.mean(errors))
