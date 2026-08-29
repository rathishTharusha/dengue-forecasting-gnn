"""Forecast metrics, on the original case scale.

RMSE/MAE/SMAPE/MAPE were extracted from ``dengue_baseline_GNN_v2.ipynb`` section 6
so that every phase of the project is scored by one implementation.

Why these:

* ``RMSE`` / ``MAE`` -- reported on raw counts, comparable to Weng et al. (2024).
* ``SMAPE`` -- bounded and symmetric; the primary percentage metric, because 9.7%
  of district-weeks are zero and plain MAPE explodes on them.
* ``MAPE`` -- computed only over weeks with ``truth >= 1`` (zero-masked), kept for
  comparability with papers that report it. ``nan`` when no week qualifies.

Predictions and targets must already be inverse-transformed out of log1p /
residual space before they get here: these are raw-count metrics.

A note on axes, because this caused a real bug
----------------------------------------------
:func:`score` and :func:`metrics` are deliberately **shape-agnostic** -- they treat
every element as an independent observation and never ask what an axis means. That
is correct for RMSE/MAE/SMAPE/MAPE, which are elementwise.

Peak timing is not elementwise. It needs to know which axis is *time*, and a
shape-agnostic function cannot know that. An earlier version computed it inside
``score()`` with ``argmax(axis=-1)``; on this project's ``(n_windows, n_nodes,
horizon)`` arrays that took the argmax over the 3 horizon steps, and on a
horizon-sliced 2D array it took the argmax over the 25 districts. It never once
looked along time, and reported 0.0 for series that were shifted by any amount.

So peak timing lives in :func:`peak_week_error`, which *requires* a
``(n_weeks, n_nodes)`` array and raises otherwise. The type signature is the fix:
you cannot call it wrongly without getting an exception.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "MAPE_FLOOR",
    "PEAK_FLOOR",
    "SMAPE_EPS",
    "metrics",
    "peak_week_error",
    "peak_week_error_by_horizon",
    "score",
    "smape",
]

#: Denominator stabilizer in SMAPE, matching the notebook implementation.
SMAPE_EPS = 1e-6

#: Weeks with ``truth`` below this are excluded from MAPE.
MAPE_FLOOR = 1.0

#: Districts whose observed maximum is below this have no outbreak to time, and
#: are excluded from :func:`peak_week_error`. A flat or near-zero series has an
#: argmax, but it is noise, not a peak.
PEAK_FLOOR = 10.0


def smape(pred: ArrayLike, truth: ArrayLike) -> float:
    """Symmetric mean absolute percentage error, as a percentage.

    Bounded at 200%, and finite when both values are zero.
    """
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    _check_pair(p, y)
    err = p - y
    return float(np.mean(2 * np.abs(err) / (np.abs(p) + np.abs(y) + SMAPE_EPS)) * 100)


def score(pred: ArrayLike, truth: ArrayLike) -> dict[str, float]:
    """Return RMSE, MAE, SMAPE and zero-masked MAPE for one set of forecasts.

    Every metric here is elementwise, so the shape of the inputs does not matter
    beyond ``pred`` and ``truth`` matching. For peak timing, which is *not*
    elementwise, use :func:`peak_week_error`.

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
    _check_pair(p, y)

    err = p - y
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    mask = y >= MAPE_FLOOR
    mape = float(np.mean(np.abs(err[mask]) / y[mask]) * 100) if mask.any() else float("nan")

    return {"RMSE": rmse, "MAE": mae, "SMAPE": smape(p, y), "MAPE": mape}


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
    _check_pair(p, y)
    if p.ndim == 0:
        raise ValueError("expected at least one axis, with horizon last")

    out = {f"h{h + 1}": score(p[..., h], y[..., h]) for h in range(p.shape[-1])}
    out["overall"] = score(p, y)
    return out


def peak_week_error(
    pred: ArrayLike,
    truth: ArrayLike,
    min_peak: float = PEAK_FLOOR,
) -> float:
    """Mean absolute error, in weeks, between predicted and observed outbreak peaks.

    For each district this finds the week of maximum incidence in the observed
    series and in the forecast series, and reports the mean absolute difference.
    This is the operationally meaningful timing question: *did we call the peak
    early, late, or on time?*

    Inputs must be ``(n_weeks, n_nodes)`` -- an explicit weekly series per
    district, with time along axis 0. Any other rank raises. This is deliberate:
    the previous implementation accepted any shape and silently took the argmax
    over whichever axis happened to be last, which was never time.

    To go from this project's ``(n_windows, n_nodes, horizon)`` forecast arrays to
    the required layout, slice a horizon: ``pred[:, :, h]``. See
    :func:`peak_week_error_by_horizon`.

    Districts whose observed maximum is below ``min_peak`` are excluded -- a
    district with no outbreak has no peak to time, and its argmax is noise.

    Args:
        pred: Predicted weekly case counts, shape ``(n_weeks, n_nodes)``.
        truth: Observed weekly case counts, same shape.
        min_peak: Observed-maximum threshold below which a district is skipped.

    Returns:
        Mean absolute peak displacement in weeks, over qualifying districts.
        ``nan`` if no district reaches ``min_peak``.

    Raises:
        ValueError: If shapes mismatch, inputs are empty, or rank is not 2.
    """
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    _check_pair(p, y)

    if p.ndim != 2:
        raise ValueError(
            f"peak_week_error requires a 2D (n_weeks, n_nodes) array with time on "
            f"axis 0, got shape {p.shape}. Slice a single horizon first, e.g. "
            f"pred[:, :, h], or use peak_week_error_by_horizon()."
        )
    if p.shape[0] < 2:
        raise ValueError("need at least 2 weeks to locate a peak")

    qualifies = y.max(axis=0) >= min_peak
    if not qualifies.any():
        return float("nan")

    peak_pred = np.argmax(p[:, qualifies], axis=0)
    peak_truth = np.argmax(y[:, qualifies], axis=0)
    return float(np.mean(np.abs(peak_pred - peak_truth)))


def peak_week_error_by_horizon(
    pred: ArrayLike,
    truth: ArrayLike,
    min_peak: float = PEAK_FLOOR,
) -> dict[str, float]:
    """Peak week error for each forecast horizon.

    Takes this project's native ``(n_windows, n_nodes, H)`` layout, where axis 0
    advances one week per window, and evaluates :func:`peak_week_error` on each
    horizon slice.

    Args:
        pred: Predicted case counts, shape ``(n_weeks, n_nodes, H)``.
        truth: Observed case counts, same shape.
        min_peak: Passed through to :func:`peak_week_error`.

    Returns:
        Mapping from ``"h1"``..``"hH"`` to peak week error in weeks.

    Raises:
        ValueError: If shapes mismatch or rank is not 3.
    """
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    _check_pair(p, y)

    if p.ndim != 3:
        raise ValueError(f"expected (n_weeks, n_nodes, horizon), got shape {p.shape}")

    return {
        f"h{h + 1}": peak_week_error(p[:, :, h], y[:, :, h], min_peak=min_peak)
        for h in range(p.shape[-1])
    }


def _check_pair(p: np.ndarray, y: np.ndarray) -> None:
    """Shared shape/emptiness validation."""
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs truth {y.shape}")
    if p.size == 0:
        raise ValueError("cannot score empty arrays")
