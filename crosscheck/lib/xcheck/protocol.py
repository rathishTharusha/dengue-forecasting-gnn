"""Evaluation protocols, reproduced as published and as corrected.

Weng et al. describe their protocol as "Rolling Cross Validation ... rolled
across 5 segments consisting of 60%, 70%, 80%, 90%, and 100% of the data. Every
GNN is reinitialized at each segment, and each segment is split into train and
test sets following a 70/30 split."

Their code does that, and then reports the error of the ``full`` loader -- every
window in the segment, training windows included. The held-out test score is
computed on the line above and discarded:

.. code-block:: python

    y_pred, y_truth, _, _ = infer(model, "cpu", test, m, s, "Test")   # dropped
    y_pred, y_truth, ma, rm = infer(model, "cpu", full, m, s, "Full")  # reported

:func:`segment_cv` therefore takes a ``report_on`` argument. ``"full"``
reproduces the published Table I; ``"test"`` is the number the protocol
description implies. Running both is the point.

A second detail worth naming: the reference accumulates RMSE *per batch* and
averages the result. With ``batch_size=1`` that is a mean of per-window RMSEs,
which is not the RMSE of the pooled predictions and is systematically smaller.
:func:`batched_mean_metric` reproduces that averaging so the comparison is like
for like.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "WENG_SEGMENTS",
    "FoldResult",
    "batched_mean_metric",
    "rolling_origin_folds",
    "segment_cv",
    "summarize",
]

#: The five nested prefixes Weng et al. cross-validate over.
WENG_SEGMENTS: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass
class FoldResult:
    """Metrics from one segment or fold.

    Attributes:
        label: Identifier for the fold (segment fraction, origin index, ...).
        n_train: Number of training windows.
        n_eval: Number of windows the reported metrics were computed over.
        metrics: Metric name to value.
        extra: Anything else worth carrying to the results CSV.
    """

    label: float | str
    n_train: int
    n_eval: int
    metrics: dict[str, float]
    extra: dict[str, float | str] = field(default_factory=dict)


def batched_mean_metric(
    pred: np.ndarray,
    truth: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    batch_size: int = 1,
) -> float:
    """Average ``metric`` over mini-batches instead of pooling all predictions.

    This is what the reference implementation does (``rmse += RMSE(...)`` inside
    the loader loop, then ``rmse /= n``). For RMSE with ``batch_size=1`` it
    returns ``mean_w RMSE_w``, which is at most the pooled RMSE by Jensen's
    inequality -- so a model looks better under this averaging than under the
    pooled metric, and the gap grows with how uneven the error is across weeks.

    Args:
        pred: ``(n_windows, ...)`` predictions, window on axis 0.
        truth: Same shape as ``pred``.
        metric: Callable taking ``(pred, truth)`` and returning a float.
        batch_size: Windows per batch.

    Returns:
        The mean of the per-batch metric values.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs truth {truth.shape}")
    if pred.shape[0] == 0:
        raise ValueError("cannot score zero windows")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    values = [
        metric(pred[i : i + batch_size], truth[i : i + batch_size])
        for i in range(0, pred.shape[0], batch_size)
    ]
    return float(np.mean(values))


def segment_cv(
    n_windows: int,
    segments: Sequence[float] = WENG_SEGMENTS,
    train_frac: float = 0.7,
    report_on: str = "full",
) -> list[tuple[float, slice, slice]]:
    """Index plan for Weng et al.'s segment cross-validation.

    Args:
        n_windows: Number of windows in the *complete* series.
        segments: Nested prefixes of the data, as fractions.
        train_frac: Train share within each segment.
        report_on: ``"full"`` evaluates on the whole segment -- training windows
            included -- which reproduces the published numbers. ``"test"``
            evaluates only on the held-out tail.

    Returns:
        One ``(segment, train_slice, eval_slice)`` triple per segment.

    Raises:
        ValueError: On an unknown ``report_on`` or an empty segment.
    """
    if report_on not in {"full", "test"}:
        raise ValueError(f"report_on must be 'full' or 'test', got {report_on!r}")

    plan: list[tuple[float, slice, slice]] = []
    for frac in segments:
        n_seg = int(n_windows * frac)
        if n_seg == 0:
            raise ValueError(f"segment {frac} of {n_windows} windows is empty")
        cut = int(n_seg * train_frac)
        train = slice(0, cut)
        evaluate = slice(0, n_seg) if report_on == "full" else slice(cut, n_seg)
        plan.append((frac, train, evaluate))
    return plan


def rolling_origin_folds(
    n_windows: int,
    origins: Sequence[float] = (0.55, 0.70, 0.85),
    test_frac: float = 0.15,
    val_windows: int = 30,
) -> list[tuple[float, slice, slice, slice]]:
    """Index plan for expanding-window rolling-origin CV.

    The protocol this project froze for its own ablation table, restated here so
    R1 can score Weng's architectures under it without importing project code.

    Args:
        n_windows: Number of windows available.
        origins: Fractions of the series at which each test block begins.
        test_frac: Width of each test block, as a fraction of the series.
        val_windows: Windows taken off the end of the training pool for
            validation and early stopping.

    Returns:
        One ``(origin, train_slice, val_slice, test_slice)`` per origin that has
        a non-empty test block.
    """
    folds: list[tuple[float, slice, slice, slice]] = []
    for origin in origins:
        cut = int(origin * n_windows)
        end = int(min(origin + test_frac, 1.0) * n_windows)
        if end <= cut:
            continue
        if cut - val_windows <= 0:
            raise ValueError(f"origin {origin} leaves no training windows before validation")
        folds.append(
            (
                origin,
                slice(0, cut - val_windows),
                slice(cut - val_windows, cut),
                slice(cut, end),
            )
        )
    return folds


def summarize(results: Sequence[FoldResult]) -> dict[str, tuple[float, float]]:
    """Mean and population standard deviation of each metric across folds.

    ``np.std`` with default ``ddof=0`` matches the reference implementation's
    reported "Standard Deviation".
    """
    if not results:
        raise ValueError("no fold results to summarize")
    names = results[0].metrics.keys()
    out: dict[str, tuple[float, float]] = {}
    for name in names:
        values = np.array([r.metrics[name] for r in results], dtype=np.float64)
        out[name] = (float(values.mean()), float(values.std()))
    return out
