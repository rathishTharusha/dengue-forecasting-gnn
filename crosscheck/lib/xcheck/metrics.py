"""Forecast metrics, each transcribed from the paper (or code) that defines it.

Three papers in this workspace report overlapping metric names that do **not**
mean the same thing, so each variant is spelled out separately rather than
unified behind one "MAPE":

* :func:`mape_weng` -- ``mean(|p - y| / (y + 1e-15)) * 100``, transcribed from
  ``reference_repo/Models/evaluation.py``. The 1e-15 floor is not a stabilizer:
  on a zero week it returns ~1e17, so any batch containing a zero dominates the
  average. Reproduced faithfully because it is what produced the published
  numbers, not because it is defensible.
* :func:`mape_masked` -- the zero-masked variant this project reports instead.
* :func:`smape` -- bounded symmetric alternative.

:func:`morans_i` and the probabilistic metrics exist for the DengueGNN paper
(Tables 4 and 5), which reports CRPS/PICP/MPIW and Moran's I.

All functions take raw case counts, already inverse-transformed out of whatever
space the model trained in.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "WENG_MAPE_EPS",
    "crps_gaussian",
    "mae",
    "mape_masked",
    "mape_weng",
    "morans_i",
    "mpiw",
    "picp",
    "rmse",
    "smape",
]

#: Denominator floor in the reference implementation's MAPE. Not a stabilizer.
WENG_MAPE_EPS = 1e-15


def _pair(pred: ArrayLike, truth: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs truth {y.shape}")
    if p.size == 0:
        raise ValueError("cannot score empty arrays")
    return p, y


def rmse(pred: ArrayLike, truth: ArrayLike) -> float:
    """Root mean squared error over all elements."""
    p, y = _pair(pred, truth)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(pred: ArrayLike, truth: ArrayLike) -> float:
    """Mean absolute error over all elements."""
    p, y = _pair(pred, truth)
    return float(np.mean(np.abs(p - y)))


def mape_weng(pred: ArrayLike, truth: ArrayLike) -> float:
    """MAPE exactly as the Weng et al. reference implementation computes it.

    Percentage. Explodes to ~1e17 on any zero-truth element -- that behaviour is
    the point of transcribing it, so keep it.
    """
    p, y = _pair(pred, truth)
    return float(np.mean(np.abs(p - y) / (y + WENG_MAPE_EPS)) * 100)


def mape_masked(pred: ArrayLike, truth: ArrayLike, floor: float = 1.0) -> float:
    """MAPE over elements with ``truth >= floor``. ``nan`` if none qualify."""
    p, y = _pair(pred, truth)
    m = y >= floor
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(p[m] - y[m]) / y[m]) * 100)


def smape(pred: ArrayLike, truth: ArrayLike, eps: float = 1e-6) -> float:
    """Symmetric MAPE, percentage, bounded at 200."""
    p, y = _pair(pred, truth)
    return float(np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + eps)) * 100)


def morans_i(values: ArrayLike, weights: ArrayLike) -> float:
    """Global Moran's I of ``values`` under spatial weight matrix ``weights``.

    Args:
        values: Length-``n`` vector, one value per spatial unit (the DengueGNN
            paper applies this to predicted and observed incidence patterns).
        weights: ``(n, n)`` adjacency/weight matrix. The diagonal is ignored.

    Returns:
        Moran's I in roughly ``[-1, 1]``. ``nan`` when ``values`` is constant
        (zero variance leaves I undefined).
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64)
    n = x.size
    if w.shape != (n, n):
        raise ValueError(f"weights must be ({n}, {n}), got {w.shape}")

    w = w.copy()
    np.fill_diagonal(w, 0.0)
    s0 = w.sum()
    if s0 == 0:
        raise ValueError("weight matrix has no off-diagonal weight")

    d = x - x.mean()
    denom = float(np.sum(d**2))
    if denom == 0:
        return float("nan")
    return float((n / s0) * (d @ w @ d) / denom)


def crps_gaussian(mu: ArrayLike, sigma: ArrayLike, truth: ArrayLike) -> float:
    """Mean CRPS of Gaussian forecasts, closed form (Gneiting & Raftery 2007).

    ``CRPS = sigma * (z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))`` with
    ``z = (y - mu)/sigma``.
    """
    from scipy.stats import norm

    m = np.asarray(mu, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if not (m.shape == s.shape == y.shape):
        raise ValueError(f"shape mismatch: mu {m.shape}, sigma {s.shape}, truth {y.shape}")
    if np.any(s <= 0):
        raise ValueError("sigma must be strictly positive")

    z = (y - m) / s
    return float(np.mean(s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def picp(lower: ArrayLike, upper: ArrayLike, truth: ArrayLike) -> float:
    """Prediction Interval Coverage Probability -- fraction of truths inside."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if not (lo.shape == hi.shape == y.shape):
        raise ValueError("lower, upper and truth must share a shape")
    return float(np.mean((y >= lo) & (y <= hi)))


def mpiw(lower: ArrayLike, upper: ArrayLike) -> float:
    """Mean Prediction Interval Width."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError("lower and upper must share a shape")
    return float(np.mean(hi - lo))
