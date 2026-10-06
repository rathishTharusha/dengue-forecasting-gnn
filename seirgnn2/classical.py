"""Classical baselines and the block bootstrap of docs/PROSPECTIVE_PLAN.md.

NumPy only, so the tests run without torch. A forecast window is named by its
start ``i``, the first target row: inputs are rows ``i-3 .. i-1``, targets rows
``i .. i+2``. Every function here returns ``(K, N, H)`` count forecasts for the
starts in ``idx`` and reads no row at or after ``i``.
"""

from __future__ import annotations

import numpy as np

HORIZON = 3
SEASON = 52
LAGS = 3


def persistence(cases: np.ndarray, idx: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """``ŷ(i+h) = y(i-1)``."""
    return np.repeat(cases[idx - 1][:, :, None], horizon, axis=2)


def seasonal_naive(cases: np.ndarray, idx: np.ndarray,
                   horizon: int = HORIZON) -> tuple[np.ndarray, int]:
    """``ŷ(j) = y(j-52)`` for each target row ``j``, and the number of cells that
    fell back to persistence because row ``j-52`` is missing or before the series.

    ``j - 52 <= i - 50``, so the forecast never reads a row the origin has not seen.
    """
    out = persistence(cases, idx, horizon)
    fallback = 0
    for k, i in enumerate(idx):
        for h in range(horizon):
            src = i + h - SEASON
            row = cases[src] if src >= 0 else np.full(cases.shape[1], np.nan)
            ok = ~np.isnan(row)
            out[k, ok, h] = row[ok]
            fallback += int((~ok).sum())
    return out, fallback


def _ar_rows(cases: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-step design: ``log1p`` lags 1-3 then a district one-hot, target ``log1p y(i)``."""
    k, n = len(idx), cases.shape[1]
    lags = np.stack([np.log1p(cases[idx - lag]) for lag in range(1, LAGS + 1)], -1)  # (K,N,3)
    onehot = np.broadcast_to(np.eye(n), (k, n, n))
    x = np.concatenate([lags, onehot], -1).reshape(k * n, LAGS + n)
    y = np.log1p(cases[idx]).reshape(k * n)
    return x, y


def ar3_fit(cases: np.ndarray, train_idx: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge coefficients of the pooled one-step AR(3), fitted on training starts.

    The penalty applies to the three lag weights only; the district intercepts
    are left free, so a large ``alpha`` shrinks towards each district's mean
    rather than towards zero. The one-step targets are the starts' own first
    target rows, so nothing outside the training targets is read.
    """
    x, y = _ar_rows(cases, train_idx)
    pen = np.zeros(x.shape[1])
    pen[:LAGS] = alpha
    return np.linalg.solve(x.T @ x + np.diag(pen), x.T @ y)


def ar3_forecast(cases: np.ndarray, idx: np.ndarray, coef: np.ndarray,
                 horizon: int = HORIZON) -> np.ndarray:
    """Recursive forecasts: each step's ``log1p`` prediction becomes the next step's lag 1."""
    n = cases.shape[1]
    lags = [np.log1p(cases[idx - lag]) for lag in range(1, LAGS + 1)]   # each (K,N)
    w, b = coef[:LAGS], coef[LAGS:]
    out = []
    for _ in range(horizon):
        z = sum(w[j] * lags[j] for j in range(LAGS)) + b[None, :n]
        out.append(z)
        lags = [z, *lags[:-1]]
    return np.clip(np.expm1(np.stack(out, -1)), 0.0, None)


def per_start_loss(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """``(K,)`` squared error averaged over districts and horizons: one value per
    forecast start, the bootstrap's unit."""
    return ((pred - truth) ** 2).mean(axis=(1, 2))


def block_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray, block: int = 8,
                    reps: int = 10_000, seed: int = 0) -> dict[str, float]:
    """Paired circular moving-block bootstrap of ``RMSE_a - RMSE_b``.

    ``loss_a`` and ``loss_b`` are per-start losses over the same starts, in time
    order. Each replicate draws contiguous blocks of ``block`` starts (wrapping at
    the end) with replacement until it has as many starts as the original, uses
    the same draw for both arms, and recomputes the full RMSE difference. The
    p-value is twice the smaller share of replicates on either side of 0, capped
    at 1.
    """
    la, lb = np.asarray(loss_a, float), np.asarray(loss_b, float)
    if la.shape != lb.shape or la.ndim != 1:
        raise ValueError("per-start losses must be 1-D and the same length")
    n = len(la)
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(reps, n_blocks))
    pick = ((starts[:, :, None] + np.arange(block)) % n).reshape(reps, -1)[:, :n]
    rep = np.sqrt(la[pick].mean(1)) - np.sqrt(lb[pick].mean(1))
    delta = float(np.sqrt(la.mean()) - np.sqrt(lb.mean()))
    p = min(1.0, 2 * min(float((rep <= 0).mean()), float((rep >= 0).mean())))
    lo, hi = np.percentile(rep, [2.5, 97.5])
    return {"delta": delta, "lo": float(lo), "hi": float(hi), "p": p, "block": block,
            "reps": reps, "n_starts": n}


def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    order = np.argsort(p)
    m, out, run = len(p), [0.0] * len(p), 0.0
    for rank, j in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[j]))
        out[j] = run
    return out
