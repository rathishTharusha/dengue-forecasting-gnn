"""k-nearest-neighbour analogue forecasting -- remedy R4b.

The 2026 dengue network meta-analysis (Benjarattanaporn et al.) ranks k-NN
first on RMSE across published dengue forecasting studies, ahead of vector
autoregression, Kalman filtering, GLMs and neural networks. It is also as
different from the neural encoders as a forecaster can be, which is the point
for remedy R5: EXP-039 found the three working encoders' residuals correlate at
0.98-0.99, so an ensemble needs a member that errs differently.

Method. Every (training window, district) pair is a stored analogue: the shape
of its last three log1p weeks relative to the last one, its level, and the
time of year. A query window finds its k nearest analogues and forecasts the
growth they went on to show, applied to its own last value. Averaging is done
on counts, not logs, so the forecast targets the mean that RMSE rewards (the
same reason the NB likelihood helped).

Only training windows whose *targets* end before the first validation origin
are analogues. ``core.build_folds`` leaves no gap, so the last two training
windows' targets reach into the first validation windows' own target weeks; the
neural arms train on those too, but k-NN could copy them directly, so they are
dropped here. A query reads nothing but its own input window.
``tests/test_knn_leakage.py`` checks both properties.

k and the two feature weights are chosen on validation RMSE, per fold, from a
small fixed grid; the test split is scored only with the chosen setting.
"""

from __future__ import annotations

import itertools

import numpy as np

import core

GRID_K = (10, 20, 40, 80, 160, 320)
GRID_LEVEL = (0.0, 0.5, 1.0)
GRID_SEASON = (0.0, 0.5)


def library_idx(fold: core.Fold) -> np.ndarray:
    """Training origins whose whole target span ends before the first validation origin."""
    tr = fold.idx["train"]
    return tr[tr + core.HORIZON - 1 < int(fold.idx["val"].min())]


def _windows(data, idx: np.ndarray, window: int, horizon: int, with_target: bool):
    """Per (window, district): log1p history, last value, season, and target growth."""
    cases = np.nan_to_num(data.cases)
    hist = np.stack([np.log1p(cases[i - window : i]).T for i in idx])          # (K,N,W)
    last = hist[..., -1]                                                       # (K,N)
    season = core.seasonal_features(data.week_start, idx)                      # (K,4)
    season = np.repeat(season[:, None, :], hist.shape[1], 1)                   # (K,N,4)
    growth = None
    if with_target:
        fut = np.stack([np.log1p(cases[i : i + horizon]).T for i in idx])      # (K,N,H)
        growth = fut - last[..., None]
    return hist, last, season, growth


def _features(hist, last, season, w_level: float, w_season: float) -> np.ndarray:
    shape = hist - last[..., None]
    return np.concatenate([shape, w_level * last[..., None], w_season * season], -1)


def _predict(lib, query, k: int, w_level: float, w_season: float) -> np.ndarray:
    lh, ll, ls, lg = lib
    qh, ql, qs, _ = query
    lf = _features(lh, ll, ls, w_level, w_season).reshape(-1, lh.shape[-1] + 5)
    qf = _features(qh, ql, qs, w_level, w_season).reshape(-1, qh.shape[-1] + 5)
    lg = lg.reshape(-1, lg.shape[-1])
    lsq = (lf ** 2).sum(1)[None, :]
    q_last = ql.reshape(-1)
    out = np.empty((len(qf), lg.shape[-1]))
    # Chunked: the training split as queries against itself would otherwise
    # need a ~560 MB distance matrix.
    for a in range(0, len(qf), 1024):
        q = qf[a : a + 1024]
        d2 = (q ** 2).sum(1)[:, None] - 2 * q @ lf.T + lsq
        nn_idx = np.argpartition(d2, kth=min(k, d2.shape[1] - 1), axis=1)[:, :k]   # (q,k)
        counts = np.expm1(q_last[a : a + 1024, None, None] + lg[nn_idx])           # (q,k,H)
        out[a : a + 1024] = counts.mean(1)
    return out.clip(min=0.0).reshape(*ql.shape, -1)                                # (K,N,H)


def run_fold(data, fold: core.Fold, *, seed: int = 0, keep: bool = False, **_ignored):
    """Choose (k, weights) on validation, score test; same row schema as ``train.run_fold``."""
    w, h = fold.window, core.HORIZON
    lib = _windows(data, library_idx(fold), w, h, with_target=True)
    packs = {s: core.build_tensors(data, fold, s, False, False, False) for s in ("train", "val", "test")}
    queries = {s: _windows(data, fold.idx[s], w, h, with_target=False) for s in packs}

    best = None
    for k, wl, ws in itertools.product(GRID_K, GRID_LEVEL, GRID_SEASON):
        v = core.rmse(_predict(lib, queries["val"], k, wl, ws), packs["val"]["y_raw"].numpy())
        if best is None or v < best[0]:
            best = (v, k, wl, ws)
    val_rmse, k, wl, ws = best

    preds = {s: _predict(lib, queries[s], k, wl, ws) for s in packs}
    truth = packs["test"]["y_raw"].numpy()
    out = core.score(preds["test"], truth)
    out.update(val_RMSE=val_rmse, origin=fold.origin, seed=seed, backbone="knn", head="-",
               loss="-", dist="-", knn_k=k, knn_level=wl, knn_season=ws)
    if keep:
        extra = {s: {"pred": preds[s], "truth": packs[s]["y_raw"].numpy(),
                     "persist": packs[s]["p_raw"].numpy(),
                     "last": packs[s]["x_raw"].numpy()[..., -1], "idx": packs[s]["idx"]}
                 for s in packs}
        return out, extra
    return out, preds["test"], truth
