"""Gradient-boosted trees on growth -- arms K6 / K7 of ``docs/CLIMATE_PLAN.md``.

EXP-043 found that a tree model, unlike a linear one, extracts a consistent
out-of-sample gain from climate at lags 2-13 weeks inside the training data.
This runs that same model on the real validation and test windows, with and
without climate, so the gain can be checked where it counts and compared with
the network.

Features, exactly as in EXP-043 (``climate_lags._rows``): the last observed
week's log1p cases, the two earlier weeks as differences from it, the seasonal
features, district indicators, and -- for K6 -- the raw climate lag blocks,
standardised on training windows. One regressor per horizon predicts log
growth; counts are recovered with Duan's smearing factor estimated on training
residuals, so the forecast targets the conditional mean that RMSE rewards
rather than the median a log-space fit returns.

Deterministic: every seed gives the same forecast.
"""

from __future__ import annotations

import numpy as np

import core

HP = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)


def _design(data, fold: core.Fold, idx: np.ndarray, clim_blocks) -> np.ndarray:
    cases = np.log1p(np.nan_to_num(data.cases))
    n = cases.shape[1]
    last = cases[idx - 1]
    hist = np.stack([cases[idx - k] - last for k in (2, 3)], -1)
    season = np.repeat(core.seasonal_features(data.week_start, idx)[:, None, :], n, 1)
    dist = np.repeat(np.eye(n)[None], len(idx), 0)
    parts = [last[..., None], hist, season, dist]
    if clim_blocks:
        blk = core.climate_blocks(data, idx, clim_blocks, data.climate)
        ref = core.climate_blocks(data, fold.idx["train"], clim_blocks, data.climate)
        parts.append((blk - ref.mean((0, 1))) / (ref.std((0, 1)) + 1e-8))
    return np.concatenate(parts, -1)


def run_fold(data, fold: core.Fold, *, seed: int = 0, keep: bool = False, clim_blocks=(),
             **_ignored):
    from sklearn.ensemble import HistGradientBoostingRegressor

    h = core.HORIZON
    if clim_blocks:
        fold = core.with_history(fold, max(b for _, b in clim_blocks))
    cases = np.log1p(np.nan_to_num(data.cases))
    x = {s: _design(data, fold, fold.idx[s], clim_blocks) for s in ("train", "val", "test")}
    n = x["train"].shape[1]
    xt = x["train"].reshape(-1, x["train"].shape[-1])
    last_t = cases[fold.idx["train"] - 1]
    preds = {s: np.zeros((len(fold.idx[s]), n, h)) for s in x}
    for k in range(h):
        yt = (cases[fold.idx["train"] + k] - last_t).reshape(-1)
        m = HistGradientBoostingRegressor(**HP).fit(xt, yt)
        smear = float(np.mean(np.exp(yt - m.predict(xt))))
        for s in x:
            g = m.predict(x[s].reshape(-1, x[s].shape[-1])).reshape(len(fold.idx[s]), n)
            base = np.nan_to_num(data.cases)[fold.idx[s] - 1] + 1.0
            preds[s][..., k] = np.clip(base * np.exp(g) * smear - 1.0, 0.0, None)

    packs = {s: core.build_tensors(data, fold, s, False, False, False) for s in x}
    truth = packs["test"]["y_raw"].numpy()
    out = core.score(preds["test"], truth)
    out.update(val_RMSE=core.rmse(preds["val"], packs["val"]["y_raw"].numpy()), origin=fold.origin,
               seed=seed, backbone="gbm", head="-", loss="-", dist="-",
               clim_blocks=[list(b) for b in clim_blocks])
    if keep:
        extra = {s: {"pred": preds[s], "truth": packs[s]["y_raw"].numpy(),
                     "persist": packs[s]["p_raw"].numpy(),
                     "last": packs[s]["x_raw"].numpy()[..., -1], "idx": packs[s]["idx"]}
                 for s in x}
        return out, extra
    return out, preds["test"], truth
