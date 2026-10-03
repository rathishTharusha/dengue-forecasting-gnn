"""Which climate lags, if any, predict dengue *growth* -- measured on training data only.

The existing climate analysis (Stage S2, ``run_s2_causal_ccf.py``) correlates
climate at lag L with the *level* of log cases. Both are seasonal, so they
correlate whether or not climate drives anything, and a forecaster already
knows the level from last week. What a forecaster needs is the *growth*
persistence cannot see: log1p(cases[t+h]) - log1p(cases[t-1]).

The models have only ever seen climate at lags 2-4 weeks (``core.build_tensors``
slices a 3-week window ending at lag 2). The biology argues for longer: rainfall
has to make breeding sites, larvae mature over 1-2 weeks, the virus incubates in
the mosquito for 1-2 weeks and in people for 4-7 days. So lags out to 26 weeks
are scanned here.

**Leakage and selection.** Everything is fitted and scored inside each fold's
*training* windows, split in time: fit on the earliest 70%, score on the latest
30% ("inner validation"). The real validation windows are not touched, so the
lag windows chosen here can be evaluated honestly on them afterwards. Climate at
lag < 2 is never read (ERA5 release delay, ``cd.LAGS``).

Three measurements:
1. Correlation of each channel's *anomaly* (minus its district x week-of-year
   training climatology) at each lag with 3-week growth, pooled over districts.
2. Out-of-sample R2 gain of a ridge regression for growth when a block of lagged
   climate is added to what B already sees (last 3 weeks of cases, season,
   district effects) -- for raw climate and anomalies, and several lag blocks.
3. The same test with a longer *case* history, to revisit EXP-036 on growth.
4. The same climate blocks through gradient-boosted trees, which can find
   thresholds and district-specific responses a pooled ridge averages away.
5. NDVI at lags 0-26 (it is already as-of, so lag 0 is allowed).

    python seirgnn2/climate_lags.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import corrected_data as cd  # noqa: E402

LAG_MAX = 26
#: Candidate climate blocks (inclusive lag ranges, weeks before the forecast origin).
BLOCKS = {
    "2-4 (current)": [(2, 4)],
    "2-5": [(2, 5)],
    "6-9": [(6, 9)],
    "10-13": [(10, 13)],
    "14-17": [(14, 17)],
    "2-5,6-9": [(2, 5), (6, 9)],
    "2-5,6-9,10-13": [(2, 5), (6, 9), (10, 13)],
    "2-5..22-25 (6 blocks)": [(2, 5), (6, 9), (10, 13), (14, 17), (18, 21), (22, 25)],
}


def _climatology(data, weeks: np.ndarray) -> np.ndarray:
    """(53, N, C) mean climate by ISO week-of-year over ``weeks`` only."""
    woy = data.week_start.dt.isocalendar().week.to_numpy().astype(int)
    out = np.zeros((54, data.climate.shape[1], data.climate.shape[2]))
    for w in np.unique(woy[weeks]):
        sel = weeks[woy[weeks] == w]
        out[w] = np.nanmean(data.climate[sel], axis=0)
    return out, woy


def _rows(data, idx: np.ndarray, clim: np.ndarray, blocks, case_lags: int, horizon: int = 3):
    """Design matrix pooled over (window, district); growth target at each horizon."""
    cases = np.log1p(np.nan_to_num(data.cases))
    n = cases.shape[1]
    last = cases[idx - 1]                                                      # (K,N)
    hist = np.stack([cases[idx - k] - last for k in range(2, case_lags + 1)], -1) \
        if case_lags > 1 else np.zeros((len(idx), n, 0))
    season = np.repeat(core.seasonal_features(data.week_start, idx)[:, None, :], n, 1)
    dist = np.repeat(np.eye(n)[None], len(idx), 0)
    parts = [last[..., None], hist, season, dist]
    for a, b in blocks:
        parts.append(np.stack([clim[idx - lag] for lag in range(a, b + 1)], 0).mean(0))
    x = np.concatenate(parts, -1).reshape(len(idx) * n, -1)
    y = np.stack([cases[idx + h] - last for h in range(horizon)], -1).reshape(len(idx) * n, -1)
    return x, y


def _ridge_r2(xt, yt, xv, yv, alpha: float = 1.0) -> float:
    mu, sd = xt.mean(0), xt.std(0) + 1e-9
    xt, xv = (xt - mu) / sd, (xv - mu) / sd
    xt1, xv1 = np.c_[xt, np.ones(len(xt))], np.c_[xv, np.ones(len(xv))]
    reg = alpha * np.eye(xt1.shape[1]); reg[-1, -1] = 0.0
    beta = np.linalg.solve(xt1.T @ xt1 + reg, xt1.T @ yt)
    res = yv - xv1 @ beta
    return float(1 - (res ** 2).sum() / ((yv - yv.mean(0)) ** 2).sum())


def _gbm_r2(xt, yt, xv, yv) -> float:
    """Out-of-sample R2 over the 3 horizons with gradient-boosted trees, fixed settings."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    num = den = 0.0
    for h in range(yt.shape[1]):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=40, random_state=0)
        m.fit(xt, yt[:, h])
        res = yv[:, h] - m.predict(xv)
        num += (res ** 2).sum(); den += ((yv[:, h] - yv[:, h].mean()) ** 2).sum()
    return float(1 - num / den)


def main() -> None:
    data = cd.load()
    chans = data.climate_channels
    folds = core.build_folds(data.cases, data.missing, window=max(core.WINDOW, 13))

    print("Inner validation = latest 30% of each fold's training windows. Real validation untouched.\n")
    corr_acc = np.zeros((len(chans), LAG_MAX + 1))
    gains: dict[str, list[float]] = {}
    gbm: dict[str, list[float]] = {}
    ndvi_acc = np.zeros(LAG_MAX + 1)
    for f in folds:
        tr = f.idx["train"]
        tr = tr[tr - LAG_MAX >= 0]
        cut = int(0.7 * len(tr))
        fit, inner = tr[:cut], tr[cut:]
        clim_c, woy = _climatology(data, np.arange(0, int(fit.max())))
        anom = data.climate - clim_c[woy]

        # 1. anomaly-lag correlation with 3-week growth, on the fitting part only
        cases = np.log1p(np.nan_to_num(data.cases))
        g3 = (cases[fit + 2] - cases[fit - 1])                                  # (K,N)
        g3 = g3 - g3.mean(0)
        for c in range(len(chans)):
            for lag in range(2, LAG_MAX + 1):
                a = anom[fit - lag, :, c]; a = a - a.mean(0)
                corr_acc[c, lag] += np.corrcoef(a.ravel(), g3.ravel())[0, 1] / len(folds)

        nd = data.ndvi
        for lag in range(0, LAG_MAX + 1):
            a = nd[fit - lag]; a = a - np.nanmean(a, 0)
            ok = np.isfinite(a)
            ndvi_acc[lag] += np.corrcoef(a[ok], g3[ok])[0, 1] / len(folds)

        # 4. nonlinear check on the three most plausible blocks
        base_g = _gbm_r2(*_rows(data, fit, anom, [], 3), *_rows(data, inner, anom, [], 3))
        for name in ("2-5,6-9,10-13", "14-17", "2-5..22-25 (6 blocks)"):
            for kind, arr in (("raw", data.climate), ("anomaly", anom)):
                r2 = _gbm_r2(*_rows(data, fit, arr, BLOCKS[name], 3), *_rows(data, inner, arr, BLOCKS[name], 3))
                gbm.setdefault(f"{kind:7s} {name}", []).append(r2 - base_g)
        gbm.setdefault("(base R2, trees)", []).append(base_g)

        # 2./3. out-of-sample R2 of growth, base vs base + block
        for case_lags in (3, 8, 13):
            base = _ridge_r2(*_rows(data, fit, anom, [], case_lags), *_rows(data, inner, anom, [], case_lags))
            gains.setdefault(f"cases {case_lags}w, no climate", []).append(base)
            if case_lags != 3:
                continue
            for name, blocks in BLOCKS.items():
                for kind, arr in (("raw", data.climate), ("anomaly", anom)):
                    r2 = _ridge_r2(*_rows(data, fit, arr, blocks, 3), *_rows(data, inner, arr, blocks, 3))
                    gains.setdefault(f"{kind:7s} {name}", []).append(r2 - base)

    print("1. Correlation of climate ANOMALY at lag L with 3-week log growth (fit part, pooled)")
    print("   " + "lag:".ljust(28) + "".join(f"{lag:>6d}" for lag in (2, 4, 6, 8, 10, 12, 16, 20, 26)))
    for c, ch in enumerate(chans):
        print(f"   {ch:28s}" + "".join(f"{corr_acc[c, lag]:+6.3f}" for lag in (2, 4, 6, 8, 10, 12, 16, 20, 26))
              + f"   peak |r| {np.abs(corr_acc[c, 2:]).max():.3f} at lag {2 + int(np.abs(corr_acc[c, 2:]).argmax())}")

    print("\n2. Out-of-sample R2 of growth on inner validation (mean over 3 folds)")
    for k, v in gains.items():
        if k.startswith("cases"):
            print(f"   {k:34s} R2 = {np.mean(v):+.4f}   (per fold {', '.join(f'{x:+.3f}' for x in v)})")
    print("\n   Gain in R2 from adding climate to the 3-week case window + season + district:")
    for k, v in sorted(((k, v) for k, v in gains.items() if not k.startswith("cases")),
                       key=lambda kv: -np.mean(kv[1])):
        print(f"   {k:34s} dR2 = {np.mean(v):+.4f}   (per fold {', '.join(f'{x:+.4f}' for x in v)})")

    print()
    print("4. Same test through gradient-boosted trees (nonlinear, district interactions)")
    for k, v in gbm.items():
        lab = "R2" if k.startswith("(") else "dR2"
        print(f"   {k:34s} {lab} = {np.mean(v):+.4f}   (per fold {', '.join(f'{x:+.4f}' for x in v)})")

    print()
    print("5. Correlation of NDVI at lag L with 3-week log growth (fit part, pooled)")
    print("   " + "".join(f"  L{lag}:{ndvi_acc[lag]:+.3f}" for lag in (0, 2, 4, 6, 8, 10, 12, 16, 20, 26)))


if __name__ == "__main__":
    main()
