"""Multivariate inputs: the ten covariate channels the pipeline currently discards.

Every model in this project so far is **univariate**. ``adaptive.load_dataset``
takes ``raw[..., 5]`` and throws away ten channels of temperature, humidity, soil
moisture, canopy interception, precipitation and NDVI -- the exact variables the
dengue literature treats as the drivers of transmission. That is the largest
untested information lever left, so it gets a module rather than a flag.

What the channels are
---------------------
From ``docs/DATA.md``, verified against the source CSV at 100% agreement:

===== ============================= =====
index channel                       lag
===== ============================= =====
0     ``meanTair_F_Inst``           0
1     ``minTair_F_Inst``            0
2     ``maxTair_F_Inst``            0
3     ``meanQair_F_Inst`` humidity  0
4     ``meanSoilmoi0_10Cm_Inst``    0
5     ``cases`` (target)            0
6     ``meanCanopint_Inst``         12
7     ``meanPrecipitationcal``      12
8     ``minPrecipitationcal``       12
9     ``maxPrecipitationcal``       12
10    ``minNdvi``                   17
===== ============================= =====

**No leakage.** Channels 0-4 sit at lag 0, so week ``i`` holds week ``i``'s
weather. The model only ever sees the input window ``[t-W, t)``, which is past
weather at forecast time. Channels 6-10 are already shifted by 12 or 17 weeks,
so they carry longer-lead information and are safer still. Do not add a second
lag on top -- the array is pre-shifted, and ``docs/DATA.md`` says so twice.

The zero trap
-------------
The released array has no NaNs because the authors ran ``np.nan_to_num`` on load,
turning missing GLDAS values into **zeros**. A 0 K air temperature is 290 K from
every real value, so feeding these raw would poison the fit far more effectively
than the covariates could help.

Measured here, the missingness is not what ``docs/DATA.md``'s "~4%" suggests. It
is almost entirely **one district**: ``Jaffna`` is zero across all five GLDAS
channels for all 459 weeks -- 1 in 25 districts, which is where the 4% comes
from. The only genuinely scattered gaps are in ``meanCanopint``, 62 weeks spread
over ten other districts.

:func:`impute_missing` therefore does two different things: linear interpolation
along time for scattered gaps, and a fill from **graph neighbours** for a
district missing outright, which interpolation cannot reach. Either way it
returns a companion mask, and :func:`load_multivariate` appends it as an explicit
indicator channel rather than letting the model infer which values are synthetic.

Precipitation and NDVI are excluded from the zero-means-missing rule: zero
rainfall in a week is an ordinary observation, not a gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = [
    "CASES_INDEX",
    "FEATURE_NAMES",
    "FEATURE_SETS",
    "GLDAS_CHANNELS",
    "impute_missing",
    "load_multivariate",
    "resolve_set",
]

CASES_INDEX = 5

FEATURE_NAMES = (
    "meanTair", "minTair", "maxTair", "meanQair", "meanSoilmoi",
    "cases", "meanCanopint", "meanPrecip", "minPrecip", "maxPrecip", "minNdvi",
)

#: Channels where an exact zero means "missing", not "none". GLDAS only.
GLDAS_CHANNELS = (0, 1, 2, 3, 4, 6)

#: Named feature subsets, smallest first.
#:
#: ``cases``
#:     The current pipeline, as a control. Any covariate arm must beat this one
#:     or the covariates are not paying for the parameters they add.
#: ``causal``
#:     Humidity, soil moisture and temperature, which a causality-tested Vietnam
#:     dengue study found to dominate its lagged-predictor set, plus mean
#:     precipitation. Deliberately small: 12.6% of windows are outbreaks and the
#:     training split is ~200 weeks, so a wide input is a variance problem.
#: ``climate``
#:     Everything. The upper bound on what these channels can contribute, and the
#:     arm most exposed to overfitting.
FEATURE_SETS = {
    "cases": (5,),
    "causal": (5, 3, 4, 0, 7),
    "climate": tuple(range(11)),
}


def resolve_set(name_or_indices) -> tuple[int, ...]:
    """Accept a named set or an explicit index tuple; always keep cases first."""
    if isinstance(name_or_indices, str):
        if name_or_indices not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {name_or_indices!r}; "
                             f"pick from {sorted(FEATURE_SETS)}")
        idx = FEATURE_SETS[name_or_indices]
    else:
        idx = tuple(int(i) for i in name_or_indices)
    if CASES_INDEX not in idx:
        raise ValueError("the cases channel must be present -- it is the target")
    # Cases first so downstream code can slice it without a lookup.
    return (CASES_INDEX, *(i for i in idx if i != CASES_INDEX))


def impute_missing(
    raw: np.ndarray, neighbours: dict[int, list[int]] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct GLDAS gaps that ``np.nan_to_num`` turned into zeros.

    Two kinds of gap exist in this array, and they need different treatment.

    **Scattered gaps** -- a handful of weeks in ``meanCanopint`` across eleven
    districts -- are filled by linear interpolation along time, held flat beyond
    the first and last observation.

    **A whole district.** ``Jaffna`` is zero in *every* GLDAS channel for *all*
    459 weeks. That is not 4% of values scattered at random, it is one district
    in twenty-five entirely absent from the weather grid, which is where
    ``docs/DATA.md``'s "~4% missing" figure actually comes from. Interpolation
    along time cannot touch it -- there is nothing to interpolate from -- and
    leaving a 0 K temperature in place would hand the model an outlier 290 K from
    every real value. Where a neighbour map is supplied, such a district is
    filled from the mean of its graph neighbours at each week, falling back to
    the all-district weekly mean if those are missing too.

    Args:
        raw: ``(weeks, districts, features)``.
        neighbours: District index -> adjacent district indices. Without it,
            fully-missing districts are left at zero and only the indicator
            marks them.

    Returns:
        ``(filled, was_missing)``, both the same shape. ``was_missing`` is 1.0
        where a value was reconstructed, so the model can be told *that* a value
        is synthetic rather than being left to infer it.
    """
    filled = raw.astype(np.float64).copy()
    was_missing = np.zeros_like(filled)
    weeks = np.arange(filled.shape[0], dtype=np.float64)
    n_districts = filled.shape[1]

    for c in GLDAS_CHANNELS:
        if c >= filled.shape[-1]:
            continue
        fully_missing = []
        for d in range(n_districts):
            series = filled[:, d, c]
            gap = series == 0.0
            if not gap.any():
                continue
            was_missing[gap, d, c] = 1.0
            if gap.all():
                fully_missing.append(d)
                continue
            good = ~gap
            series[gap] = np.interp(weeks[gap], weeks[good], series[good])

        # Spatial fallback, after the time pass so donors are already filled.
        for d in fully_missing:
            donors = [n for n in (neighbours or {}).get(d, []) if n not in fully_missing]
            if donors:
                filled[:, d, c] = filled[:, donors, c].mean(axis=1)
            else:
                others = [j for j in range(n_districts) if j not in fully_missing]
                if others:
                    filled[:, d, c] = filled[:, others, c].mean(axis=1)
    return filled, was_missing


def load_multivariate(npy_path: str | Path, adj_path: str | Path, features="cases",
                      with_missing_indicator: bool = True):
    """Load the array as ``(weeks, districts, k)`` plus the graph.

    Args:
        npy_path: The processed ``.npy``.
        adj_path: District adjacency JSON.
        features: A key of :data:`FEATURE_SETS` or an explicit index tuple.
        with_missing_indicator: Append one indicator channel per imputed GLDAS
            channel that actually had a gap.

    Returns:
        ``(cases, stack, adjacency, names)`` where ``cases`` is
        ``(weeks, districts)`` raw counts -- the target, untouched -- and
        ``stack`` is ``(weeks, districts, k)`` of input channels with cases
        first. Nothing here is normalised; the folds do that from training weeks
        only.
    """
    raw = np.nan_to_num(np.load(Path(npy_path), allow_pickle=True)).astype(np.float64)

    # The graph is built first because imputation needs it: one district is
    # missing every GLDAS channel outright and can only be filled from its
    # neighbours.
    adj_raw = json.loads(Path(adj_path).read_text(encoding="utf-8"))
    district_names = sorted(adj_raw)
    index = {name: i for i, name in enumerate(district_names)}
    neighbours = {
        index[d]: [index[n] for n in ns if n in index] for d, ns in adj_raw.items()
    }
    filled, was_missing = impute_missing(raw, neighbours)

    idx = resolve_set(features)
    stack = filled[..., list(idx)]
    names = [FEATURE_NAMES[i] for i in idx]

    if with_missing_indicator:
        for i in idx:
            if i in GLDAS_CHANNELS and was_missing[..., i].any():
                stack = np.concatenate([stack, was_missing[..., i : i + 1]], axis=-1)
                names.append(f"{FEATURE_NAMES[i]}_missing")

    n = len(district_names)
    a = np.eye(n)
    for district, ns in adj_raw.items():
        for neighbour in ns:
            if neighbour in index:
                a[index[district], index[neighbour]] = 1.0
    a = np.maximum(a, a.T)
    deg = a.sum(1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))

    return filled[..., CASES_INDEX], stack, d_inv_sqrt @ a @ d_inv_sqrt, names


def build_folds_mv(
    cases: np.ndarray,
    stack: np.ndarray,
    window: int = 3,
    horizon: int = 3,
    origins: tuple[float, ...] = (0.55, 0.70, 0.85),
    test_frac: float = 0.15,
    val_weeks: int = 30,
):
    """Rolling-origin folds with multivariate inputs.

    Deliberately mirrors ``adaptive.build_folds`` line for line in how it splits,
    normalises and packs, so a covariate arm and the univariate control differ in
    the input channels and in nothing else. The target, the persistence term and
    the fold boundaries are identical.

    The cases channel is normalised in ``log1p`` space, matching the target.
    Every other channel is z-scored on its raw scale. Both use **training weeks
    only**.

    Inputs are packed feature-major -- ``x[..., f * window + w]`` -- so a model
    that wants proper ``(nodes, features, window)`` semantics can reshape without
    a transpose, while a model that treats the input as one flat vector still
    sees each variable's window contiguously.

    Returns:
        A list of ``adaptive.Fold``, with ``x_*`` of shape
        ``(n_windows, nodes, features * window)``.
    """
    import adaptive as base
    import torch

    n_weeks = cases.shape[0]
    n_feat = stack.shape[-1]
    ids = list(range(window, n_weeks - horizon))
    folds = []

    for origin in origins:
        cut = int(origin * len(ids))
        end = int(min(origin + test_frac, 1.0) * len(ids))
        if end <= cut:
            continue
        train_ids = ids[: cut - val_weeks]
        val_ids = ids[cut - val_weeks : cut]
        test_ids = ids[cut:end]
        last_train_week = train_ids[-1] + 1

        # Target: log1p, z-scored on training weeks only -- unchanged.
        train_weeks = np.log1p(cases[:last_train_week])
        mean, std = float(train_weeks.mean()), float(train_weeks.std() + 1e-8)
        z = (np.log1p(cases) - mean) / std

        # Inputs: cases in log1p space to match the target, everything else raw.
        chans = []
        for c in range(n_feat):
            series = np.log1p(stack[..., c]) if c == 0 else stack[..., c]
            m = float(series[:last_train_week].mean())
            s = float(series[:last_train_week].std() + 1e-8)
            chans.append((series - m) / s)
        norm = np.stack(chans, axis=-1)  # (weeks, districts, features)

        def pack(chosen, scaled=z, feats=norm):
            # Bound at definition time: closing over the loop variables would
            # silently use the last origin's scaling if called outside the
            # iteration that created it.
            x = np.stack([
                np.concatenate([feats[i - window : i, :, f].T for f in range(n_feat)], axis=1)
                for i in chosen
            ])
            y = np.stack([scaled[i : i + horizon].T for i in chosen])
            p = np.stack([np.repeat(scaled[i - 1][:, None], horizon, axis=1) for i in chosen])
            to = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
            return to(x), to(y), to(p)

        xtr, ytr, ptr = pack(train_ids)
        xva, yva, pva = pack(val_ids)
        xte, yte, pte = pack(test_ids)
        folds.append(
            base.Fold(
                origin=origin,
                x_train=xtr, y_train=ytr, p_train=ptr,
                x_val=xva, y_val=yva, p_val=pva,
                x_test=xte, y_test=yte, p_test=pte,
                mean=mean, std=std,
                train_index=np.array(train_ids),
                val_index=np.array(val_ids),
                test_index=np.array(test_ids),
            )
        )
    return folds
