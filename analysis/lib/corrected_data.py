"""The corrected, dated dataset -- with the no-future-information rule enforced in code.

Everything the SEIR-GNN experiments read comes through this module. It exists so
that "no future information" is a property the code guarantees, not a convention
someone has to remember.

The rule
--------
A forecast is made at the start of week ``i`` and targets weeks ``i .. i+H-1``.
It may only use what was **published before week i began**. Each input gets a
minimum lag -- the latest row it may read is ``i - lag``:

============  ====  ===========================================================
input         lag   why
============  ====  ===========================================================
cases         1     the report for week i-1 is the latest published
climate       2     ERA5 is released ~5 days after the fact, so week i-1's
                    weather is not complete at the start of week i
ndvi          0     rows are already as-of: each week holds the latest MODIS
                    composite available *before* that week began
population    0     rows are already lagged: each year uses the previous year's
                    official figure
============  ====  ===========================================================

:func:`windows` applies these lags when it slices, and the leakage test perturbs
every value at or after the limit and checks the result does not move.

What is NOT an input
--------------------
* The 2022-23 district seroprevalence survey: collected after almost the whole
  series, so it is for **validation only** (does the model's implied immunity
  match it?), never for initial conditions.
* Serotype-switch dates (DENV-2 in 2016-17, DENV-3 in late 2019): known only in
  hindsight. An immunity reset placed at those dates is an **oracle** arm and must
  be labelled as such.

Nothing is filled
-----------------
Weeks with no source report are NaN. :func:`windows` drops any window whose case
input or target touches one; nothing is interpolated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
CORRECTED = REPO / "data" / "corrected"
EXTERNAL = REPO / "data" / "external"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"

#: Minimum lag, in weeks, between a forecast's first target week and the latest row
#: each input may read. See the module docstring.
LAGS = {"cases": 1, "climate": 2, "ndvi": 0, "population": 0}

__all__ = ["LAGS", "CorrectedData", "load", "windows"]


@dataclass
class CorrectedData:
    names: list[str]
    week_start: pd.Series        # (T,) dates
    cases: np.ndarray            # (T, N) float, NaN where no report exists
    missing: np.ndarray          # (T,) bool
    climate: np.ndarray          # (T, N, 6) ERA5, same calendar week as the row
    climate_channels: list[str]
    ndvi: np.ndarray             # (T, N) as-of MODIS NDVI
    population: np.ndarray       # (T, N) previous-year official figure


def load() -> CorrectedData:
    names = sorted(json.loads(ADJ.read_text(encoding="utf-8")))
    index = pd.read_csv(CORRECTED / "rebuilt_index.csv", parse_dates=["week_start"])
    cases = np.load(CORRECTED / "rebuilt_cases.npy")[..., 0].astype(np.float64)
    missing = index["status"].eq("missing").to_numpy()
    if not np.array_equal(np.isnan(cases).any(axis=1), missing):
        raise ValueError("NaN case rows do not match the rows flagged missing")

    climate = np.load(CORRECTED / "rebuilt_climate_era5.npy").astype(np.float64)
    meta = json.loads((CORRECTED / "rebuilt_climate_era5.json").read_text(encoding="utf-8"))

    nd = pd.read_csv(EXTERNAL / "modis_ndvi_weekly_by_district.csv",
                     parse_dates=["week_start", "composite_available_from"])
    if (nd["composite_available_from"] > nd["week_start"]).any():
        raise ValueError("an NDVI row uses a composite that was not yet available")
    ndvi = nd.pivot(index="row", columns="district", values="ndvi")[names].to_numpy(dtype=np.float64)

    population = np.load(CORRECTED / "rebuilt_population.npy").astype(np.float64)
    src = pd.read_csv(CORRECTED / "rebuilt_population_sources.csv")
    for rec in src.itertuples(index=False):
        ref = 2012 if "Census 2012" in rec.figure else int(str(rec.figure).split()[-1])
        if ref >= rec.weeks_in_year:
            raise ValueError(f"population for {rec.weeks_in_year} uses a {ref} figure")

    for arr, label in ((climate, "climate"), (ndvi, "ndvi"), (population, "population")):
        if arr.shape[0] != len(index):
            raise ValueError(f"{label} has {arr.shape[0]} rows, cases have {len(index)}")
    return CorrectedData(names, index["week_start"], cases, missing, climate,
                         list(meta["channels"]), ndvi, population)


def _block(array: np.ndarray, i: int, window: int, lag: int) -> np.ndarray:
    """``window`` rows ending at ``i - lag`` inclusive -- never later."""
    end = i - lag
    start = end - window + 1
    if start < 0:
        raise IndexError("not enough history")
    out = array[start: end + 1]
    assert out.shape[0] == window
    return out


def windows(data: CorrectedData, window: int = 3, horizon: int = 3,
            covariate_window: int = 4) -> dict[str, np.ndarray]:
    """Forecast windows with every input sliced under :data:`LAGS`.

    Returns a dict of stacked arrays over the kept windows:
    ``origin`` (i), ``x_cases`` (K, N, window), ``y`` (K, N, horizon),
    ``x_climate`` (K, N, covariate_window, C), ``x_ndvi`` (K, N, covariate_window),
    ``population`` (K, N). A window is kept only if its case input
    ``[i - window, i)`` and target ``[i, i + horizon)`` contain no missing week.
    """
    T = data.cases.shape[0]
    first = max(window, LAGS["climate"] + covariate_window - 1, covariate_window - 1)
    keep, parts = [], {k: [] for k in ("x_cases", "y", "x_climate", "x_ndvi", "population")}
    for i in range(first, T - horizon + 1):
        if data.missing[i - window: i + horizon].any():
            continue
        keep.append(i)
        parts["x_cases"].append(_block(data.cases, i, window, LAGS["cases"]).T)
        parts["y"].append(data.cases[i: i + horizon].T)
        parts["x_climate"].append(np.moveaxis(
            _block(data.climate, i, covariate_window, LAGS["climate"]), 0, 1))
        parts["x_ndvi"].append(_block(data.ndvi, i, covariate_window, LAGS["ndvi"]).T)
        parts["population"].append(data.population[i - LAGS["population"]])
    out = {k: np.stack(v) for k, v in parts.items()}
    out["origin"] = np.asarray(keep)
    return out
