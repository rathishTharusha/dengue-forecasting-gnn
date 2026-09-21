"""No input may carry information from after a forecast is made.

Each test perturbs data at or after the point a forecast is allowed to see, and
checks the forecast's inputs do not move. They also check the opposite -- that
the latest *permitted* row really is used -- so a lag cannot silently grow.

Skips where the corrected data files are absent. Pure numpy/pandas.
"""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "corrected_data", REPO / "analysis" / "lib" / "corrected_data.py"
)
cd = importlib.util.module_from_spec(_SPEC)
sys.modules["corrected_data"] = cd
_SPEC.loader.exec_module(cd)

HAS_DATA = (REPO / "data" / "corrected" / "rebuilt_cases.npy").exists()
pytestmark = pytest.mark.skipif(not HAS_DATA, reason="corrected data not present")


@pytest.fixture(scope="module")
def data():
    return cd.load()


def _inputs_for(data, origin):
    w = cd.windows(data)
    k = int(np.flatnonzero(w["origin"] == origin)[0])
    return {key: w[key][k] for key in ("x_cases", "x_climate", "x_ndvi", "population")}


def _origin(data):
    w = cd.windows(data)
    return int(w["origin"][len(w["origin"]) // 2])


def test_nothing_at_or_after_the_origin_reaches_the_inputs(data):
    origin = _origin(data)
    before = _inputs_for(data, origin)
    poisoned = deepcopy(data)
    poisoned.cases[origin:] *= 1000.0
    poisoned.climate[origin:] += 50.0
    poisoned.ndvi[origin + 1 :] += 5.0
    poisoned.population[origin + 1 :] *= 3.0
    after = _inputs_for(poisoned, origin)
    for key in before:
        assert np.array_equal(before[key], after[key]), f"{key} reads data from the future"


def test_climate_skips_the_week_before_the_origin(data):
    """ERA5 for week i-1 is not complete at the start of week i."""
    origin = _origin(data)
    before = _inputs_for(data, origin)["x_climate"]
    poisoned = deepcopy(data)
    poisoned.climate[origin - 1] += 50.0
    assert np.array_equal(before, _inputs_for(poisoned, origin)["x_climate"])
    poisoned.climate[origin - 2] += 50.0
    assert not np.array_equal(before, _inputs_for(poisoned, origin)["x_climate"]), (
        "the latest permitted climate week is not being used"
    )


def test_cases_use_the_week_before_the_origin(data):
    origin = _origin(data)
    before = _inputs_for(data, origin)["x_cases"]
    poisoned = deepcopy(data)
    poisoned.cases[origin - 1] += 1.0
    assert not np.array_equal(before, _inputs_for(poisoned, origin)["x_cases"])


def test_no_window_reads_a_missing_week(data):
    w = cd.windows(data)
    assert not np.isnan(w["x_cases"]).any()
    assert not np.isnan(w["y"]).any()
    for i in w["origin"]:
        assert not data.missing[i - 3 : i + 3].any()


def test_ndvi_rows_use_only_composites_already_available():
    nd = pd.read_csv(
        REPO / "data" / "external" / "modis_ndvi_weekly_by_district.csv",
        parse_dates=["week_start", "composite_available_from"],
    )
    assert (nd["composite_available_from"] <= nd["week_start"]).all()


def test_population_uses_only_earlier_years():
    src = pd.read_csv(REPO / "data" / "corrected" / "rebuilt_population_sources.csv")
    for rec in src.itertuples(index=False):
        ref = 2012 if "Census 2012" in rec.figure else int(str(rec.figure).split()[-1])
        assert ref < rec.weeks_in_year


def test_nothing_is_filled(data):
    """Missing weeks stay NaN; no value was invented for them."""
    assert data.missing.sum() > 0
    assert np.isnan(data.cases[data.missing]).all()
