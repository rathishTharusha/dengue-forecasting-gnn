"""The rebuilt week index must be one weekly grid.

Every covariate (ERA5 climate, NDVI, policy, seasonal features) is joined to the cases
through ``week_start``. Eight rows once carried a date with the right year and the wrong
month, so those weeks read another week's weather.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis" / "_build"))

from build_corrected_cases import snap_to_weekly_grid  # noqa: E402


def test_every_consecutive_pair_of_week_starts_is_seven_days_apart():
    idx = pd.read_csv(REPO / "data" / "corrected" / "rebuilt_index.csv", parse_dates=["week_start"])
    gaps = idx["week_start"].diff().dropna().dt.days
    assert (gaps == 7).all(), idx.loc[
        gaps[gaps != 7].index, ["row", "year", "week_no", "week_start"]
    ]
    assert idx["week_start"].iloc[0] == pd.Timestamp("2013-06-15")
    assert idx["week_start"].iloc[-1] == pd.Timestamp("2024-02-24")
    assert (idx["row"] == range(len(idx))).all()


def test_snap_repairs_a_date_with_the_right_year_and_the_wrong_month():
    good = pd.Series(pd.date_range("2013-06-15", periods=60, freq="7D"))
    bad = good.copy()
    bad.iloc[0] = pd.Timestamp("2013-05-15")  # the first build's row 0
    bad.iloc[30] = pd.Timestamp("2013-12-21")
    snapped = snap_to_weekly_grid(bad)
    assert (snapped == good).all()
