"""The climate lag-block input path must not read a week closer than the ERA5 delay.

CLAUDE.md: any new input path needs its own leakage test. ``core.climate_blocks``
(docs/CLIMATE_PLAN.md) is one. For a validation window, every climate value
from one week before its origin onward -- lag 1, lag 0 and the future -- is
perturbed; the window's features must not move, raw or as anomalies. A control
checks that perturbing lag 2 *does* move them, so the test cannot pass on a dead
path.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "seirgnn2"))
sys.path.insert(0, str(REPO / "analysis" / "lib"))

import core  # noqa: E402
import corrected_data as cd  # noqa: E402

BLOCKS = ((2, 5), (6, 9), (10, 13))


@pytest.fixture(scope="module")
def setup():
    data = cd.load()
    fold = next(f for f in core.build_folds(data.cases, data.missing) if f.origin == 0.70)
    origin = int(fold.idx["val"][5])
    # Exactly what train.run_fold does before building tensors with lag blocks.
    one = core.with_history(
        dataclasses.replace(fold, idx={**fold.idx, "val": np.array([origin])}),
        max(b for _, b in BLOCKS),
    )
    return data, one, origin


def _feats(data, fold, anom):
    return core.build_tensors(data, fold, "val", False, False, False, BLOCKS, anom)["x"].numpy()


@pytest.mark.parametrize("anom", [False, True])
def test_blocks_ignore_lag_below_two(setup, anom):
    data, fold, origin = setup
    tampered = data.climate.copy()
    tampered[origin - 1 :] = tampered[origin - 1 :] * 3.0 + 50.0
    after = _feats(dataclasses.replace(data, climate=tampered), fold, anom)
    np.testing.assert_array_equal(_feats(data, fold, anom), after)


def test_blocks_do_read_lag_two(setup):
    data, fold, origin = setup
    tampered = data.climate.copy()
    tampered[origin - 2] = tampered[origin - 2] + 100.0
    after = _feats(dataclasses.replace(data, climate=tampered), fold, False)
    assert not np.array_equal(_feats(data, fold, False), after)


def test_short_history_is_refused_not_wrapped(setup):
    """A negative index would wrap to the end of the series -- the test period."""
    data, fold, _ = setup
    with pytest.raises(ValueError, match="no climate at lag"):
        core.climate_blocks(data, np.array([7]), BLOCKS, data.climate)
    trimmed = core.with_history(fold, 13)
    assert int(trimmed.idx["train"].min()) - 13 >= 0
    assert np.array_equal(trimmed.idx["val"], fold.idx["val"])


def test_a_lag_below_the_release_delay_is_refused(setup):
    data, fold, _ = setup
    with pytest.raises(ValueError):
        core.climate_blocks(data, fold.idx["val"], ((1, 4),), data.climate)
