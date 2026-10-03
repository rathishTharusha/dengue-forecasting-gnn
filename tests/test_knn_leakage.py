"""The k-NN analogue forecaster must not read any week at or after its origin.

CLAUDE.md: any new input path needs its own leakage test. ``seirgnn2/knn.py``
is one -- it reads training windows *and their targets* as analogues -- so this
perturbs every case value from each validation window's forecast origin onward
and checks the forecast for that window does not move.
"""

from __future__ import annotations

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
import knn  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    data = cd.load()
    fold = next(f for f in core.build_folds(data.cases, data.missing) if f.origin == 0.70)
    lib = knn._windows(data, knn.library_idx(fold), fold.window, core.HORIZON, with_target=True)
    return data, fold, lib


def test_forecast_ignores_the_future(setup):
    data, fold, lib = setup
    origin = int(fold.idx["val"][10])
    query_idx = np.array([origin])
    q = knn._windows(data, query_idx, fold.window, core.HORIZON, with_target=False)
    before = knn._predict(lib, q, 10, 0.5, 0.5)

    tampered = data.cases.copy()
    tampered[origin:] = tampered[origin:] * 7.0 + 123.0
    shifted = type(data)(**{**data.__dict__, "cases": tampered})
    q2 = knn._windows(shifted, query_idx, fold.window, core.HORIZON, with_target=False)
    after = knn._predict(lib, q2, 10, 0.5, 0.5)

    np.testing.assert_array_equal(before, after)


def test_no_analogue_target_reaches_a_query(setup):
    """Every analogue's last target week precedes the earliest validation origin."""
    _, fold, _ = setup
    lib = knn.library_idx(fold)
    assert set(lib) <= set(fold.idx["train"])
    assert int(lib.max()) + core.HORIZON - 1 < int(fold.idx["val"].min())
