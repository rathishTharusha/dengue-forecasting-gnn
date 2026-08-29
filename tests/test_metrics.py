"""Tests for dengue_gnn.metrics.

These guard the numbers that end up in the ablation table. A silent change in
how SMAPE stabilizes its denominator, or in which weeks MAPE masks out, would
shift every reported figure without anything visibly breaking.

The peak_week_error tests exist because the original implementation was wrong in
a way that unit tests on identical arrays could not catch: it scored 0.0 for a
perfect forecast, which looks correct, while also scoring 0.0 for a forecast
shifted by ten weeks. The regression test below shifts the series on purpose.
"""

import math

import numpy as np
import pytest

from dengue_gnn.metrics import (
    metrics,
    peak_week_error,
    peak_week_error_by_horizon,
    score,
    smape,
)

# --------------------------------------------------------------------------
# elementwise metrics
# --------------------------------------------------------------------------


def test_perfect_forecast_scores_zero():
    y = np.array([0.0, 13.0, 250.0, 2631.0])
    out = score(y, y)
    assert out["RMSE"] == pytest.approx(0.0)
    assert out["MAE"] == pytest.approx(0.0)
    assert out["SMAPE"] == pytest.approx(0.0)
    assert out["MAPE"] == pytest.approx(0.0)


def test_known_values():
    pred = np.array([2.0, 4.0])
    truth = np.array([4.0, 4.0])
    out = score(pred, truth)
    assert out["RMSE"] == pytest.approx(math.sqrt(2.0))
    assert out["MAE"] == pytest.approx(1.0)
    # SMAPE: mean(2*2/(2+4), 0) * 100 = 33.33...
    assert out["SMAPE"] == pytest.approx(100 * (2 * 2 / 6) / 2, rel=1e-4)
    # MAPE: mean(2/4, 0/4) * 100 = 25
    assert out["MAPE"] == pytest.approx(25.0, rel=1e-4)


def test_score_has_no_peak_metric():
    """Peak timing is not elementwise and must not live in a shape-agnostic API.

    This is the design fix for the original bug, so it is asserted explicitly.
    """
    out = score(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert set(out) == {"RMSE", "MAE", "SMAPE", "MAPE"}


def test_standalone_smape():
    pred = np.array([2.0, 4.0])
    truth = np.array([4.0, 4.0])
    assert smape(pred, truth) == pytest.approx(100 * (2 * 2 / 6) / 2, rel=1e-4)


def test_mape_masks_zero_weeks():
    """9.7% of district-weeks are zero; they must not blow up MAPE."""
    pred = np.array([5.0, 10.0])
    truth = np.array([0.0, 20.0])
    out = score(pred, truth)
    assert math.isfinite(out["MAPE"])
    assert out["MAPE"] == pytest.approx(50.0)  # only the second element counts


def test_mape_is_nan_when_all_weeks_masked():
    out = score(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    assert math.isnan(out["MAPE"])


def test_smape_bounded_at_200_percent():
    out = score(np.array([0.0]), np.array([100.0]))
    assert out["SMAPE"] == pytest.approx(200.0, rel=1e-4)


def test_smape_finite_when_both_zero():
    """Both zero is a correct forecast; the eps must keep it from being nan."""
    out = score(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert out["SMAPE"] == pytest.approx(0.0)


def test_metrics_splits_by_horizon():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 100, size=(7, 25, 3))
    pred = truth.copy()
    pred[..., 2] += 10.0  # only the 3-week horizon is wrong

    out = metrics(pred, truth)
    assert set(out) == {"h1", "h2", "h3", "overall"}
    assert out["h1"]["MAE"] == pytest.approx(0.0)
    assert out["h2"]["MAE"] == pytest.approx(0.0)
    assert out["h3"]["MAE"] == pytest.approx(10.0)
    assert out["overall"]["MAE"] == pytest.approx(10.0 / 3)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        score(np.zeros(3), np.zeros(4))


def test_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        score(np.array([]), np.array([]))


# --------------------------------------------------------------------------
# peak timing
# --------------------------------------------------------------------------


def _outbreak(peak_week: int, n_weeks: int = 40, height: float = 500.0) -> np.ndarray:
    """A single-district gaussian outbreak curve peaking at ``peak_week``."""
    w = np.arange(n_weeks)
    return np.exp(-((w - peak_week) ** 2) / 18.0) * height


def test_peak_week_error_detects_a_shift():
    """REGRESSION: the original implementation returned 0.0 for this case.

    Observed peak at week 20, forecast peak at week 23. The only correct
    answer is 3.0 weeks.
    """
    truth = _outbreak(20).reshape(-1, 1)
    pred = _outbreak(23).reshape(-1, 1)
    assert peak_week_error(pred, truth) == pytest.approx(3.0)


def test_peak_week_error_zero_for_perfect_timing():
    truth = _outbreak(20).reshape(-1, 1)
    pred = _outbreak(20) * 0.4  # wrong magnitude, right timing
    assert peak_week_error(pred.reshape(-1, 1), truth) == pytest.approx(0.0)


def test_peak_week_error_averages_over_districts():
    truth = np.stack([_outbreak(20), _outbreak(10)], axis=1)
    pred = np.stack([_outbreak(22), _outbreak(10)], axis=1)  # errors 2 and 0
    assert peak_week_error(pred, truth) == pytest.approx(1.0)


def test_peak_week_error_skips_districts_with_no_outbreak():
    """A flat district has an argmax, but it is noise, not a peak."""
    flat = np.full(40, 0.5)
    truth = np.stack([_outbreak(20), flat], axis=1)
    pred = np.stack([_outbreak(24), flat[::-1]], axis=1)
    # only the real outbreak counts -> 4.0, not the mean of 4 and something random
    assert peak_week_error(pred, truth) == pytest.approx(4.0)


def test_peak_week_error_nan_when_no_district_qualifies():
    flat = np.full((40, 2), 0.5)
    assert math.isnan(peak_week_error(flat, flat))


def test_peak_week_error_rejects_wrong_rank():
    """The type signature is the fix: misuse must raise, not silently compute."""
    bad = np.zeros((10, 25, 3))
    with pytest.raises(ValueError, match="2D"):
        peak_week_error(bad, bad)

    with pytest.raises(ValueError, match="2D"):
        peak_week_error(np.zeros(10), np.zeros(10))


def test_peak_week_error_by_horizon():
    truth = np.stack([_outbreak(20)] * 3, axis=-1).reshape(40, 1, 3)
    pred = np.stack([_outbreak(20), _outbreak(22), _outbreak(25)], axis=-1).reshape(40, 1, 3)
    out = peak_week_error_by_horizon(pred, truth)
    assert out["h1"] == pytest.approx(0.0)
    assert out["h2"] == pytest.approx(2.0)
    assert out["h3"] == pytest.approx(5.0)


def test_peak_week_error_by_horizon_rejects_wrong_rank():
    bad = np.zeros((40, 3))
    with pytest.raises(ValueError, match="n_weeks, n_nodes, horizon"):
        peak_week_error_by_horizon(bad, bad)
