"""Tests for dengue_gnn.metrics.

These guard the numbers that end up in the ablation table. A silent change in
how SMAPE stabilizes its denominator, or in which weeks MAPE masks out, would
shift every reported figure without anything visibly breaking.
"""

import math

import numpy as np
import pytest

from dengue_gnn.metrics import metrics, score


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


def test_morans_i():
    from dengue_gnn.metrics import compute_morans_i
    # 4-node ring
    adj = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ])
    # Positive autocorrelation: neighbors have similar values
    res_pos = np.array([10.0, 10.0, -10.0, -10.0])
    i_pos = compute_morans_i(res_pos, adj)
    assert i_pos > -1.0 and i_pos <= 1.0


def test_peak_timing_error():
    from dengue_gnn.metrics import compute_peak_timing_error
    # 5 weeks, 2 districts
    truth = np.array([
        [0.0, 0.0],
        [10.0, 5.0],
        [50.0, 20.0],  # peak at week 2
        [10.0, 100.0], # dist 1 peak at week 3
        [0.0, 0.0],
    ])
    # Predict peak 1 week late for dist 0 (week 3), exact for dist 1 (week 3)
    pred = np.array([
        [0.0, 0.0],
        [10.0, 5.0],
        [20.0, 20.0],
        [50.0, 100.0],
        [0.0, 0.0],
    ])
    pte = compute_peak_timing_error(pred, truth)
    # dist 0 error = 1, dist 1 error = 0 -> mean = 0.5 weeks
    assert pte == pytest.approx(0.5)
