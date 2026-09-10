"""Tests for the renewal-equation physics.

These guard the properties the design argument rests on. If any of them breaks,
the claim that this addresses the measured under-reaction stops being true, and
that is exactly the kind of silent drift EXP-014 caught too late.

``analysis/lib`` is not a package, so the module is loaded by path. CI installs
only numpy/pytest/ruff, but this module needs torch -- the tests skip cleanly
where it is absent, matching the convention in ``tests/test_mechanistic.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_SPEC = importlib.util.spec_from_file_location(
    "renewal", Path(__file__).resolve().parent.parent / "analysis" / "lib" / "renewal.py"
)
renewal = importlib.util.module_from_spec(_SPEC)
sys.modules["renewal"] = renewal
_SPEC.loader.exec_module(renewal)


class _Stages:
    """SEIR-SEI stage rates, per day, from Phaijoo & Gurung section 4."""

    nu_h, gamma_h, nu_v, mu_v = 0.1667, 0.328833, 0.1428, 0.25


def test_generation_interval_is_a_distribution():
    w = renewal.generation_interval(_Stages())
    assert w.shape == (renewal.KERNEL_WEEKS,)
    assert w.min() > 0
    assert w.sum() == pytest.approx(1.0)


def test_generation_interval_matches_the_stage_durations():
    """Mean GI should be near the sum of mean stage durations, in weeks.

    The four stages sum to 1/0.1667 + 1/0.328833 + 1/0.1428 + 1/0.25 = 22.0 days
    = 3.15 weeks. Ceiling-binning to whole weeks pushes the discrete mean a little
    above that; anything far from it means the convolution is wrong.
    """
    w = renewal.generation_interval(_Stages())
    mean_weeks = float((np.arange(1, len(w) + 1) * w).sum())
    exact_days = sum(1 / r for r in (_Stages.nu_h, _Stages.gamma_h,
                                     _Stages.nu_v, _Stages.mu_v))
    assert exact_days / 7 < mean_weeks < exact_days / 7 + 1.0


def test_r_equal_one_holds_a_flat_series_flat():
    """The fixed point: constant history, R=1, forecast stays constant.

    ``w`` sums to 1, so the force of a flat history is that same constant.
    """
    w = torch.tensor(renewal.generation_interval(_Stages()), dtype=torch.float32)
    history = torch.full((2, 25, renewal.KERNEL_WEEKS), 40.0)
    out = renewal.renewal_forecast(torch.zeros(2, 25, 3), history, w)
    assert torch.allclose(out, torch.full_like(out, 40.0), atol=1e-3)


def test_growth_compounds_across_the_horizon():
    """R>1 must produce accelerating growth, not a constant offset.

    This is the property the residual-over-persistence baselines lack, and the
    reason for the decoder.
    """
    w = torch.tensor(renewal.generation_interval(_Stages()), dtype=torch.float32)
    history = torch.full((1, 1, renewal.KERNEL_WEEKS), 100.0)
    out = renewal.renewal_forecast(torch.full((1, 1, 3), 0.7), history, w)[0, 0]
    assert out[0] < out[1] < out[2]
    assert (out[2] - out[1]) > (out[1] - out[0])


def test_forecast_is_never_negative():
    w = torch.tensor(renewal.generation_interval(_Stages()), dtype=torch.float32)
    history = torch.rand(4, 25, renewal.KERNEL_WEEKS) * 500
    out = renewal.renewal_forecast(torch.randn(4, 25, 3) * 5, history, w)
    assert (out >= 0).all()


def test_log_r_bounds_clamp_without_killing_the_gradient():
    """The clamp must not bind on realistic outbreaks.

    The observed p99 of back-solved R is 5.36; the upper bound is exp(2.5)=12.2.
    A model asked for R=5 must get R=5, or the bound reproduces the EXP-014 error
    of forbidding real dynamics.
    """
    w = torch.tensor(renewal.generation_interval(_Stages()), dtype=torch.float32)
    history = torch.full((1, 1, renewal.KERNEL_WEEKS), 10.0)
    log_r = torch.full((1, 1, 1), float(np.log(5.0)), requires_grad=True)
    out = renewal.renewal_forecast(log_r, history, w)
    assert out.item() == pytest.approx(50.0, rel=1e-4)
    out.backward()
    assert log_r.grad.abs().item() > 0


def test_estimate_r_recovers_a_known_multiplier():
    """A history built with a constant R must back-solve to that R."""
    w_np = renewal.generation_interval(_Stages())
    w = torch.tensor(w_np, dtype=torch.float32)
    series = [50.0] * renewal.KERNEL_WEEKS
    for _ in range(5):
        force = sum(w_np[lag] * series[-(lag + 1)] for lag in range(renewal.KERNEL_WEEKS))
        series.append(1.6 * force)
    history = torch.tensor(series[-renewal.HISTORY_WEEKS:],
                           dtype=torch.float32).reshape(1, 1, -1)
    assert renewal.estimate_r(history, w).item() == pytest.approx(1.6, rel=0.05)


def test_estimate_r_falls_back_to_one_on_an_empty_district():
    """No recent transmission means no information, not R=0."""
    w = torch.tensor(renewal.generation_interval(_Stages()), dtype=torch.float32)
    history = torch.zeros(1, 1, renewal.HISTORY_WEEKS)
    assert renewal.estimate_r(history, w).item() == pytest.approx(1.0)


def test_penalty_is_zero_on_a_renewal_consistent_forecast():
    """The constraint must not charge for obeying it."""
    w_np = renewal.generation_interval(_Stages())
    w = torch.tensor(w_np, dtype=torch.float32)
    series = [50.0] * renewal.KERNEL_WEEKS
    for _ in range(5):
        force = sum(w_np[lag] * series[-(lag + 1)] for lag in range(renewal.KERNEL_WEEKS))
        series.append(1.6 * force)
    history = torch.tensor(series[-renewal.HISTORY_WEEKS:],
                           dtype=torch.float32).reshape(1, 1, -1)
    r = renewal.estimate_r(history, w)
    consistent = renewal.renewal_forecast(
        torch.log(r).reshape(1, 1, 1).expand(1, 1, 3).contiguous(),
        history[..., -renewal.KERNEL_WEEKS:], w)
    assert renewal.renewal_penalty(consistent, history, w).item() < 1e-4


def test_penalty_pushes_a_flat_forecast_upward_when_r_exceeds_one():
    """The measured failure mode is under-reaction, so the gradient must point up.

    A ceiling could only ever push down, which is why EXP-014 found it inert
    against a model that already under-predicts.
    """
    w_np = renewal.generation_interval(_Stages())
    w = torch.tensor(w_np, dtype=torch.float32)
    series = [50.0] * renewal.KERNEL_WEEKS
    for _ in range(5):
        force = sum(w_np[lag] * series[-(lag + 1)] for lag in range(renewal.KERNEL_WEEKS))
        series.append(2.0 * force)
    history = torch.tensor(series[-renewal.HISTORY_WEEKS:],
                           dtype=torch.float32).reshape(1, 1, -1)

    flat = torch.full((1, 1, 3), float(series[-1]), requires_grad=True)
    renewal.renewal_penalty(flat, history, w).backward()
    # Descending the penalty means stepping against the gradient, so a negative
    # gradient on the first step is an instruction to forecast more cases.
    assert flat.grad[0, 0, 0].item() < 0


def test_decoder_is_identity_on_r_when_untrained():
    """A zero-output backbone must give R=1, not a collapse to zero."""
    w = torch.tensor(renewal.generation_interval(_Stages()), dtype=torch.float32)
    decoder = renewal.RenewalDecoder(w)
    history = torch.full((3, 25, renewal.KERNEL_WEEKS), 12.0)
    out = decoder(torch.zeros(3, 25, 3), history)
    assert torch.allclose(out, torch.full_like(out, 12.0), atol=1e-3)


def test_decoder_gradient_reaches_the_backbone_output():
    w = torch.tensor(renewal.generation_interval(_Stages()), dtype=torch.float32)
    decoder = renewal.RenewalDecoder(w)
    history = torch.full((2, 25, renewal.KERNEL_WEEKS), 30.0)
    raw = torch.zeros(2, 25, 3, requires_grad=True)
    decoder(raw, history).sum().backward()
    assert raw.grad.abs().sum().item() > 0
