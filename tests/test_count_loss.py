"""Tests for the negative-binomial count head.

These pin the properties the design argument rests on. The claim is that this
head removes the log-space retransformation bias *by construction* rather than
correcting it afterwards, so the tests that matter are the ones about what the
likelihood is optimised at and what the point forecast means.

``analysis/lib`` is not a package, so the module is loaded by path. CI installs
only numpy/pytest/ruff, but this module needs torch, so the whole file skips
cleanly where torch is absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SPEC = importlib.util.spec_from_file_location(
    "count_loss", Path(__file__).resolve().parent.parent / "analysis" / "lib" / "count_loss.py"
)
count_loss = importlib.util.module_from_spec(_SPEC)
sys.modules["count_loss"] = count_loss
_SPEC.loader.exec_module(count_loss)


def test_nll_is_minimised_at_the_sample_mean():
    """The property the whole head exists for.

    Fitting mu to a sample minimises the NB2 likelihood at the sample **mean**.
    A log1p regression fit to the same sample would land near its median, which
    on this heavy-tailed target is far lower -- that gap is the -12.9 outbreak
    bias the pipeline measures.
    """
    sample = torch.tensor([1.0, 2.0, 3.0, 5.0, 400.0])  # median 3, mean 82.2
    alpha = torch.full_like(sample, 0.5)

    # One shared mu, as in a fit: with a free mu per observation each element
    # would simply chase its own target and the gradient would say nothing.
    mu_scalar = torch.tensor(float(sample.mean()), requires_grad=True)
    loss = count_loss.nb_nll(mu_scalar.expand_as(sample), alpha, sample)
    (grad,) = torch.autograd.grad(loss, mu_scalar)
    assert grad.abs() < 1e-4, f"NLL gradient should vanish at the sample mean, got {grad}"

    at_median = count_loss.nb_nll(torch.full_like(sample, 3.0), alpha, sample)
    assert at_median > loss, "the median must be a worse fit than the mean"


def test_point_forecast_needs_no_retransformation():
    """``nb_mean`` is the identity on mu, and mu is already on the count scale."""
    mu = torch.tensor([0.5, 10.0, 900.0])
    assert torch.equal(count_loss.nb_mean(mu), mu)


def test_split_preserves_persistence_when_the_residual_is_zero():
    """A zero residual must reproduce last week's value exactly.

    This is the residual-over-persistence contract the rest of the project
    depends on; if the parameterisation broke it, the head would be starting
    from somewhere other than the strong baseline.
    """
    last = torch.tensor([0.0, 7.0, 130.0])
    persistence_log = torch.log1p(last)
    raw = torch.zeros(3, 2)
    mu, _ = count_loss.split_nb(raw, persistence_log)
    assert torch.allclose(mu, last.clamp_min(1e-6), atol=1e-4)


def test_dispersion_is_bounded_both_ways():
    """An unbounded dispersion lets the model buy likelihood with uncertainty."""
    raw = torch.stack([torch.zeros(5), torch.tensor([-1e4, -10.0, 0.0, 10.0, 1e4])], dim=-1)
    _, alpha = count_loss.split_nb(raw, torch.zeros(5))
    # float32, so the bounds are exact only to within its resolution.
    assert float(alpha.min()) == pytest.approx(count_loss.ALPHA_MIN, rel=1e-5)
    assert float(alpha.min()) >= count_loss.ALPHA_MIN * (1 - 1e-5)
    assert float(alpha.max()) <= count_loss.ALPHA_MAX * (1 + 1e-5)


def test_overdispersion_beats_poisson_on_overdispersed_data():
    """NB2 must prefer alpha > 0 where the variance exceeds the mean.

    Poisson is the alpha -> 0 limit. On a sample whose variance is far above its
    mean, a fitted dispersion has to improve the likelihood, or the head is no
    better specified than the Poisson it replaces.
    """
    sample = torch.tensor([0.0, 1.0, 2.0, 3.0, 200.0])
    mu = torch.full_like(sample, float(sample.mean()))
    near_poisson = count_loss.nb_nll(mu, torch.full_like(sample, count_loss.ALPHA_MIN), sample)
    dispersed = count_loss.nb_nll(mu, torch.full_like(sample, 1.5), sample)
    assert dispersed < near_poisson


def test_nll_is_finite_on_zeros_and_extremes():
    """9.7% of this target is zero and its max is 2631; neither may produce NaN."""
    target = torch.tensor([0.0, 0.0, 1.0, 2631.0])
    mu = torch.tensor([1e-6, 5.0, 1.0, 1200.0])
    alpha = torch.tensor([count_loss.ALPHA_MIN, 1.0, 0.5, count_loss.ALPHA_MAX])
    loss = count_loss.nb_nll(mu, alpha, target)
    assert torch.isfinite(loss)
