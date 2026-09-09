"""Tests for dengue_gnn.mechanistic.

These check the properties that make these constraints different from the spatial
smoothness penalty that failed in Stage 2b -- in particular that the band loss
does NOT push toward a flat forecast, which is why it should not fight the
adaptive graph.
"""

import pytest

torch = pytest.importorskip("torch")

from dengue_gnn.mechanistic import (  # noqa: E402
    MAX_WEEKLY_LOG_GROWTH,
    curriculum_weight,
    growth_band_loss,
    growth_smoothness_loss,
    implied_log_growth,
    mechanistic_regularisation,
)


def _flat(value: float, h: int = 3) -> torch.Tensor:
    return torch.full((1, 1, h), value)


# --------------------------------------------------------------------------
# implied growth
# --------------------------------------------------------------------------


def test_growth_is_zero_for_a_constant_forecast():
    pred = _flat(100.0)
    last = torch.tensor([[100.0]])
    assert torch.allclose(implied_log_growth(pred, last), torch.zeros(1, 1, 3), atol=1e-6)


def test_growth_matches_hand_computation():
    pred = torch.tensor([[[e - 1 for e in (torch.e**1, torch.e**2, torch.e**3)]]])
    last = torch.tensor([[0.0]])  # log1p(0) = 0
    g = implied_log_growth(pred, last)
    assert torch.allclose(g, torch.tensor([[[1.0, 1.0, 1.0]]]), atol=1e-5)


def test_growth_rejects_wrong_rank():
    with pytest.raises(ValueError, match="batch, n_nodes, horizon"):
        implied_log_growth(torch.zeros(3, 3), torch.zeros(3))


def test_growth_rejects_mismatched_anchor():
    with pytest.raises(ValueError, match="last_observed must be"):
        implied_log_growth(torch.zeros(2, 4, 3), torch.zeros(2, 5))


# --------------------------------------------------------------------------
# band loss
# --------------------------------------------------------------------------


def test_band_is_zero_inside_the_plausible_range():
    """A forecast growing slowly is not penalised at all."""
    last = torch.tensor([[100.0]])
    pred = torch.tensor([[[110.0, 120.0, 130.0]]])  # ~0.09/wk, well inside
    assert growth_band_loss(pred, last) == pytest.approx(0.0)


def test_band_penalises_implausible_explosions():
    last = torch.tensor([[10.0]])
    pred = torch.tensor([[[2000.0, 2100.0, 2200.0]]])  # first step is a huge jump
    assert growth_band_loss(pred, last) > 0.0


def test_band_does_not_reward_flatness():
    """The property that distinguishes this from the Stage-2b smoothness penalty.

    A flat forecast and a moderately varying one are both unpenalised, so the
    term has no incentive to collapse structure -- unlike a Laplacian smoothness
    prior, which is minimised by uniformity and therefore fought the adaptive
    graph.
    """
    last = torch.tensor([[100.0]])
    flat = torch.tensor([[[100.0, 100.0, 100.0]]])
    varied = torch.tensor([[[130.0, 110.0, 145.0]]])
    assert growth_band_loss(flat, last) == pytest.approx(0.0)
    assert growth_band_loss(varied, last) == pytest.approx(0.0)


def test_band_threshold_is_respected():
    last = torch.tensor([[100.0]])
    # one step at exactly the ceiling -> no penalty; beyond it -> penalty
    at = torch.tensor([[[float(101 * torch.e**MAX_WEEKLY_LOG_GROWTH - 1)]]])
    assert growth_band_loss(at, last, MAX_WEEKLY_LOG_GROWTH) == pytest.approx(0.0, abs=1e-5)
    assert growth_band_loss(at, last, max_growth=0.1) > 0.0


# --------------------------------------------------------------------------
# smoothness of growth
# --------------------------------------------------------------------------


def test_growth_smoothness_zero_for_constant_growth():
    """Exponential growth has constant log-growth, so it is unpenalised."""
    last = torch.tensor([[500.0]])
    pred = torch.tensor([[[1000.0, 2000.0, 4000.0]]])
    assert float(growth_smoothness_loss(pred, last)) == pytest.approx(0.0, abs=1e-4)


def test_growth_smoothness_penalises_a_jump():
    last = torch.tensor([[100.0]])
    steady = torch.tensor([[[110.0, 121.0, 133.0]]])
    jumpy = torch.tensor([[[110.0, 900.0, 120.0]]])
    assert growth_smoothness_loss(jumpy, last) > growth_smoothness_loss(steady, last)


def test_growth_smoothness_handles_horizon_one():
    last = torch.tensor([[100.0]])
    out = growth_smoothness_loss(torch.tensor([[[110.0]]]), last)
    assert out == pytest.approx(0.0)


# --------------------------------------------------------------------------
# combined
# --------------------------------------------------------------------------


def test_lambda_zero_is_a_differentiable_zero():
    pred = (torch.rand(2, 3, 3) * 100).requires_grad_(True)  # leaf, or .grad stays None
    last = torch.rand(2, 3) * 100
    out = mechanistic_regularisation(pred, last, 0.0)
    assert float(out) == pytest.approx(0.0)
    out.backward()
    assert pred.grad is not None


def test_modes_select_terms():
    last = torch.tensor([[100.0]])
    jumpy = torch.tensor([[[110.0, 9000.0, 120.0]]])
    band = mechanistic_regularisation(jumpy, last, 1.0, mode="band")
    smooth = mechanistic_regularisation(jumpy, last, 1.0, mode="smooth")
    both = mechanistic_regularisation(jumpy, last, 1.0, mode="both")
    assert both == pytest.approx(float(band) + float(smooth), rel=1e-5)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        mechanistic_regularisation(torch.zeros(1, 1, 3), torch.zeros(1, 1), 1.0, mode="nope")


def test_regularisation_is_differentiable():
    pred = (torch.rand(2, 4, 3) * 100).requires_grad_(True)
    last = torch.rand(2, 4) * 100
    mechanistic_regularisation(pred, last, 0.5).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# --------------------------------------------------------------------------
# curriculum
# --------------------------------------------------------------------------


def test_curriculum_ramps_then_holds():
    assert curriculum_weight(0, 100, 1.0, warmup=0.5) == pytest.approx(0.0)
    assert curriculum_weight(25, 100, 1.0, warmup=0.5) == pytest.approx(0.5)
    assert curriculum_weight(50, 100, 1.0, warmup=0.5) == pytest.approx(1.0)
    assert curriculum_weight(99, 100, 1.0, warmup=0.5) == pytest.approx(1.0)


def test_curriculum_disabled_returns_target():
    assert curriculum_weight(0, 100, 0.3, warmup=0.0) == pytest.approx(0.3)


# --------------------------------------------------------------------------
# gradient-norm adaptive weighting (Wang, Teng & Perdikaris)
# --------------------------------------------------------------------------


def test_adaptive_weight_raises_a_weak_constraint():
    from torch import nn

    from dengue_gnn.mechanistic import adaptive_weight

    torch.manual_seed(0)
    m = nn.Linear(4, 3)
    x = torch.randn(8, 4)
    data = ((m(x) - 1) ** 2).mean()
    weak = (m(x) ** 2).mean() * 1e-4
    assert adaptive_weight(m, data, weak, current=1.0) > 1.0


def test_adaptive_weight_lowers_an_overpowering_constraint():
    from torch import nn

    from dengue_gnn.mechanistic import adaptive_weight

    torch.manual_seed(0)
    m = nn.Linear(4, 3)
    x = torch.randn(8, 4)
    data = ((m(x) - 1) ** 2).mean() * 1e-6
    strong = (m(x) ** 2).mean() * 1e6
    assert adaptive_weight(m, data, strong, current=1.0) < 1.0


def test_adaptive_weight_ignores_an_inert_constraint():
    """Our growth-band term contributed zero gradient in 5 of 24 runs. Dividing
    by that would send the weight to infinity."""
    from torch import nn

    from dengue_gnn.mechanistic import adaptive_weight

    m = nn.Linear(4, 3)
    x = torch.randn(8, 4)
    data = ((m(x) - 1) ** 2).mean()
    inert = (m(x) * 0).sum()
    assert adaptive_weight(m, data, inert, current=5.0) == 5.0
