"""Tests for the biological envelope and spatial physics loss functions."""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))

import physics_loss as ploss  # noqa: E402


def test_envelope_loss_zero_when_within_bounds():
    # History with 10 cases at t-1
    history = torch.ones(2, 5, 9) * 10.0
    # Predictions growing reasonably: 12, 14, 16 (growth ~ 0.15 << 2.38)
    pred = torch.tensor([[[12.0, 14.0, 16.0]] * 5] * 2)
    loss = ploss.biological_envelope_loss(pred, history, r_max=2.3884, gamma_week=2.3026)
    assert loss.item() == 0.0, "Loss must be zero when growth is within bounds"


def test_envelope_loss_penalizes_impossible_growth():
    history = torch.ones(2, 5, 9) * 10.0
    # Prediction jumps from 10 to 1000 (log growth ~ 4.6 >> 2.3884)
    pred = torch.tensor([[[1000.0, 1000.0, 1000.0]] * 5] * 2)
    loss = ploss.biological_envelope_loss(pred, history, r_max=2.3884, gamma_week=2.3026)
    assert loss.item() > 0.0, "Loss must penalize growth exceeding r_max"


def test_envelope_loss_penalizes_impossible_decay():
    history = torch.ones(2, 5, 9) * 1000.0
    # Prediction drops from 1000 to 0.01 in one week (log drop > 11 >> 2.3026)
    pred = torch.tensor([[[0.01, 0.01, 0.01]] * 5] * 2)
    loss = ploss.biological_envelope_loss(pred, history, r_max=2.3884, gamma_week=2.3026)
    assert loss.item() > 0.0, "Loss must penalize decay exceeding clearance rate"


def test_envelope_is_inert_at_the_growth_our_models_actually_produce():
    """The reason EXP-014 retired the ceiling, kept as a regression test.

    The verified architectures predict weekly log-growth with a maximum of 0.31
    against a ceiling of 2.3884. A constraint that never activates contributes
    exactly zero gradient, so an arm carrying it must be numerically identical to
    its own baseline -- which is what the sweep observed on STGAT and A3TGCN.
    """
    history = torch.ones(2, 25, 9) * 40.0
    # Growth of 0.31 in log1p space, the observed maximum for those architectures.
    step = float(torch.expm1(torch.log1p(torch.tensor(40.0)) + 0.31))
    pred = torch.full((2, 25, 3), step)
    loss = ploss.biological_envelope_loss(pred, history)
    assert loss.item() == 0.0


def test_normalized_smoothness_zero_when_identical_relative_rates():
    adj = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    scales = torch.tensor([100.0, 10.0])  # District 0 is 10x larger than District 1
    # Both districts have 2.0x their baseline rate: 200 and 20
    pred = torch.tensor([[[200.0, 200.0, 200.0], [20.0, 20.0, 20.0]]])
    loss = ploss.normalized_smoothness_loss(pred, adj, scales)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), (
        "Normalized smoothness must be 0 for identical relative incidence"
    )


def test_log_smoothness_zero_when_identical_relative_rates():
    """The log variant must share the ratio form's fixed point.

    If the two disagreed about what "no spatial disparity" means, swapping one for
    the other would change the target rather than the conditioning.
    """
    adj = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    scales = torch.tensor([100.0, 10.0])
    pred = torch.tensor([[[200.0, 200.0, 200.0], [20.0, 20.0, 20.0]]])
    # log1p(200) - log1p(100) = 0.6882; log1p(20) - log1p(10) = 0.6539. Close but
    # not equal, because log1p is not scale-equivariant at small counts -- so the
    # fixed point is approximate at this magnitude and exact as counts grow.
    assert ploss.log_smoothness_loss(pred, adj, scales).item() < 1e-2

    big = torch.tensor([[[20000.0] * 3, [2000.0] * 3]])
    big_scales = torch.tensor([10000.0, 1000.0])
    assert ploss.log_smoothness_loss(big, adj, big_scales).item() < 1e-4


def test_log_smoothness_does_not_concentrate_on_outbreaks():
    """The property the variant exists for.

    Averaged over the observed record the ratio penalty is 85.6x larger on weeks
    containing an outbreak than on quiet weeks, against 1.1x for the log form. This
    test uses a single extreme pair rather than the whole record -- one quiet week
    at 2x baseline, one outbreak week at 20x -- so the absolute numbers are much
    larger for both. What must hold in either setting is the *relationship*: the
    ratio form grows quadratically in outbreak magnitude while the log form grows
    logarithmically, so its concentration is smaller by an order of magnitude.
    """
    adj = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    scales = torch.tensor([100.0, 100.0])
    quiet = torch.tensor([[[200.0], [100.0]]])       # 2x vs 1x baseline
    outbreak = torch.tensor([[[2000.0], [100.0]]])   # 20x vs 1x baseline

    ratio_conc = (ploss.normalized_smoothness_loss(outbreak, adj, scales)
                  / ploss.normalized_smoothness_loss(quiet, adj, scales)).item()
    log_conc = (ploss.log_smoothness_loss(outbreak, adj, scales)
                / ploss.log_smoothness_loss(quiet, adj, scales)).item()
    assert ratio_conc > 100.0, "ratio form should blow up on the outbreak week"
    assert log_conc < ratio_conc / 10.0, (
        f"log concentration {log_conc:.1f} should be an order of magnitude below "
        f"the ratio form's {ratio_conc:.1f}"
    )


def test_both_smoothness_forms_pass_gradients():
    adj = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    scales = torch.tensor([100.0, 10.0])
    for fn in (ploss.normalized_smoothness_loss, ploss.log_smoothness_loss):
        pred = torch.tensor([[[200.0, 150.0, 90.0], [20.0, 40.0, 5.0]]]).requires_grad_(True)
        fn(pred, adj, scales).backward()
        assert pred.grad.abs().sum().item() > 0, f"{fn.__name__} must pass gradients"


def test_nonnegativity_loss():
    pred_pos = torch.tensor([[[10.0, 20.0]]])
    assert ploss.nonnegativity_loss(pred_pos).item() == 0.0
    pred_neg = torch.tensor([[[-5.0, 10.0]]])
    assert ploss.nonnegativity_loss(pred_neg).item() > 0.0


def test_asymmetric_outbreak_loss():
    target = torch.tensor([[[100.0]]])
    under_pred = torch.tensor([[[90.0]]])  # error = 10, under-prediction
    over_pred = torch.tensor([[[110.0]]])  # error = 10, over-prediction
    l_under = ploss.asymmetric_outbreak_loss(under_pred, target, alpha=0.5)
    l_over = ploss.asymmetric_outbreak_loss(over_pred, target, alpha=0.5)
    assert l_under > l_over, "Under-prediction must incur higher penalty than over-prediction"
