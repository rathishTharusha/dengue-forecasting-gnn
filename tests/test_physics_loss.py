"""Tests for the biological envelope and spatial physics loss functions."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))

import physics_loss as ploss


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


def test_normalized_smoothness_zero_when_identical_relative_rates():
    adj = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    scales = torch.tensor([100.0, 10.0])  # District 0 is 10x larger than District 1
    # Both districts have 2.0x their baseline rate: 200 and 20
    pred = torch.tensor([[[200.0, 200.0, 200.0], [20.0, 20.0, 20.0]]])
    loss = ploss.normalized_smoothness_loss(pred, adj, scales)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), "Normalized smoothness must be 0 for identical relative incidence"


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
