"""Unit tests for the SEIR-LSTM model and simulator interface."""

import pytest
import torch

from seir_lstm import SEIRLSTMHead, step_seir_window


def test_seir_lstm_head_dimensions():
    model = SEIRLSTMHead(in_dim=8, hidden_dim=32, lambda_max=2.0 / 7.0)
    x = torch.randn(2, 25, 4, 8)  # Batch 2, 25 districts, 4 timesteps, 8 features
    lam = model(x)
    assert lam.shape == (2, 25)
    assert (lam >= 0.0).all()
    assert (lam <= 2.0 / 7.0 + 1e-6).all()


def test_step_seir_window_conservation():
    pop = torch.full((25,), 1e5)
    state0 = torch.stack([pop * 0.5, pop * 0.01, pop * 0.01, pop * 0.48], dim=-1)
    lam = torch.full((1, 25), 0.05)
    states, inc = step_seir_window(state0.unsqueeze(0), lam, horizon=3)

    assert states.shape == (1, 25, 4, 4)
    assert inc.shape == (1, 25, 3)

    # Check mass conservation per week
    totals = states.sum(dim=-1)  # (1, 25, 4)
    init_total = totals[:, :, 0]
    for w in range(1, 4):
        diff = (totals[:, :, w] - init_total).abs().max()
        assert diff < 1e-3
