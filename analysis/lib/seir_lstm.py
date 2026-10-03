"""SEIR-LSTM model (Liu et al. 2025, made causal).

Encoder: LSTM over causal input window [cases, era5, ndvi].
Head: predicts force of infection lambda_i(t) = lambda_max * sigmoid(Linear(h)).
Dynamics: seir_sim exponential-flow compartmental integration.

Supports two formulations:
- F-win: independent windows, state rebuilt per window from past cases.
- F-seq: continuous rollout over the series.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import seir_sim


class SEIRLSTMHead(nn.Module):
    """LSTM encoder + sigmoid foi head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        lambda_max: float = 2.0 / 7.0,  # daily rate corresponding to lambda_max per week
    ):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.lambda_max = lambda_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, T, C) input features per district per timestep
        Returns:
            lambda: (B, N, 1) or (B, N) force of infection per day
        """
        B, N, T, C = x.shape
        x_flat = x.view(B * N, T, C)
        out, (h_n, c_n) = self.lstm(x_flat)
        h_last = h_n[-1]  # (B*N, hidden_dim)
        sig = self.head(h_last).view(B, N)
        return self.lambda_max * sig


def step_seir_window(
    state0: torch.Tensor,
    lam_daily: torch.Tensor,
    horizon: int = 3,
    omega: float = 0.7 / 7.0,
    gamma: float = 1.0 / 7.0,
    substeps: int = 7
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate horizon weeks under lam_daily held or projected.

    Returns (states, weekly_incidence)
    """
    # Expand lam_daily across horizon weeks
    lam_seq = lam_daily.unsqueeze(-1).repeat(1, 1, horizon)  # (B, N, H)
    return seir_sim.simulate_weeks(state0, lam_seq, omega, gamma, substeps=substeps)
