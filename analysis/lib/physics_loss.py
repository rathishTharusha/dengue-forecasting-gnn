"""Epidemiological physics loss functions derived from the SEIR-SEI compartmental model.

Formulation:
    L_total = L_data + lambda_env * L_envelope + lambda_smooth * L_smooth + lambda_cons * L_cons

Unlike rigid equality penalties that force predictions to match an inaccurate
backward-looking extrapolation, these terms act as *inductive biological boundaries*
and *spatial flux constraints*:
1. Biological Envelope (L_envelope):
   Penalizes predicted weekly growth exceeding the biological maximum reproduction rate
   r_max = 2.3884 derived from Phaijoo & Gurung (2018), or decay faster than host clearance.
   Within biological limits, the penalty is identically zero.
2. Normalized Smoothness (L_smooth):
   Penalizes spatial disparity in relative incidence (y_i / baseline_i) across connected
   districts, preventing hallucinated spatial spikes while preserving natural urban/rural
   scale differences.
3. Conservation (L_cons):
   Penalizes negative case counts.
"""

from __future__ import annotations

import torch

__all__ = [
    "asymmetric_outbreak_loss",
    "biological_envelope_loss",
    "log_smoothness_loss",
    "nonnegativity_loss",
    "normalized_smoothness_loss",
]

#: Maximum weekly log-growth rate from SEIR-SEI stage durations (Phaijoo & Gurung 2018).
DEFAULT_R_MAX: float = 2.3884

#: Maximum weekly disease clearance rate from human infectious period (mean ~3 days -> gamma_h ~ 0.33/day -> ~2.3/week).
DEFAULT_GAMMA_WEEK: float = 2.3026


def biological_envelope_loss(
    pred_counts: torch.Tensor,
    history: torch.Tensor,
    r_max: float = DEFAULT_R_MAX,
    gamma_week: float = DEFAULT_GAMMA_WEEK,
) -> torch.Tensor:
    """Penalize predictions exceeding biological growth ceilings or clearance rates.

    For horizon step 1: growth is relative to the last observed history week t-1.
    For horizon steps > 1: growth is step-to-step along the predicted trajectory.

    Args:
        pred_counts: (batch, nodes, horizon) on raw count scale.
        history: (batch, nodes, history_weeks) observed history, most recent last.
        r_max: Upper bound on weekly log growth.
        gamma_week: Maximum rate of weekly exponential drop.

    Returns:
        Scalar tensor penalty. Identically zero when predictions stay within bounds.
    """
    if pred_counts.dim() != 3:
        raise ValueError(f"expected (batch, nodes, horizon), got {tuple(pred_counts.shape)}")

    # Extract last observed week (t-1)
    last_obs = history[..., -1:].clamp_min(0.0)  # (batch, nodes, 1)

    # Chain history with predictions to get trajectory [y_{t-1}, y_t, y_{t+1}, ...]
    # Evaluate in log1p space
    log_traj = torch.log1p(torch.cat([last_obs, pred_counts.clamp_min(0.0)], dim=-1))

    # Weekly log-growth rates: Δ log(1 + y)
    growth_rates = log_traj[..., 1:] - log_traj[..., :-1]  # (batch, nodes, horizon)

    # Upper violation: growth > r_max
    upper_viol = torch.relu(growth_rates - r_max)

    # Lower violation: growth < -gamma_week (i.e. -gamma_week - growth > 0)
    lower_viol = torch.relu(-gamma_week - growth_rates)

    return (upper_viol.pow(2) + lower_viol.pow(2)).mean()


def normalized_smoothness_loss(
    pred_counts: torch.Tensor,
    adj: torch.Tensor,
    district_scales: torch.Tensor,
) -> torch.Tensor:
    """Penalize spatial disparity in relative incidence across connected districts.

    Normalizing each district by its baseline scale (e.g. historical mean incidence)
    prevents high-incidence districts (Colombo) from being artificially flattened
    toward low-incidence rural districts.

        L_smooth = Σ_ij A_ij (ŷ_i / s_i − ŷ_j / s_j)² / (B · H · ‖A‖₁)

    Args:
        pred_counts: (batch, nodes, horizon) on raw count scale.
        adj: (nodes, nodes) adjacency matrix.
        district_scales: (nodes,) positive scale factor per district.

    Returns:
        Scalar tensor penalty.
    """
    if pred_counts.dim() != 3:
        raise ValueError(f"expected (batch, nodes, horizon), got {tuple(pred_counts.shape)}")
    n = pred_counts.shape[1]
    if adj.shape != (n, n):
        raise ValueError(f"adj must be ({n}, {n}), got {tuple(adj.shape)}")

    scales = district_scales.view(1, n, 1).clamp_min(1.0)
    rel_pred = pred_counts / scales  # (batch, nodes, horizon)

    # Pairwise differences: (B, N, 1, H) - (B, 1, N, H) -> (B, N, N, H)
    diff = rel_pred.unsqueeze(2) - rel_pred.unsqueeze(1)
    weighted = adj.unsqueeze(0).unsqueeze(-1) * diff.pow(2)

    batch, _, _, horizon = weighted.shape
    denom = batch * horizon * adj.abs().sum().clamp_min(1e-8)
    return weighted.sum() / denom


def log_smoothness_loss(
    pred_counts: torch.Tensor,
    adj: torch.Tensor,
    district_scales: torch.Tensor,
) -> torch.Tensor:
    """Spatial smoothness on *log* relative incidence rather than the ratio.

        L = Σ_ij A_ij ( [log(1+ŷ_i) − log(1+s_i)] − [log(1+ŷ_j) − log(1+s_j)] )²
            / (B · H · ‖A‖₁)

    Same intent as :func:`normalized_smoothness_loss` -- penalise disagreement in
    *relative* incidence between neighbours, not in raw counts -- but bounded.

    Why this variant exists. The ratio form ``ŷ_i / s_i`` is scale-free in the
    mean and not in the tail: during an outbreak a district reaches 10--20 times
    its baseline while its neighbour sits at 1, so the squared difference grows
    quadratically in outbreak magnitude. Measured on the observed record, the
    ratio penalty is **85.6x larger on weeks containing an outbreak than on quiet
    weeks**; the log form is 1.1x, i.e. essentially flat.

    That matters because outbreak windows are 12.6% of this dataset and 61.7% of
    its squared error (``analysis/results/error_diagnosis.json``). A penalty
    concentrated there is asking the model to flatten outbreaks toward their
    neighbours in exactly the windows where accuracy is decided, and it forces the
    weight down to keep that pressure tolerable. Working in log space
    de-concentrates the term, so the same weight regularises the whole range --
    or a larger weight becomes usable.

    Args:
        pred_counts: ``(batch, nodes, horizon)`` on the raw count scale.
        adj: ``(nodes, nodes)`` adjacency.
        district_scales: ``(nodes,)`` positive baseline scale per district.

    Returns:
        Scalar tensor penalty.
    """
    if pred_counts.dim() != 3:
        raise ValueError(f"expected (batch, nodes, horizon), got {tuple(pred_counts.shape)}")
    n = pred_counts.shape[1]
    if adj.shape != (n, n):
        raise ValueError(f"adj must be ({n}, {n}), got {tuple(adj.shape)}")

    offset = torch.log1p(district_scales.clamp_min(0.0)).view(1, n, 1)
    rel = torch.log1p(pred_counts.clamp_min(0.0)) - offset

    diff = rel.unsqueeze(2) - rel.unsqueeze(1)
    weighted = adj.unsqueeze(0).unsqueeze(-1) * diff.pow(2)

    batch, _, _, horizon = weighted.shape
    denom = batch * horizon * adj.abs().sum().clamp_min(1e-8)
    return weighted.sum() / denom


def nonnegativity_loss(pred_counts: torch.Tensor) -> torch.Tensor:
    """Penalize negative predicted case counts.

        L_cons = mean( ReLU(−ŷ)² )
    """
    return torch.relu(-pred_counts).pow(2).mean()


def asymmetric_outbreak_loss(
    pred_counts: torch.Tensor,
    target_counts: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Asymmetric loss placing extra weight on under-predictions (ŷ < y).

    Addresses the -12.9 case under-prediction bias documented in error_diagnosis.json.

        L = mean( (ŷ − y)² · (1 + α · 1_{ŷ < y}) )

    Args:
        pred_counts: Predicted counts.
        target_counts: Ground truth counts.
        alpha: Weight boost for under-prediction.
    """
    err = pred_counts - target_counts
    weight = 1.0 + alpha * (err < 0.0).float()
    return (weight * err.pow(2)).mean()
