"""Regularisation terms for the spatially-constrained GNN objective.

    L_total = L_data + λ · ( L_smooth + cons_weight · L_cons )

Reconstructed from ``paper/sections/03_framework.tex``.

Which space these operate in, and why it matters
------------------------------------------------
The network is trained on a **residual over persistence, in log1p space** -- it
predicts how much this week differs from last week, on a log scale. Both terms
here are statements about *case counts*, not about residuals, so every function
in this module takes ``pred_counts``: predictions already inverted back to the
raw count scale.

This is not a stylistic preference. ``L_cons`` penalises negative predictions on
the grounds that a district cannot have negative cases. Applied to the raw
network output, a negative value does not mean "negative cases" -- it means
"fewer cases than last week", which is a correct and frequent forecast. Penalising
it would systematically bias the model against ever predicting a decline, which
on a seasonal disease is a serious distortion.

The original implementation is lost, so which space it used cannot be determined.
The API here forces the correct one.

A note on naming
----------------
The project proposal committed to a physics-informed loss derived from the
SEIR-SEI host-vector compartmental model. What is implemented here -- and what
the Phase-2 paper described -- is a graph-Laplacian smoothness penalty plus a
non-negativity constraint. That is spatial regularisation, not mechanistic
epidemiology: there are no compartments, no transmission dynamics, no ODE
residual. The module and its functions are named accordingly. See
``docs/PHASE2_REVIEW.md`` finding F5.
"""

from __future__ import annotations

import torch

__all__ = ["nonnegativity_loss", "smoothness_loss", "spatial_regularisation"]


def smoothness_loss(pred_counts: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """Penalise disagreement between districts the graph considers connected.

    A graph-Laplacian quadratic form: for every district pair, the squared
    difference in prediction, weighted by the edge joining them.

        L_smooth = Σ_ij A_ij (ŷ_i − ŷ_j)²   /   (B · H · ‖A‖₁)

    The intent is to suppress the failure mode Weng et al. describe on this
    dataset -- a GNN forecasting a large outbreak in one district and nothing at
    all next door, purely as an artefact of unconstrained message passing.

    Normalising by ``‖A‖₁`` keeps the term comparable across adjacency matrices
    with different total edge mass, which matters here because the blended
    adjacency changes shape during training.

    Args:
        pred_counts: Predictions on the raw count scale, shape
            ``(batch, n_nodes, horizon)``.
        adj: Adjacency used for message passing, shape ``(n_nodes, n_nodes)``.

    Returns:
        Scalar tensor.

    Raises:
        ValueError: If ranks or node counts do not line up.
    """
    if pred_counts.dim() != 3:
        raise ValueError(f"expected (batch, n_nodes, horizon), got {tuple(pred_counts.shape)}")
    n = pred_counts.shape[1]
    if adj.shape != (n, n):
        raise ValueError(f"adj must be ({n}, {n}), got {tuple(adj.shape)}")

    # (B, N, 1, H) - (B, 1, N, H) -> (B, N, N, H) pairwise differences
    diff = pred_counts.unsqueeze(2) - pred_counts.unsqueeze(1)
    weighted = adj.unsqueeze(0).unsqueeze(-1) * diff.pow(2)

    batch, _, _, horizon = weighted.shape
    denom = batch * horizon * adj.abs().sum().clamp_min(1e-8)
    return weighted.sum() / denom


def nonnegativity_loss(pred_counts: torch.Tensor) -> torch.Tensor:
    """Penalise negative predicted case counts.

        L_cons = mean( ReLU(−ŷ)² )

    ``pred_counts`` must be on the **raw count scale**. Passing the network's
    residual-space output here inverts the meaning of the constraint: see the
    module docstring.

    Args:
        pred_counts: Predictions on the raw count scale, any shape.

    Returns:
        Scalar tensor. Exactly zero when no prediction is negative.
    """
    return torch.relu(-pred_counts).pow(2).mean()


def spatial_regularisation(
    pred_counts: torch.Tensor,
    adj: torch.Tensor,
    lambda_phys: float,
    cons_weight: float = 10.0,
) -> torch.Tensor:
    """Combined regulariser: ``λ · (L_smooth + cons_weight · L_cons)``.

    Returns a zero scalar when ``lambda_phys == 0``, without building the pairwise
    difference tensor -- so the ``λ=0`` ablation row costs nothing extra and is
    exactly the unregularised model rather than an approximation of it.

    Args:
        pred_counts: Predictions on the raw count scale,
            shape ``(batch, n_nodes, horizon)``.
        adj: Adjacency used for message passing.
        lambda_phys: Overall regularisation weight.
        cons_weight: Relative weight of the non-negativity term, 10.0 in the
            Phase-2 configuration.

    Returns:
        Scalar tensor to add to the data loss.
    """
    if lambda_phys == 0.0:
        return pred_counts.sum() * 0.0  # keeps the graph connected, contributes nothing

    smooth = smoothness_loss(pred_counts, adj)
    cons = nonnegativity_loss(pred_counts)
    return lambda_phys * (smooth + cons_weight * cons)
