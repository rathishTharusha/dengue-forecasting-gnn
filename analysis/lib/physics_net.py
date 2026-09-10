"""Model wrapper incorporating relaxed biological envelope and spatial physics constraints.

Supported arms:
- base: unconstrained residual GNN baseline.
- envelope: biological growth ceiling (r_max = 2.3884) and host clearance rate bounds.
- spatial: district-normalized spatial smoothness across adjacent districts.
- composite: envelope + spatial + conservation (non-negativity).
- outbreak_aware: composite + asymmetric weighting on under-predictions.
"""

from __future__ import annotations

import numpy as np
import physics_loss as ploss
import torch
from torch import nn

__all__ = ["PHYSICS_ARMS", "RelaxedPhysicsNet"]

PHYSICS_ARMS = ("base", "envelope", "spatial", "composite", "outbreak_aware")


class RelaxedPhysicsNet(nn.Module):
    """Wraps any spatio-temporal GNN backbone with relaxed epidemiological constraints.

    Args:
        backbone: Spatio-temporal GNN mapping (batch, nodes, window) -> (batch, nodes, horizon).
        edge_index: Graph edge index tensor.
        adj_dense: Dense (N, N) adjacency matrix for spatial smoothness.
        district_scales: (N,) tensor of baseline district incidence (training mean).
        mode: One of PHYSICS_ARMS.
        mean: Training-fold mean of log1p(cases).
        std: Training-fold standard deviation of log1p(cases).
        lambda_env: Weight for biological envelope loss (default 0.1).
        lambda_smooth: Weight for normalized spatial smoothness (default 0.05).
        lambda_cons: Weight for non-negativity loss (default 0.1).
    """

    def __init__(
        self,
        backbone: nn.Module,
        edge_index: torch.Tensor,
        adj_dense: torch.Tensor,
        district_scales: torch.Tensor,
        mode: str = "composite",
        mean: float = 0.0,
        std: float = 1.0,
        lambda_env: float = 0.05,
        lambda_smooth: float = 0.001,
        lambda_cons: float = 0.01,
    ) -> None:
        super().__init__()
        if mode not in PHYSICS_ARMS:
            raise ValueError(f"mode must be one of {PHYSICS_ARMS}, got {mode!r}")

        self.backbone = backbone
        self.mode = mode
        self.mean = mean
        self.std = std
        self.lambda_env = lambda_env
        self.lambda_smooth = lambda_smooth
        self.lambda_cons = lambda_cons

        self.register_buffer("edge_index", edge_index)
        self.register_buffer("adj_dense", adj_dense.to(torch.float32))
        self.register_buffer("district_scales", district_scales.to(torch.float32))

    def to_counts(self, z: torch.Tensor) -> torch.Tensor:
        """Normalized log1p space -> raw counts, differentiably."""
        return torch.expm1((z * self.std + self.mean).clamp(0.0, 12.0))

    def forward(
        self, x: torch.Tensor, persistence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z_prediction, counts_prediction)."""
        raw = self.backbone(x, self.edge_index)
        z = raw + persistence  # residual over persistence
        return z, self.to_counts(z)

    def compute_loss(
        self,
        z_pred: torch.Tensor,
        counts_pred: torch.Tensor,
        z_true: torch.Tensor,
        counts_true: torch.Tensor,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the total loss combining data objective and physics terms.

        Returns:
            (total_loss, metrics_dict)
        """
        # Base data loss: Huber loss on z-scale (robust to heavy-tailed extremes)
        data_loss = nn.functional.smooth_l1_loss(z_pred, z_true)
        total_loss = data_loss
        metrics = {"data_loss": data_loss.item()}

        if self.mode == "base":
            return total_loss, metrics

        # 1. Biological envelope
        if self.mode in ("envelope", "composite", "outbreak_aware"):
            l_env = ploss.biological_envelope_loss(counts_pred, history)
            total_loss = total_loss + self.lambda_env * l_env
            metrics["l_env"] = l_env.item()

        # 2. Normalized spatial smoothness
        if self.mode in ("spatial", "composite", "outbreak_aware"):
            l_smooth = ploss.normalized_smoothness_loss(
                counts_pred, self.adj_dense, self.district_scales
            )
            total_loss = total_loss + self.lambda_smooth * l_smooth
            metrics["l_smooth"] = l_smooth.item()

        # 3. Non-negativity conservation
        l_cons = ploss.nonnegativity_loss(counts_pred)
        total_loss = total_loss + self.lambda_cons * l_cons
        metrics["l_cons"] = l_cons.item()

        # 4. Asymmetric outbreak weighting
        if self.mode == "outbreak_aware":
            l_asym = ploss.asymmetric_outbreak_loss(counts_pred, counts_true, alpha=0.3)
            # Scale down to avoid dominating normalized data loss
            total_loss = total_loss + 1e-3 * l_asym
            metrics["l_asym"] = l_asym.item()

        return total_loss, metrics
