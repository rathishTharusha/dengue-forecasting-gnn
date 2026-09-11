"""Physics-informed and mechanistic domain loss functions for dengue forecasting.

Implements Contribution (a):
1. Spatial smoothness regularization via the normalized graph Laplacian:
   penalizes unphysical spatial discontinuities across geographically/mobility connected districts.
2. Temporal epidemiological dynamics regularization:
   penalizes discrete second-order acceleration (curvature) exceeding plausible
   vector-host transmission dynamics across forecast horizons.
3. Composite PhysicsInformedLoss integrating data-driven MSE with domain regularizers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """Compute the symmetric normalized graph Laplacian L_norm = I - D^(-1/2) A D^(-1/2).

    Args:
        adj: Adjacency matrix of shape (N, N).

    Returns:
        L_norm: Symmetric normalized Laplacian matrix of shape (N, N).
    """
    N = adj.shape[0]
    # Symmetrize adjacency if directed
    a_sym = 0.5 * (adj + adj.t())
    # Ensure positive diagonal
    a_sym = a_sym + torch.eye(N, device=adj.device, dtype=adj.dtype)

    deg = a_sym.sum(dim=-1)
    deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-6), -0.5)
    d_mat = torch.diag(deg_inv_sqrt)

    norm_adj = torch.matmul(torch.matmul(d_mat, a_sym), d_mat)
    laplacian = torch.eye(N, device=adj.device, dtype=adj.dtype) - norm_adj
    return laplacian


class SpatialSmoothnessLoss(nn.Module):
    """Dirichlet energy smoothness loss over graph topology: Tr(Y^T L Y) / (N * H)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """Compute spatial smoothness loss.

        Args:
            pred: Predicted values of shape (N, H) or (B, N, H).
            laplacian: Normalized Laplacian matrix of shape (N, N).

        Returns:
            Scalar smoothness loss.
        """
        if pred.ndim == 2:
            # (N, H) -> sum over horizons of y_h^T L y_h
            # L @ pred: (N, H)
            l_pred = torch.matmul(laplacian, pred)
            # sum over nodes and horizons: (pred * l_pred).sum()
            loss = torch.sum(pred * l_pred) / (pred.shape[0] * pred.shape[1])
            return torch.clamp(loss, min=0.0)
        elif pred.ndim == 3:
            # (B, N, H)
            B, N, H = pred.shape
            l_pred = torch.matmul(laplacian.unsqueeze(0), pred)
            loss = torch.sum(pred * l_pred) / (B * N * H)
            return torch.clamp(loss, min=0.0)
        else:
            raise ValueError(f"Expected pred ndim 2 or 3, got {pred.ndim}")


class EpidemicDynamicsLoss(nn.Module):
    """Discrete second-order temporal curvature loss across forecast horizons.

    Penalizes sudden erratic shifts in acceleration across consecutive weeks:
        Delta^2 y_h = y_{h+2} - 2*y_{h+1} + y_h
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute temporal acceleration loss.

        Args:
            pred: Predicted values of shape (..., H), where H >= 3.

        Returns:
            Scalar temporal curvature loss.
        """
        if pred.shape[-1] < 3:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Discrete second-order derivative along horizon dimension
        # shape: (..., H-2)
        d2 = pred[..., 2:] - 2.0 * pred[..., 1:-1] + pred[..., :-2]
        return torch.mean(torch.square(d2))


class PhysicsInformedLoss(nn.Module):
    """Composite loss combining data MSE, spatial graph smoothness, and temporal dynamics."""

    def __init__(
        self,
        lambda_smooth: float = 0.0,
        lambda_phys: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_phys = lambda_phys
        self.mse_loss = nn.MSELoss()
        self.spatial_loss = SpatialSmoothnessLoss()
        self.temporal_loss = EpidemicDynamicsLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        laplacian: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute total physics-informed loss and its individual components.

        Args:
            pred: Predicted tensor of shape (N, H).
            target: Ground-truth target tensor of shape (N, H).
            laplacian: Optional normalized graph Laplacian matrix (N, N).

        Returns:
            Dict containing "loss", "loss_data", "loss_smooth", and "loss_phys".
        """
        l_data = self.mse_loss(pred, target)

        l_smooth = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if self.lambda_smooth > 0.0 and laplacian is not None:
            l_smooth = self.spatial_loss(pred, laplacian)

        l_phys = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if self.lambda_phys > 0.0:
            l_phys = self.temporal_loss(pred)

        total_loss = l_data + self.lambda_smooth * l_smooth + self.lambda_phys * l_phys

        return {
            "loss": total_loss,
            "loss_data": l_data,
            "loss_smooth": l_smooth,
            "loss_phys": l_phys,
        }
