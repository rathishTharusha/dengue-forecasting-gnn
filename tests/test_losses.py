"""Unit tests for physics-informed and mechanistic loss functions."""

import pytest
import torch

from dengue_gnn.losses import (
    EpidemicDynamicsLoss,
    PhysicsInformedLoss,
    SpatialSmoothnessLoss,
    compute_normalized_laplacian,
)


def test_normalized_laplacian():
    N = 5
    # Adjacency with bidirectional ring edges
    adj = torch.zeros(N, N)
    for i in range(N):
        adj[i, (i + 1) % N] = 1.0
        adj[(i + 1) % N, i] = 1.0

    laplacian = compute_normalized_laplacian(adj)
    assert laplacian.shape == (N, N)
    # Check symmetry: L == L.T
    torch.testing.assert_close(laplacian, laplacian.t(), rtol=1e-5, atol=1e-5)


def test_spatial_smoothness_zero_on_constant():
    N = 10
    H = 3
    adj = torch.ones(N, N)  # fully connected
    lap = compute_normalized_laplacian(adj)

    # Spatially constant predictions: all districts have identical count
    pred_const = torch.ones(N, H) * 5.0
    loss_fn = SpatialSmoothnessLoss()
    loss = loss_fn(pred_const, lap)

    # Dirichlet energy for constant function on graph is ~0
    assert float(loss.item()) < 1e-4

    # Non-smooth predictions
    pred_erratic = torch.tensor([[float(i % 2 * 10)] * H for i in range(N)])
    loss_erratic = loss_fn(pred_erratic, lap)
    assert float(loss_erratic.item()) > 0.1


def test_epidemic_dynamics_zero_on_linear():
    loss_fn = EpidemicDynamicsLoss()

    # Linear trajectory along horizon [10, 20, 30] -> second derivative = 0
    pred_linear = torch.tensor([
        [[10.0, 20.0, 30.0]],
        [[5.0, 10.0, 15.0]],
    ])
    loss_linear = loss_fn(pred_linear)
    assert float(loss_linear.item()) < 1e-6

    # Curved/oscillating trajectory [0, 50, 0] -> high second derivative
    pred_curved = torch.tensor([
        [[0.0, 50.0, 0.0]],
    ])
    loss_curved = loss_fn(pred_curved)
    assert float(loss_curved.item()) > 100.0


def test_physics_informed_loss_gradients():
    N = 25
    H = 3
    in_dim = 16
    x = torch.randn(N, in_dim)
    w = torch.randn(in_dim, H, requires_grad=True)

    adj = torch.eye(N)
    lap = compute_normalized_laplacian(adj)

    pred = torch.matmul(x, w)
    target = torch.randn(N, H)

    loss_fn = PhysicsInformedLoss(lambda_smooth=0.1, lambda_phys=0.1)
    loss_dict = loss_fn(pred, target, laplacian=lap)

    assert "loss" in loss_dict
    assert "loss_data" in loss_dict
    assert "loss_smooth" in loss_dict
    assert "loss_phys" in loss_dict

    loss = loss_dict["loss"]
    loss.backward()
    assert w.grad is not None
    assert torch.all(torch.isfinite(w.grad))
