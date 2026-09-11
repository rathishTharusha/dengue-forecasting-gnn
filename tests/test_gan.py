"""Unit tests for Conditional WGAN-GP and augmentation pipelines."""

import pytest
import torch

from dengue_gnn.data import FoldData
from dengue_gnn.gan import (
    ConditionalCritic,
    ConditionalGenerator,
    compute_gradient_penalty,
    heuristic_augmentation,
    synthesize_gan_augmentation,
)


@pytest.fixture
def dummy_batch():
    B, N, in_dim, horizon, latent_dim = 4, 25, 33, 3, 16
    x = torch.randn(B, N, in_dim)
    y = torch.randn(B, N, horizon)
    p = torch.randn(B, N, horizon)
    noise = torch.randn(B, N, latent_dim)
    return x, y, p, noise, in_dim, horizon, latent_dim


def test_conditional_generator(dummy_batch):
    x, _, _, noise, in_dim, horizon, latent_dim = dummy_batch
    gen = ConditionalGenerator(in_dim=in_dim, latent_dim=latent_dim, horizon=horizon)

    fake_y = gen(noise, x)
    assert fake_y.shape == (x.shape[0], x.shape[1], horizon)

    # Backward pass
    fake_y.sum().backward()
    for name, param in gen.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"


def test_conditional_critic(dummy_batch):
    x, y, _, _, in_dim, horizon, _ = dummy_batch
    critic = ConditionalCritic(in_dim=in_dim, horizon=horizon)

    score = critic(y, x)
    assert score.shape == (x.shape[0], 1, 1)

    score.sum().backward()
    for name, param in critic.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"


def test_gradient_penalty(dummy_batch):
    x, y, _, _, in_dim, horizon, _ = dummy_batch
    critic = ConditionalCritic(in_dim=in_dim, horizon=horizon)
    fake_y = torch.randn_like(y)

    gp = compute_gradient_penalty(critic, y, fake_y, x, device="cpu", lambda_gp=10.0)
    assert gp.ndim == 0  # scalar
    assert float(gp.item()) >= 0.0
    assert torch.isfinite(gp)


def test_heuristic_augmentation():
    B, N, in_dim, horizon = 10, 25, 33, 3
    x = torch.randn(B, N, in_dim)
    y = torch.randn(B, N, horizon)
    p = torch.randn(B, N, horizon)

    fold_data = FoldData(X=x, Y=y, P=p, inv_fn=lambda z: z)
    aug_data = heuristic_augmentation(fold_data, noise_std=0.02)

    # Dataset should be doubled in length
    assert aug_data.X.shape == (20, N, in_dim)
    assert aug_data.Y.shape == (20, N, horizon)
    assert aug_data.P.shape == (20, N, horizon)


def test_synthesize_gan_augmentation(dummy_batch):
    x, y, p, _, in_dim, horizon, latent_dim = dummy_batch
    fold_data = FoldData(X=x, Y=y, P=p, inv_fn=lambda z: z)

    gen = ConditionalGenerator(in_dim=in_dim, latent_dim=latent_dim, horizon=horizon)
    aug_data = synthesize_gan_augmentation(gen, fold_data, ratio=0.5, latent_dim=latent_dim)

    # 4 original + 2 synthetic = 6
    assert aug_data.X.shape[0] == 6
    assert aug_data.Y.shape[0] == 6
    assert aug_data.P.shape[0] == 6
