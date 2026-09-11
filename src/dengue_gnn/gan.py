"""Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP) for dengue time-series augmentation.

Implements Contribution (b):
1. ConditionalGenerator: Synthesizes multi-horizon target sequences conditioned on
   historical node features and random latent vectors.
2. ConditionalCritic: 1-Lipschitz discriminator estimating Wasserstein distance.
3. Gradient penalty regularization for stable training without mode collapse on scarce data.
4. Heuristic comparator: Jittering and magnitude warping baseline.
5. Augmentation pipeline producing augmented FoldData for downstream GNN training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

from dengue_gnn.data import FoldData


class ConditionalGenerator(nn.Module):
    """Conditional Generator producing synthetic multi-horizon forecasts."""

    def __init__(
        self,
        in_dim: int = 33,
        latent_dim: int = 16,
        horizon: int = 3,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.horizon = horizon

        self.net = nn.Sequential(
            nn.Linear(in_dim + latent_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden // 2, horizon),
        )

    def forward(self, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Generate synthetic targets.

        Args:
            noise: Random Gaussian noise of shape (..., N, latent_dim).
            cond: Conditioning node features of shape (..., N, in_dim).

        Returns:
            out: Synthetic target predictions of shape (..., N, horizon).
        """
        x_in = torch.cat([cond, noise], dim=-1)
        return self.net(x_in)


class ConditionalCritic(nn.Module):
    """Conditional Wasserstein Critic scoring sequence realism."""

    def __init__(
        self,
        in_dim: int = 33,
        horizon: int = 3,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.horizon = horizon

        self.net = nn.Sequential(
            nn.Linear(in_dim + horizon, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Score sequence realism.

        Args:
            target: Target sequence of shape (..., N, horizon).
            cond: Conditioning features of shape (..., N, in_dim).

        Returns:
            score: Mean scalar Wasserstein score across nodes.
        """
        x_in = torch.cat([cond, target], dim=-1)
        node_scores = self.net(x_in)  # (..., N, 1)
        return torch.mean(node_scores, dim=(-2, -1), keepdim=True)


def compute_gradient_penalty(
    critic: ConditionalCritic,
    real_target: torch.Tensor,
    fake_target: torch.Tensor,
    cond: torch.Tensor,
    device: torch.device | str = "cpu",
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """Compute 1-Lipschitz gradient penalty on interpolated points."""
    dev = torch.device(device)
    # Shape: (1, 1, 1) to broadcast across batch, nodes, horizon
    alpha = torch.rand((real_target.shape[0], 1, 1), device=dev)
    interpolates = (alpha * real_target + (1.0 - alpha) * fake_target).requires_grad_(True)

    d_interpolates = critic(interpolates, cond)

    fake_grad_output = torch.ones_like(d_interpolates, requires_grad=False)
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_grad_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(gradients.shape[0], -1)
    grad_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * torch.mean((grad_norm - 1.0) ** 2)
    return gradient_penalty


def train_wgan_gp(
    train_data: FoldData,
    in_dim: int = 33,
    latent_dim: int = 16,
    horizon: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
    n_critic: int = 5,
    lr: float = 1e-4,
    device: torch.device | str = "cpu",
    verbose: bool = False,
) -> tuple[ConditionalGenerator, ConditionalCritic]:
    """Train a Conditional WGAN-GP on training-fold data."""
    dev = torch.device(device)
    generator = ConditionalGenerator(in_dim=in_dim, latent_dim=latent_dim, horizon=horizon).to(dev)
    critic = ConditionalCritic(in_dim=in_dim, horizon=horizon).to(dev)

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))

    X_train = train_data.X.to(dev)
    # Target in normalized residual space
    Y_train = (train_data.Y - train_data.P).to(dev)
    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)

        for start in range(0, n_samples, batch_size):
            batch_idx = perm[start : start + batch_size]
            b_x = X_train[batch_idx]
            b_y = Y_train[batch_idx]
            b_size = b_x.shape[0]

            # ---------------------
            #  Train Critic
            # ---------------------
            for _ in range(n_critic):
                opt_d.zero_grad()

                noise = torch.randn(b_size, b_x.shape[1], latent_dim, device=dev)
                fake_y = generator(noise, b_x).detach()

                d_real = critic(b_y, b_x)
                d_fake = critic(fake_y, b_x)

                gp = compute_gradient_penalty(critic, b_y, fake_y, b_x, device=dev)
                d_loss = torch.mean(d_fake) - torch.mean(d_real) + gp
                d_loss.backward()
                opt_d.step()

            # ---------------------
            #  Train Generator
            # ---------------------
            opt_g.zero_grad()
            gen_noise = torch.randn(b_size, b_x.shape[1], latent_dim, device=dev)
            gen_fake = generator(gen_noise, b_x)
            g_loss = -torch.mean(critic(gen_fake, b_x))
            g_loss.backward()
            opt_g.step()

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  [WGAN-GP] Epoch {epoch:3d}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    return generator, critic


def heuristic_augmentation(
    train_data: FoldData,
    noise_std: float = 0.03,
    scale_range: tuple[float, float] = (0.95, 1.05),
) -> FoldData:
    """Mandatory comparator: cheap jittering and magnitude scaling baseline.

    Args:
        train_data: Original FoldData.
        noise_std: Standard deviation of Gaussian jitter added to features.
        scale_range: Uniform scaling factor range for targets.

    Returns:
        Augmented FoldData with doubled training set.
    """
    X, Y, P = train_data.X, train_data.Y, train_data.P

    # 1. Jittering
    noise = torch.randn_like(X) * noise_std
    X_aug = X + noise

    # 2. Magnitude scaling
    scale = torch.empty(Y.shape[0], 1, 1).uniform_(*scale_range)
    Y_aug = Y * scale
    P_aug = P * scale

    X_combined = torch.cat([X, X_aug], dim=0)
    Y_combined = torch.cat([Y, Y_aug], dim=0)
    P_combined = torch.cat([P, P_aug], dim=0)

    return FoldData(X=X_combined, Y=Y_combined, P=P_combined, inv_fn=train_data.inv_fn)


def synthesize_gan_augmentation(
    generator: ConditionalGenerator,
    train_data: FoldData,
    ratio: float = 1.0,
    latent_dim: int = 16,
    device: torch.device | str = "cpu",
) -> FoldData:
    """Generate synthetic (X, Y) sequences using the trained generator.

    Args:
        generator: Trained ConditionalGenerator.
        train_data: Original FoldData.
        ratio: Ratio of synthetic samples to add (1.0 = double dataset).
        latent_dim: Latent noise dimension.
        device: PyTorch device.

    Returns:
        Augmented FoldData with synthetic samples appended.
    """
    generator.eval()
    dev = torch.device(device)

    X_orig = train_data.X
    Y_orig = train_data.Y
    P_orig = train_data.P

    n_syn = int(len(X_orig) * ratio)
    if n_syn == 0:
        return train_data

    # Sample random training indices to provide conditioning context
    indices = torch.randint(0, len(X_orig), (n_syn,))
    cond_x = X_orig[indices].to(dev)
    cond_p = P_orig[indices].to(dev)

    with torch.no_grad():
        noise = torch.randn(n_syn, cond_x.shape[1], latent_dim, device=dev)
        # Generator produces synthetic residual: Delta y_syn
        syn_residual = generator(noise, cond_x)
        syn_y = syn_residual + cond_p

    X_aug = torch.cat([X_orig, cond_x.cpu()], dim=0)
    Y_aug = torch.cat([Y_orig, syn_y.cpu()], dim=0)
    P_aug = torch.cat([P_orig, cond_p.cpu()], dim=0)

    return FoldData(X=X_aug, Y=Y_aug, P=P_aug, inv_fn=train_data.inv_fn)
