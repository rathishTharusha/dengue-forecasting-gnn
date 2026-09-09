"""Training-set augmentation for scarce epidemic series — Contribution (b).

The proposal's third contribution: 459 weeks x 25 districts is a small corpus for a
deep model, so synthesise additional ``(window -> horizon)`` training pairs.

**The comparator is not optional.** ``docs/ROADMAP.md`` Stage 5 requires that any
GAN be judged against cheap augmentation -- jittering and window warping -- on
*downstream* forecast accuracy, never on how realistic the synthetic series look.
A generator that produces convincing series but does not improve forecasting has
not contributed anything, and distributional similarity is the metric that lets a
GAN appear successful while being useless. So the cheap methods are implemented
first and the GAN must beat them.

Everything here operates in the same normalised log-residual space the models
train in, and augments only the *training* split. Augmenting validation or test
would leak synthetic structure into the numbers being reported.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "AUGMENT_KINDS",
    "TimeSeriesWGAN",
    "augment_training_set",
    "jitter",
    "window_warp",
]

AUGMENT_KINDS = ("none", "jitter", "window_warp", "gan")


def jitter(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """Additive Gaussian noise, the cheapest useful augmentation.

    Args:
        x: Any tensor of normalised features.
        sigma: Noise scale, relative to the (unit-variance) normalised features.
    """
    return x + torch.randn_like(x) * sigma


def window_warp(x: torch.Tensor, window: int, n_feat: int, scale: float = 0.2) -> torch.Tensor:
    """Stretch or compress the time axis of a node window by interpolation.

    ``x`` is ``(n_nodes, n_feat * window)`` laid out feature-major, matching
    ``_build_fold``. It is reshaped to ``(n_nodes, n_feat, window)``, resampled
    along the last axis at a random rate in ``1 +/- scale``, then resampled back to
    the original length. The result is a window whose dynamics ran slightly faster
    or slower -- plausible for an epidemic whose speed varies with climate.
    """
    n_nodes = x.shape[0]
    seq = x.reshape(n_nodes, n_feat, window)
    if window < 2:
        return x
    rate = float(1.0 + (torch.rand(1).item() * 2 - 1) * scale)
    warped_len = max(2, round(window * rate))
    stretched = nn.functional.interpolate(seq, size=warped_len, mode="linear", align_corners=True)
    back = nn.functional.interpolate(stretched, size=window, mode="linear", align_corners=True)
    return back.reshape(n_nodes, n_feat * window)


class TimeSeriesWGAN(nn.Module):
    """Conditional WGAN-GP over ``(condition -> target)`` pairs.

    RCGAN-style rather than TimeGAN: the generator is conditioned on the observed
    covariate window and emits the corresponding target sequence, so a synthetic
    sample is a plausible *continuation of real conditions* rather than a free
    invention. That matters at this sample size -- an unconditional generator has
    459 sequences to learn from and will memorise or collapse.

    WGAN-GP (gradient penalty) is used rather than the original GAN objective
    because mode collapse is the named risk for this dataset in the project's own
    risk register, and the Wasserstein critic with a gradient penalty is the
    standard mitigation.

    Args:
        cond_dim: Width of the conditioning vector (``n_feat * window``).
        target_dim: Width of the generated target (``horizon``).
        noise_dim: Latent width.
        hidden: Hidden width for both networks.
    """

    def __init__(self, cond_dim: int, target_dim: int, noise_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.noise_dim = noise_dim
        self.generator = nn.Sequential(
            nn.Linear(cond_dim + noise_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, target_dim),
        )
        # No sigmoid: a Wasserstein critic scores, it does not classify.
        self.critic = nn.Sequential(
            nn.Linear(cond_dim + target_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def generate(self, cond: torch.Tensor) -> torch.Tensor:
        z = torch.randn(cond.shape[0], self.noise_dim, device=cond.device)
        return self.generator(torch.cat([cond, z], dim=-1))

    def score(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.critic(torch.cat([cond, target], dim=-1))


def _gradient_penalty(gan: TimeSeriesWGAN, cond, real, fake) -> torch.Tensor:
    """Two-sided penalty pulling the critic's gradient norm toward 1."""
    eps = torch.rand(real.shape[0], 1, device=real.device)
    mixed = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = gan.score(cond, mixed)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=mixed,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def fit_gan(
    cond: torch.Tensor,
    target: torch.Tensor,
    epochs: int = 200,
    batch: int = 256,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    lr: float = 1e-4,
    seed: int = 0,
) -> TimeSeriesWGAN:
    """Train the critic and generator on flattened ``(condition, target)`` pairs.

    Args:
        cond: ``(n_samples, cond_dim)`` conditioning windows.
        target: ``(n_samples, target_dim)`` residual targets.
        epochs: Passes over the data.
        n_critic: Critic steps per generator step, as WGAN prescribes.
        lambda_gp: Gradient-penalty weight.
    """
    torch.manual_seed(seed)
    gan = TimeSeriesWGAN(cond.shape[1], target.shape[1])
    opt_c = torch.optim.Adam(gan.critic.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_g = torch.optim.Adam(gan.generator.parameters(), lr=lr, betas=(0.5, 0.9))
    n = cond.shape[0]

    for _ in range(epochs):
        idx = torch.randperm(n)[: min(batch, n)]
        c, real = cond[idx], target[idx]

        for _ in range(n_critic):
            fake = gan.generate(c).detach()
            loss_c = (
                gan.score(c, fake).mean()
                - gan.score(c, real).mean()
                + lambda_gp * _gradient_penalty(gan, c, real, fake)
            )
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

        fake = gan.generate(c)
        loss_g = -gan.score(c, fake).mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

    return gan


def augment_training_set(
    x: torch.Tensor,
    y: torch.Tensor,
    p: torch.Tensor,
    kind: str,
    ratio: float,
    window: int,
    n_feat: int,
    seed: int = 0,
    gan_epochs: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the training tensors with ``ratio * len(x)`` synthetic windows appended.

    Args:
        x: ``(n_windows, n_nodes, n_feat*window)`` inputs.
        y: ``(n_windows, n_nodes, horizon)`` targets.
        p: ``(n_windows, n_nodes, horizon)`` persistence anchors.
        kind: One of :data:`AUGMENT_KINDS`.
        ratio: Synthetic windows as a fraction of real ones. 0 disables.
        window: Weeks per input window.
        n_feat: Features per week.
        seed: Seeds the augmentation RNG.
        gan_epochs: Training length for ``kind="gan"``.

    Returns:
        ``(x_aug, y_aug, p_aug)``. Synthetic rows are appended, never substituted,
        so the real data is always present.

    Raises:
        ValueError: On an unknown ``kind``.
    """
    if kind not in AUGMENT_KINDS:
        raise ValueError(f"unknown augmentation {kind!r}; expected one of {AUGMENT_KINDS}")
    if kind == "none" or ratio <= 0:
        return x, y, p

    torch.manual_seed(seed)
    n_new = max(1, round(ratio * x.shape[0]))
    pick = torch.randint(0, x.shape[0], (n_new,))

    if kind == "jitter":
        xs = torch.stack([jitter(x[i]) for i in pick])
        ys, ps = y[pick].clone(), p[pick].clone()

    elif kind == "window_warp":
        xs = torch.stack([window_warp(x[i], window, n_feat) for i in pick])
        ys, ps = y[pick].clone(), p[pick].clone()

    else:  # gan
        n_win, n_nodes, cond_dim = x.shape
        horizon = y.shape[-1]
        # Flatten (window, node) into independent samples: the generator models a
        # district-week, not the whole graph, so 25 nodes multiply the corpus.
        cond = x.reshape(n_win * n_nodes, cond_dim)
        resid = (y - p).reshape(n_win * n_nodes, horizon)
        gan = fit_gan(cond, resid, epochs=gan_epochs, seed=seed)

        with torch.no_grad():
            xs = x[pick].clone()
            flat_cond = xs.reshape(n_new * n_nodes, cond_dim)
            fake_resid = gan.generate(flat_cond).reshape(n_new, n_nodes, horizon)
        ps = p[pick].clone()
        ys = ps + fake_resid  # synthetic level = anchor + synthetic correction

    return torch.cat([x, xs]), torch.cat([y, ys]), torch.cat([p, ps])
