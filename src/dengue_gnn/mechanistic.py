"""Mechanistic constraints that use only observed quantities.

**Status: exploratory prototype.** Rule D13 reserves the physics formulation for a
human author. This module exists to produce evidence for that decision, not to be
the decision.

``MAX_WEEKLY_LOG_GROWTH`` is no longer a placeholder: it is derived from the
linearised SEIR-SEI model of Phaijoo & Gurung (2018) in :mod:`dengue_gnn.seir`.
What still needs an owner is the epidemiological input to that derivation -- the
``R0 <= 6`` cap and the parameter ranges -- not the arithmetic between them.

Why constraints on observables rather than a compartmental residual
-------------------------------------------------------------------
The cited SEIR-SEI model has seven compartments. Exactly one of them --
``I_h``, and only as under-ascertained reported cases -- is observed in this
dataset. A hard residual therefore asks the network to infer 150 latent weekly
trajectories (6 compartments x 25 districts) plus nine rate constants from 25
noisy series. That system is structurally under-determined: many latent
configurations reproduce the same observed cases and nothing in the loss prefers
the right one. See ``docs/STAGE3_EXPERIMENTS.md``.

The constraints here instead act on quantities computable from predictions alone,
and derive their form from the **generation interval** -- the mechanistic
quantity that survives when compartments are unobservable.

Both terms act on *rates of change*, which is deliberate: the measured gap
against persistence is in peak timing, not magnitude, and a peak is where the
growth rate crosses zero.
"""

from __future__ import annotations

import torch

__all__ = [
    "CEILING_R0_MAX",
    "MAX_WEEKLY_LOG_GROWTH",
    "adaptive_weight",
    "growth_band_loss",
    "growth_smoothness_loss",
    "implied_log_growth",
    "mechanistic_regularisation",
]

#: Ceiling on ``|d log(1+cases)/dweek|``, derived rather than guessed.
#:
#: This was 0.70, from ``ln(R)/GI`` with R=8 and a 3-week generation interval.
#: That was wrong, for a reason worth stating: ``ln(R)/GI`` assumes every onward
#: infection lands exactly one generation interval later, but the SEIR-SEI stage
#: durations are exponential, and for a given R0 a high-variance generation
#: interval produces markedly faster early growth than a fixed delay does
#: (Wallinga & Lipsitch 2007). It understated the bound by 3-4x.
#:
#: The consequence was not academic. **20.9% of the observed district-week growth
#: rates in our own data exceeded 0.70** -- the term was declaring a fifth of the
#: record physically impossible and penalising forecasts for tracking real
#: outbreaks. That is the most likely reason the band constraint never helped in
#: Stage 2b or Stage 3: not the weight, the constant.
#:
#: The value below is the largest weekly log growth the linearised SEIR-SEI model
#: can produce subject to ``R0 <= 6``, i.e. ``7 * max Re eig(F - V)``, from
#: Phaijoo & Gurung (2018). Only 1.07% of observations exceed it, which is the
#: right order for a genuine physical ceiling. Recomputed and asserted in
#: ``tests/test_seir.py``; derivation and data comparison in
#: ``scripts/verify_seir_paper.py`` and ``results/table_seir_validation.md``.
#:
#: Hard-coded rather than computed at import because the search takes ~3 s and
#: every pool worker would pay it.
#:
#: OWNER: confirm the ``R0 <= 6`` cap and the parameter ranges in
#: ``seir.DEFAULT_RANGES``. Those are the epidemiological judgements (D13);
#: everything downstream of them is mechanical.
MAX_WEEKLY_LOG_GROWTH = 2.3884

#: The R0 cap the ceiling above was derived at. Kept beside it so the two cannot
#: drift apart, and so the test can recompute one from the other.
CEILING_R0_MAX = 6.0


def implied_log_growth(pred_counts: torch.Tensor, last_observed: torch.Tensor) -> torch.Tensor:
    """Week-on-week log growth implied by a multi-horizon forecast.

    Args:
        pred_counts: Predicted case counts, shape ``(batch, n_nodes, horizon)``.
        last_observed: Observed counts at the forecast origin, shape
            ``(batch, n_nodes)`` -- the anchor for the first horizon step.

    Returns:
        Growth rates of shape ``(batch, n_nodes, horizon)``, where element ``h``
        is ``log1p(y_h) - log1p(y_{h-1})`` and ``y_{-1}`` is ``last_observed``.
    """
    if pred_counts.dim() != 3:
        raise ValueError(f"expected (batch, n_nodes, horizon), got {tuple(pred_counts.shape)}")
    if last_observed.shape != pred_counts.shape[:2]:
        raise ValueError(
            f"last_observed must be {tuple(pred_counts.shape[:2])}, "
            f"got {tuple(last_observed.shape)}"
        )

    log_pred = torch.log1p(pred_counts.clamp_min(0))
    anchor = torch.log1p(last_observed.clamp_min(0)).unsqueeze(-1)
    series = torch.cat([anchor, log_pred], dim=-1)
    return series[..., 1:] - series[..., :-1]


def growth_band_loss(
    pred_counts: torch.Tensor,
    last_observed: torch.Tensor,
    max_growth: float = MAX_WEEKLY_LOG_GROWTH,
) -> torch.Tensor:
    """Penalise forecast growth faster than transmission can produce.

    A hinge on ``|g| - max_growth``: zero inside the plausible band, quadratic
    outside it. Unlike the spatial smoothness penalty this does not push toward a
    uniform solution -- it constrains only the tails, so it has no incentive to
    flatten the graph structure the adaptive component is learning.
    """
    g = implied_log_growth(pred_counts, last_observed)
    return torch.relu(g.abs() - max_growth).pow(2).mean()


def growth_smoothness_loss(pred_counts: torch.Tensor, last_observed: torch.Tensor) -> torch.Tensor:
    """Penalise abrupt changes in growth rate between consecutive weeks.

    The effective reproduction number moves on the timescale of the generation
    interval (~3 weeks), so week-to-week growth should not jump. This is the
    renewal-equation intuition without instantiating the renewal kernel: it
    constrains the second difference of ``log(1+cases)``.

    Requires ``horizon >= 2``; returns a differentiable zero otherwise.
    """
    g = implied_log_growth(pred_counts, last_observed)
    if g.shape[-1] < 2:
        return g.sum() * 0.0
    return (g[..., 1:] - g[..., :-1]).pow(2).mean()


def mechanistic_regularisation(
    pred_counts: torch.Tensor,
    last_observed: torch.Tensor,
    lambda_mech: float,
    mode: str = "both",
    max_growth: float = MAX_WEEKLY_LOG_GROWTH,
) -> torch.Tensor:
    """Combined mechanistic penalty.

    Args:
        pred_counts: Predicted counts, ``(batch, n_nodes, horizon)``.
        last_observed: Counts at the forecast origin, ``(batch, n_nodes)``.
        lambda_mech: Weight. Zero returns a differentiable zero without building
            the intermediate tensors.
        mode: ``"band"``, ``"smooth"`` or ``"both"`` -- one variable at a time,
            so each term gets its own ablation row (D7).
        max_growth: Passed to :func:`growth_band_loss`.

    Raises:
        ValueError: On an unknown ``mode``.
    """
    if mode not in ("band", "smooth", "both"):
        raise ValueError(f"mode must be 'band', 'smooth' or 'both', got {mode!r}")
    if lambda_mech == 0.0:
        return pred_counts.sum() * 0.0

    total = pred_counts.sum() * 0.0
    if mode in ("band", "both"):
        total = total + growth_band_loss(pred_counts, last_observed, max_growth)
    if mode in ("smooth", "both"):
        total = total + growth_smoothness_loss(pred_counts, last_observed)
    return lambda_mech * total


def curriculum_weight(epoch: int, total_epochs: int, target: float, warmup: float = 0.5) -> float:
    """Ramp a constraint weight from 0 to ``target`` over the first ``warmup``.

    Krishnapriyan et al. find PINN failures are optimisation failures caused by
    the soft constraint deforming the loss landscape, and that raising the
    coefficient gradually rather than fixing it from step one recovers one to two
    orders of magnitude of error. Stage 2b used a fixed weight throughout, so this
    is the cheapest available test of whether that -- rather than the constraint
    itself -- is what failed.

    Args:
        epoch: Zero-based current epoch.
        total_epochs: Total planned epochs.
        target: Final weight.
        warmup: Fraction of training over which to reach ``target``.

    Returns:
        The weight for this epoch. Linear ramp, then constant.
    """
    if warmup <= 0:
        return target
    frac = min(1.0, epoch / max(1.0, warmup * total_epochs))
    return target * frac


def adaptive_weight(
    model: torch.nn.Module,
    data_loss: torch.Tensor,
    constraint_loss: torch.Tensor,
    current: float,
    alpha: float = 0.1,
    eps: float = 1e-8,
) -> float:
    """Gradient-norm loss balancing, after Wang, Teng & Perdikaris.

    Their learning-rate-annealing procedure tunes the constraint weight from
    back-propagated gradient statistics rather than fixing it by hand. The
    diagnosis is that the two loss terms produce gradients of wildly different
    magnitude, so one dominates the update and the other is effectively ignored --
    which is precisely what we measured here, where the spatial penalty exceeded
    the data loss by four orders of magnitude.

    The estimate balances the *largest* data-loss gradient against the *mean*
    constraint gradient,

        lambda_hat = max|grad L_data| / mean|grad L_constraint|

    and is applied with exponential smoothing, ``lambda <- (1-alpha) lambda +
    alpha lambda_hat``, with the paper's recommended ``alpha = 0.1``. Smoothing
    matters: the raw ratio is noisy at batch size 1 and an unsmoothed weight
    oscillates.

    This replaces both a fixed weight and the linear curriculum in
    :func:`curriculum_weight`. Unlike either, it needs no grid search -- which is
    the point, since the usable range turned out to be two orders of magnitude
    from where a hand-chosen grid would have looked.

    Args:
        model: The network whose parameters carry the gradients.
        data_loss: Scalar data-fit loss, with a graph.
        constraint_loss: Scalar constraint loss, with a graph.
        current: The weight in force now.
        alpha: Smoothing factor.
        eps: Guards division when the constraint gradient vanishes.

    Returns:
        The updated weight. Returns ``current`` unchanged when the constraint
        contributes no gradient, so an inert term cannot drive the weight to
        infinity.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    g_data = torch.autograd.grad(data_loss, params, retain_graph=True, allow_unused=True)
    g_cons = torch.autograd.grad(constraint_loss, params, retain_graph=True, allow_unused=True)

    max_data = max(
        (float(g.abs().max()) for g in g_data if g is not None),
        default=0.0,
    )
    cons_vals = [float(g.abs().mean()) for g in g_cons if g is not None]
    mean_cons = sum(cons_vals) / len(cons_vals) if cons_vals else 0.0

    if mean_cons <= eps or max_data <= eps:
        return current  # inert constraint, or a converged data term

    lambda_hat = max_data / mean_cons
    return (1.0 - alpha) * current + alpha * lambda_hat
