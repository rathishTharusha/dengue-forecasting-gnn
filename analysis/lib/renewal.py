"""Renewal-equation physics, derived from the project's validated SEIR-SEI model.

The measurement this exists to fix
----------------------------------
``analysis/results/error_diagnosis.json``: the sweep baselines are almost flat.
A3TGCN's predicted ``|d log(1+cases)/dweek|`` tops out at **0.336** against an
observed **3.638** -- a 7.7x under-reaction, 17.2x for STGAT -- and they carry a
**-12.9** case bias on outbreak windows, which are 12.6% of the data and 61.7% of
the squared error. They are sophisticated persistence. Forecasting outbreak weeks
as well as quiet ones would take RMSE from ~29 to ~20.

Why the renewal equation and not an ODE residual
------------------------------------------------
EXP-014 established that a hard SEIR-SEI residual is under-determined here: seven
compartments, exactly one observed (``I_h``, and only as under-ascertained
reported cases). It also killed the growth *ceiling*, for a reason that still
holds -- an upper bound is slack by 7x against a model that under-reacts, so it
contributes exactly zero gradient.

The renewal equation is the mechanistic identity that survives when the
compartments are unobservable, because it is a statement about the observable:

    cases_t  =  R_t * sum_{s>=1} w_s * cases_{t-s}

``w`` is the generation-interval distribution, and it follows from the SEIR-SEI
*stage durations* -- host incubation, host infectious period, vector incubation,
vector lifespan -- which :mod:`dengue_gnn.seir` implements and EXP-014 validated
against all nine of Phaijoo & Gurung's Table 1 sensitivity indices.

Only the stage durations are used, never their ``R0``. At the paper's section-4
values ``R0 = 0.783``, below the epidemic threshold, so their absolute calibration
is not Sri Lanka's. The generation interval is a property of the disease's
biology rather than of local transmission intensity, so it transfers; ``R0`` does
not, and is estimated from data instead.

Measured on this dataset (``analysis/_build/renewal_feasibility.py``): mean
generation interval **3.30 weeks**; back-solved ``R_t`` has median 0.911, p95
2.871, 44% above 1.0 across 91% of the record -- the endemic-around-threshold
shape the epidemiology predicts. A 3-week replay with oracle ``R_t`` scores RMSE
13.44 where persistence scores 55.03 on the same windows.

Two ways to use it
------------------
:class:`RenewalDecoder`
    The network predicts ``log R_t`` and cases are *reconstructed* through the
    equation. Structurally incapable of a flat forecast: ``R_t > 1`` grows.
:func:`renewal_penalty`
    A soft term for the unchanged architectures, penalising forecasts that no
    admissible ``R_t`` trajectory could produce.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

__all__ = [
    "HISTORY_WEEKS",
    "KERNEL_WEEKS",
    "LOG_R_BOUNDS",
    "RenewalDecoder",
    "estimate_r",
    "generation_interval",
    "renewal_forecast",
    "renewal_penalty",
]

#: Lags the renewal sum runs over. The SEIR-SEI generation interval is ~3.3 weeks
#: with a thin tail; 6 covers it without inventing long-range memory.
KERNEL_WEEKS = 6

#: Weeks of observed history a window needs: ``KERNEL_WEEKS`` to seed the sum,
#: plus 3 more so :func:`estimate_r` can back-solve R over a short recent stretch.
HISTORY_WEEKS = KERNEL_WEEKS + 3

#: ``log R_t`` is clamped to this, i.e. ``R_t`` in [0.05, 12.2]. The upper end sits
#: above the observed p99 of 5.36 so the bound does not bind on real outbreaks --
#: the mistake EXP-014 diagnosed in the old growth ceiling, in reverse. The lower
#: end keeps the reconstruction away from an exact zero, which has no gradient.
LOG_R_BOUNDS = (-3.0, 2.5)


def generation_interval(params, weeks: int = KERNEL_WEEKS, samples: int = 400_000,
                        seed: int = 0) -> np.ndarray:
    """Weekly generation-interval distribution implied by SEIR-SEI stage durations.

    A host-to-host transmission chain passes through four exponential stages:
    host incubation ``1/nu_h``, host infectious period ``1/gamma_h``, vector
    incubation ``1/nu_v``, and the vector's remaining lifespan ``1/mu_v``. The
    generation interval is their sum, so its density is a convolution of four
    exponentials -- taken by Monte Carlo here, because the closed form is
    numerically delicate when two rates nearly coincide, and this is computed once.

    Args:
        params: A :class:`dengue_gnn.seir.Params`, rates per day.
        weeks: Number of weekly lags to keep; the tail beyond is folded into the
            last bin.
        samples: Monte Carlo draws.
        seed: RNG seed, so the kernel is identical across runs and machines.

    Returns:
        Probabilities over lags ``1..weeks``, summing to 1. Lag 0 is excluded: a
        case cannot infect anyone during the week it is reported.
    """
    rng = np.random.default_rng(seed)
    days = (rng.exponential(1 / params.nu_h, samples)
            + rng.exponential(1 / params.gamma_h, samples)
            + rng.exponential(1 / params.nu_v, samples)
            + rng.exponential(1 / params.mu_v, samples))
    binned = np.clip(np.ceil(days / 7).astype(int), 1, weeks)
    counts = np.bincount(binned, minlength=weeks + 1)[1:].astype(float)
    return counts / counts.sum()


def _force(history: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Total infectious pressure ``sum_s w_s * cases_{t-s}``.

    Args:
        history: ``(..., KERNEL_WEEKS)`` raw counts, **most recent last**, so
            ``history[..., -1]`` is week ``t-1``.
        w: ``(KERNEL_WEEKS,)`` with ``w[0]`` the lag-1 weight.

    Returns:
        ``(...)`` -- the history axis is contracted away.
    """
    return (history * torch.flip(w, dims=(0,))).sum(dim=-1)


def renewal_forecast(log_r: torch.Tensor, history: torch.Tensor,
                     w: torch.Tensor) -> torch.Tensor:
    """Roll the renewal equation forward, differentiably.

    Each step consumes its own previous output, so the horizon compounds the way
    an epidemic does: sustained ``R_t > 1`` produces exponential growth rather
    than a constant offset. That compounding is the entire point -- it is what the
    residual-over-persistence baselines structurally cannot do.

    Args:
        log_r: ``(batch, nodes, horizon)`` predicted log reproduction numbers.
        history: ``(batch, nodes, KERNEL_WEEKS)`` observed raw counts, most recent
            last.
        w: ``(KERNEL_WEEKS,)`` generation-interval weights.

    Returns:
        ``(batch, nodes, horizon)`` forecast raw counts, non-negative by
        construction.
    """
    r = torch.exp(log_r.clamp(*LOG_R_BOUNDS))
    hist = history
    steps = []
    for step in range(log_r.shape[-1]):
        nxt = r[..., step] * _force(hist, w)
        steps.append(nxt)
        hist = torch.cat([hist[..., 1:], nxt.unsqueeze(-1)], dim=-1)
    return torch.stack(steps, dim=-1)


def estimate_r(history: torch.Tensor, w: torch.Tensor, weeks: int = 3,
               floor: float = 1.0) -> torch.Tensor:
    """Back-solve a recent ``R`` from observed counts alone.

    Pooled over the last ``weeks`` rather than taken at a single week: a
    week-by-week ratio of small counts is dominated by reporting noise, and this
    is used as a *target* for the physics penalty, so its variance goes straight
    into the gradient.

    Args:
        history: ``(batch, nodes, HISTORY_WEEKS)`` observed raw counts, most
            recent last.
        w: ``(KERNEL_WEEKS,)`` generation-interval weights.
        weeks: How many recent weeks to pool over.
        floor: Minimum pooled force. Below it the district has essentially no
            recent transmission and the ratio is meaningless, so ``R`` falls back
            to 1.0 -- a neutral "carry on as you are", not a fabricated 0.

    Returns:
        ``(batch, nodes)`` estimated ``R``.
    """
    kernel = w.shape[0]
    numer = history.new_zeros(history.shape[:-1])
    denom = history.new_zeros(history.shape[:-1])
    for back in range(weeks):
        end = history.shape[-1] - back
        numer = numer + history[..., end - 1]
        denom = denom + _force(history[..., end - 1 - kernel : end - 1], w)
    return torch.where(denom >= floor, numer / denom.clamp_min(1e-6),
                       torch.ones_like(denom))


class RenewalDecoder(nn.Module):
    """Turn a backbone's per-node output into cases via the renewal equation.

    The backbone is unchanged -- it still emits ``(batch, nodes, horizon)``. Only
    the *interpretation* changes: those numbers are ``log R_t`` rather than a
    residual in normalised log1p space, and cases come from the mechanism.

    That substitution is the experiment. Backbone, protocol, folds, normalisation
    and loss scale are all held fixed, so a difference against the same backbone's
    ``base`` arm is attributable to the decoder.

    Args:
        w: ``(KERNEL_WEEKS,)`` generation-interval weights, registered as a buffer
            so it moves with the module and is saved in the checkpoint.
        bias_init: Initial additive offset on ``log R``. Defaults to 0, i.e.
            ``R = 1``, so an untrained decoder reproduces the infectious pressure
            of recent weeks rather than collapsing to zero.
    """

    def __init__(self, w: torch.Tensor, bias_init: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("w", w.to(torch.float32))
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))

    def forward(self, raw: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """``(batch, nodes, horizon)`` backbone output -> forecast raw counts."""
        return renewal_forecast(raw + self.bias, history, self.w)


def renewal_penalty(pred_counts: torch.Tensor, history: torch.Tensor,
                    w: torch.Tensor, weeks: int = 3) -> torch.Tensor:
    """Penalise forecasts inconsistent with renewal dynamics at the recent ``R``.

    For each horizon step, the mechanism says the next value should be
    ``R * force``, where ``force`` accumulates the forecast's own recent past.
    The penalty is the squared discrepancy in ``log1p`` space -- the scale the
    rest of this project trains in, and the scale on which a factor-of-two error
    at 10 cases counts the same as at 1000.

    Unlike the growth ceiling EXP-014 retired, this is two-sided. It pushes a flat
    forecast *upward* when recent transmission implies ``R > 1``, which is the
    measured failure mode, and downward when ``R < 1``.

    ``R`` is detached: it is an observed-data estimate acting as a target, and
    letting gradients flow into it would let the model satisfy the constraint by
    revising the epidemiology instead of the forecast.

    Args:
        pred_counts: ``(batch, nodes, horizon)`` forecast raw counts.
        history: ``(batch, nodes, HISTORY_WEEKS)`` observed raw counts.
        w: ``(KERNEL_WEEKS,)`` generation-interval weights.
        weeks: Weeks pooled by :func:`estimate_r`.

    Returns:
        Scalar tensor.
    """
    kernel = w.shape[0]
    r = estimate_r(history, w, weeks).detach().unsqueeze(-1)
    hist = history[..., -kernel:]
    target = []
    for step in range(pred_counts.shape[-1]):
        target.append(r[..., 0] * _force(hist, w))
        hist = torch.cat([hist[..., 1:], pred_counts[..., step].unsqueeze(-1)], dim=-1)
    expected = torch.stack(target, dim=-1)
    return (torch.log1p(pred_counts.clamp_min(0.0)) - torch.log1p(expected.clamp_min(0.0))
            ).pow(2).mean()
