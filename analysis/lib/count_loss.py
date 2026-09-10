"""Negative-binomial output head: predict the count mean, not a log-space median.

Why a count likelihood at all
-----------------------------
The rest of this project trains on ``log1p`` and scores on counts. That is not
free: for ``log(1 + y) = m + e``, ``expm1(m)`` is the conditional **median**,
while RMSE is minimised by the conditional **mean**
``E[y] = exp(m) E[exp(e)] - 1``. The pipeline therefore carries a downward bias
by construction, growing with residual variance -- which is exactly what
``error_diagnosis.json`` measures: A3TGCN is biased **-12.92 on outbreak windows
and +0.38 on quiet ones**.

``run_beat_baseline`` corrects that after the fact. This module removes the need
for a correction instead: the network emits the count mean directly, so there is
no retransformation step to be biased by.

Why negative binomial rather than Poisson
-----------------------------------------
Poisson fixes ``Var = mu``. This target has a variance-to-mean ratio far above 1
(median 13, max 2631, 9.7% zeros), so Poisson would be badly misspecified and
would drive the fit toward the many quiet weeks. NB2 adds a dispersion parameter,
``Var = mu + alpha * mu^2``, which is the standard specification for
overdispersed epidemiological counts and the one the BiLSTM-NB dengue literature
uses.

Parameterisation follows NB2 with ``r = 1 / alpha`` and ``p = r / (r + mu)``.
Both ``mu`` and ``alpha`` are bounded: an unbounded dispersion lets the model buy
likelihood by declaring everything uncertain, the same failure mode
``gaussian_nll`` clamps for.

The residual-over-persistence structure is preserved -- the network predicts a
correction to ``log(1 + y_{t-1})`` rather than the level -- because that is what
moved baseline RMSE from 66 to 45 and is not up for renegotiation here.
"""

from __future__ import annotations

import torch

__all__ = ["ALPHA_MAX", "ALPHA_MIN", "LOG_MU_MAX", "nb_mean", "nb_nll", "split_nb"]

#: Bounds on the NB2 dispersion. ``alpha -> 0`` is the Poisson limit.
ALPHA_MIN: float = 1e-4
ALPHA_MAX: float = 2.0

#: Cap on log-mean, matching the ``log1p`` clip used everywhere else in the
#: pipeline so the two parameterisations describe the same reachable range.
LOG_MU_MAX: float = 12.0


def split_nb(raw: torch.Tensor, persistence_log: torch.Tensor):
    """Split a ``(..., 2)`` head output into ``(mu, alpha)`` on the count scale.

    Args:
        raw: Head output, ``(batch, nodes, horizon, 2)``. Channel 0 is a residual
            in log space; channel 1 is an unconstrained dispersion logit.
        persistence_log: ``log(1 + y_{t-1})``, broadcast over the horizon.

    Returns:
        ``(mu, alpha)``, both strictly positive and ``mu`` on raw counts.
    """
    log_mu = (raw[..., 0] + persistence_log).clamp(0.0, LOG_MU_MAX)
    mu = torch.expm1(log_mu).clamp_min(1e-6)
    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * torch.sigmoid(raw[..., 1])
    return mu, alpha


def nb_nll(mu: torch.Tensor, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of ``target`` under NB2(``mu``, ``alpha``).

    ``Var = mu + alpha * mu^2``. Constant terms in the target are kept because
    ``lgamma(y + 1)`` is cheap and dropping it makes the reported loss
    uninterpretable across runs with different targets.
    """
    y = target.clamp_min(0.0)
    r = 1.0 / alpha
    log_r_mu = torch.log(r + mu)
    return -(
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1.0)
        + r * (torch.log(r) - log_r_mu)
        + y * (torch.log(mu) - log_r_mu)
    ).mean()


def nb_mean(mu: torch.Tensor) -> torch.Tensor:
    """Point forecast under squared error: the NB2 mean is ``mu`` itself.

    Trivial, and the whole point. Where the log1p pipeline needs a smearing or
    ``exp(v/2)`` correction to turn a median into a mean, this head has already
    emitted the mean, so the RMSE-optimal point forecast needs no adjustment.
    """
    return mu
