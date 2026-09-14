"""Differentiable human SEIR simulator for the SEIR-GNN.

The SEIR-GNN (Liu et al. 2025, with the LSTM replaced by a graph network)
predicts each district's force of infection ``lambda_i(t)`` and lets a
compartmental model turn it into new infections. This module is that
compartmental model. It is deliberately small, because everything the network
learns has to flow back through it.

    dS/dt = -lambda S
    dE/dt =  lambda S - omega E
    dI/dt =  omega E  - gamma I
    dR/dt =  gamma I

Why daily steps, and why not plain Euler
----------------------------------------
Liu et al. step the model once per week with forward Euler. With their
``gamma = 1/week`` that makes ``gamma * dt = 1``, so the whole infectious class
empties every step, and with Phaijoo & Gurung's daily rates ``gamma_h * 7 = 2.3``
the scheme is unstable. The model therefore takes ``substeps`` steps per week.

Each step moves compartments with **exponential flows** --
``S * (1 - exp(-lambda dt))`` leaves S in a step, and so on. For rates held
constant over the step this is the exact outflow of each compartment, and for
any rate it keeps every compartment non-negative and the total population fixed.
Forward Euler has neither property, and a network that proposes a large
``lambda`` would otherwise be able to drive S below zero.

Rates are **per day** throughout, matching ``dengue_gnn.seir``.

Weekly incidence is the flow E -> I summed over the week's steps: the number of
people who became infectious that week, which is what surveillance can report.
"""

from __future__ import annotations

import torch

__all__ = ["DAYS_PER_WEEK", "closed_loop_foi", "reset_immunity", "simulate_closed_loop",
           "simulate_weeks"]

DAYS_PER_WEEK = 7


def _step(state: torch.Tensor, lam: torch.Tensor, omega: float, gamma: float,
          dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One exponential-flow step. Returns ``(new_state, flow_E_to_I)``."""
    s, e, i, r = state.unbind(-1)
    infect = s * (1.0 - torch.exp(-lam * dt))
    onset = e * (1.0 - torch.exp(torch.as_tensor(-omega * dt, dtype=state.dtype)))
    recover = i * (1.0 - torch.exp(torch.as_tensor(-gamma * dt, dtype=state.dtype)))
    new = torch.stack([s - infect, e + infect - onset, i + onset - recover, r + recover], dim=-1)
    return new, onset


def simulate_weeks(state0: torch.Tensor, foi: torch.Tensor, omega: float, gamma: float,
                   substeps: int = DAYS_PER_WEEK) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the SEIR forward, one force of infection per district per week.

    Args:
        state0: ``(..., 4)`` compartment counts ``(S, E, I, R)`` at the start.
        foi: ``(..., weeks)`` force of infection per day, held constant within a week.
            Must be non-negative; the SEIR-GNN guarantees it with a bounded sigmoid.
        omega: E -> I rate per day (1 / intrinsic incubation period).
        gamma: I -> R rate per day (1 / infectious period).
        substeps: Steps per week. The default steps one day at a time.

    Returns:
        ``(states, incidence)``: ``states`` is ``(..., weeks + 1, 4)`` including
        the start, and ``incidence`` is ``(..., weeks)`` new infectious people
        per week.
    """
    if (foi < 0).any():
        raise ValueError("force of infection must be non-negative")
    dt = DAYS_PER_WEEK / substeps
    if max(omega, gamma) * dt > 1.0:
        raise ValueError(f"rates too fast for {substeps} substeps: omega*dt={omega * dt:.2f}, "
                         f"gamma*dt={gamma * dt:.2f}")
    states, weekly = [state0], []
    state = state0
    for w in range(foi.shape[-1]):
        lam = foi[..., w]
        total = torch.zeros_like(lam)
        for _ in range(substeps):
            state, onset = _step(state, lam, omega, gamma, dt)
            total = total + onset
        states.append(state)
        weekly.append(total)
    return torch.stack(states, dim=-2), torch.stack(weekly, dim=-1)


def closed_loop_foi(state: torch.Tensor, beta: float | torch.Tensor) -> torch.Tensor:
    """Mass-action force of infection ``beta * I / N``, for simulations and tests.

    The SEIR-GNN does not use this: its force of infection comes from the network.
    The twin experiment does, to generate a synthetic epidemic with a known answer.
    """
    n = state.sum(-1).clamp_min(1.0)
    return beta * state[..., 2] / n


def simulate_closed_loop(state0: torch.Tensor, beta: torch.Tensor, omega: float, gamma: float,
                         substeps: int = DAYS_PER_WEEK,
                         coupling: torch.Tensor | None = None,
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mass-action epidemic with the force of infection recomputed every step.

    Args:
        state0: ``(districts, 4)`` compartments.
        beta: ``(districts, weeks)`` transmission rate per day, held within a week.
        omega, gamma: rates per day.
        substeps: steps per week.
        coupling: optional ``(districts, districts)`` row-stochastic mixing matrix.
            District i's force of infection is then
            ``beta_i * sum_j C_ij I_j / N_j`` -- the metapopulation form the
            SEIR-GNN's graph layer is meant to learn.

    Returns:
        ``(states, incidence, foi)``: weekly states ``(districts, weeks + 1, 4)``,
        weekly incidence ``(districts, weeks)``, and the **week-averaged** force of
        infection ``(districts, weeks)``, the quantity a weekly model can recover.

    Unlike :func:`simulate_weeks`, feedback here acts within the week. Holding the
    force of infection fixed for a whole week while prevalence grows understates
    early growth, by about 30% at dengue-like rates, so this is what synthetic
    truth is generated with.
    """
    dt = DAYS_PER_WEEK / substeps
    if max(omega, gamma) * dt > 1.0:
        raise ValueError(f"rates too fast for {substeps} substeps")
    state, states, weekly, foi_avg = state0, [state0], [], []
    for w in range(beta.shape[-1]):
        total = torch.zeros(state.shape[:-1], dtype=state.dtype)
        lam_sum = torch.zeros_like(total)
        for _ in range(substeps):
            prevalence = state[..., 2] / state.sum(-1).clamp_min(1.0)
            if coupling is not None:
                prevalence = coupling @ prevalence
            lam = beta[..., w] * prevalence
            state, onset = _step(state, lam, omega, gamma, dt)
            total = total + onset
            lam_sum = lam_sum + lam
        states.append(state)
        weekly.append(total)
        foi_avg.append(lam_sum / substeps)
    return torch.stack(states, dim=-2), torch.stack(weekly, dim=-1), torch.stack(foi_avg, dim=-1)


def reset_immunity(state: torch.Tensor, fraction: float | torch.Tensor) -> torch.Tensor:
    """Return a fraction of the recovered class to susceptible.

    Models a serotype switch -- DENV-2 in 2016-17, DENV-3 in late 2019 -- to which
    earlier recovery gives no lasting protection.
    """
    s, e, i, r = state.unbind(-1)
    moved = r * fraction
    return torch.stack([s + moved, e, i, r - moved], dim=-1)
