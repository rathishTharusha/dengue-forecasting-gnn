"""Tests for the differentiable SEIR simulator behind the SEIR-GNN.

These pin the properties a network-driven compartmental model cannot do without:
it must never create or destroy people, never produce negative compartments
whatever force of infection the network proposes, grow at the rate the
linearised model predicts, and pass gradients back to the force of infection.

``analysis/lib`` is not a package, so the module is loaded by path. CI installs
only numpy/pytest/ruff, so the whole file skips cleanly where torch is absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_SPEC = importlib.util.spec_from_file_location(
    "seir_sim", Path(__file__).resolve().parent.parent / "analysis" / "lib" / "seir_sim.py"
)
seir_sim = importlib.util.module_from_spec(_SPEC)
sys.modules["seir_sim"] = seir_sim
_SPEC.loader.exec_module(seir_sim)

OMEGA, GAMMA = 1 / 5.9, 1 / 3.04  # Chan & Johansson 2012; Phaijoo & Gurung 2018


def _state(n=1_000_000.0, e=100.0, i=100.0, r=0.0, districts=3):
    s = n - e - i - r
    return torch.tensor([[s, e, i, r]] * districts, dtype=torch.float64)


def test_population_is_conserved():
    torch.manual_seed(0)
    s0 = _state()
    foi = torch.rand(3, 52, dtype=torch.float64) * 0.5
    states, _ = seir_sim.simulate_weeks(s0, foi, OMEGA, GAMMA)
    totals = states.sum(-1)
    assert torch.allclose(totals, totals[:, :1].expand_as(totals), rtol=1e-12)


def test_compartments_stay_non_negative_under_an_absurd_force_of_infection():
    """A network can propose anything; the simulator must not break."""
    s0 = _state()
    foi = torch.full((3, 10), 50.0, dtype=torch.float64)
    states, incidence = seir_sim.simulate_weeks(s0, foi, OMEGA, GAMMA)
    assert (states >= 0).all()
    assert torch.isfinite(incidence).all()


def test_no_force_of_infection_means_no_new_infections():
    s0 = _state()
    states, _ = seir_sim.simulate_weeks(s0, torch.zeros(3, 20, dtype=torch.float64), OMEGA, GAMMA)
    assert torch.allclose(states[:, -1, 0], s0[:, 0])
    assert states[:, -1, 1].max() < 1.0, "exposed class should drain"


def test_negative_force_of_infection_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        seir_sim.simulate_weeks(_state(), -torch.ones(3, 2, dtype=torch.float64), OMEGA, GAMMA)


def test_weekly_incidence_is_the_E_to_I_flow():
    """Summed over the run, incidence equals everyone who passed through E."""
    s0 = _state(e=0.0, i=500.0)
    foi = torch.full((3, 60), 0.02, dtype=torch.float64)
    states, incidence = seir_sim.simulate_weeks(s0, foi, OMEGA, GAMMA)
    entered_i = incidence.sum(-1)
    left_s = s0[:, 0] - states[:, -1, 0]
    still_in_e = states[:, -1, 1]
    assert torch.allclose(entered_i + still_in_e, left_s + s0[:, 1], rtol=1e-10)


def test_early_growth_matches_the_linearised_model():
    """With lambda = beta I / N and S ~ N, infections grow at the leading eigenvalue."""
    beta = 0.6
    jac = np.array([[-OMEGA, beta], [OMEGA, -GAMMA]])
    r_true = float(np.max(np.linalg.eigvals(jac).real))

    state = _state(n=1e9, e=10.0, i=10.0, districts=1)
    betas = torch.full((1, 16), beta, dtype=torch.float64)
    errors = []
    for substeps in (7, 28, 140):
        states, _, _ = seir_sim.simulate_closed_loop(state, betas, OMEGA, GAMMA, substeps=substeps)
        sizes = (states[0, :, 1] + states[0, :, 2]).numpy()
        r_sim = np.log(sizes[-1] / sizes[-5]) / (4 * 7)
        errors.append(abs(r_sim - r_true) / r_true)
    # First-order scheme: the error must shrink with the step, and vanish in the limit.
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.01, f"finest step still {errors[2]:.1%} off the linearised growth rate"


def test_holding_the_force_of_infection_for_a_week_understates_growth():
    """Why synthetic truth uses the closed loop, not weekly-held lambda."""
    beta = 0.6
    state = _state(n=1e9, e=10.0, i=10.0, districts=1)
    closed, _, _ = seir_sim.simulate_closed_loop(
        state, torch.full((1, 12), beta, dtype=torch.float64), OMEGA, GAMMA, substeps=28
    )
    held = state
    for _ in range(12):
        lam = seir_sim.closed_loop_foi(held, beta)[..., None]
        held = seir_sim.simulate_weeks(held, lam, OMEGA, GAMMA, substeps=28)[0][:, -1]
    assert float(held[0, 2]) < float(closed[0, -1, 2])


def test_coupling_spreads_infection_to_a_district_with_none():
    """The metapopulation term the graph layer is meant to learn."""
    state = torch.tensor(
        [[1e6 - 200, 100.0, 100.0, 0.0], [1e6, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    betas = torch.full((2, 8), 0.5, dtype=torch.float64)
    isolated, _, _ = seir_sim.simulate_closed_loop(state, betas, OMEGA, GAMMA)
    mixing = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64)
    coupled, _, foi = seir_sim.simulate_closed_loop(state, betas, OMEGA, GAMMA, coupling=mixing)
    assert float(isolated[1, -1, 2]) == 0.0
    assert float(coupled[1, -1, 2]) > 0.0
    assert (foi >= 0).all()


def test_gradients_reach_the_force_of_infection():
    s0 = _state()
    foi = torch.full((3, 8), 0.05, dtype=torch.float64, requires_grad=True)
    _, incidence = seir_sim.simulate_weeks(s0, foi, OMEGA, GAMMA)
    incidence.sum().backward()
    assert foi.grad is not None and (foi.grad > 0).all()


def test_too_coarse_a_step_is_refused():
    """Liu et al.'s weekly Euler step is exactly the case this guards against."""
    with pytest.raises(ValueError, match="too fast"):
        seir_sim.simulate_weeks(
            _state(), torch.zeros(3, 2, dtype=torch.float64), OMEGA, GAMMA, substeps=1
        )


def test_reset_immunity_moves_exactly_the_fraction():
    state = torch.tensor([[100.0, 5.0, 5.0, 890.0]], dtype=torch.float64)
    out = seir_sim.reset_immunity(state, 0.25)
    assert out[0, 0].item() == pytest.approx(100.0 + 222.5)
    assert out[0, 3].item() == pytest.approx(890.0 - 222.5)
    assert out.sum().item() == pytest.approx(state.sum().item())
