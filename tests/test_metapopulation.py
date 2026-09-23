"""The metapopulation SEIR used by the ``foi_meta`` head.

``seir_sim.simulate_closed_loop`` now multiplies prevalence as ``p @ C^T`` so a
batch axis works. These pin that the batched result equals running each sample
alone, that coupling actually moves infection across districts, and that the
head starts out at the border graph.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "seirgnn2"))

import seir_sim  # noqa: E402


def _setup(batch=3, n=4, weeks=3):
    g = torch.Generator().manual_seed(0)
    i0 = torch.rand(batch, n, generator=g) * 1e-3
    s0 = torch.full((batch, n), 0.3)
    e0 = i0 * 0.5
    state = torch.stack([s0, e0, i0, 1 - s0 - e0 - i0], -1)
    beta = 0.2 + torch.rand(batch, n, weeks, generator=g)
    c = torch.softmax(torch.rand(n, n, generator=g), -1)
    return state, beta, c


def test_batched_equals_one_sample_at_a_time():
    state, beta, c = _setup()
    _, inc, _ = seir_sim.simulate_closed_loop(state, beta, 0.1, 1 / 7, coupling=c)
    for b in range(state.shape[0]):
        _, one, _ = seir_sim.simulate_closed_loop(state[b], beta[b], 0.1, 1 / 7, coupling=c)
        torch.testing.assert_close(inc[b], one)


def test_coupling_moves_infection_across_districts():
    state, beta, _ = _setup(batch=1, n=2)
    state[0, 1, 1:3] = 0.0                      # district 1 starts with no E and no I
    iso = torch.eye(2)
    mix = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    _, a, _ = seir_sim.simulate_closed_loop(state, beta, 0.1, 1 / 7, coupling=iso)
    _, b, _ = seir_sim.simulate_closed_loop(state, beta, 0.1, 1 / 7, coupling=mix)
    assert float(a[0, 1].sum()) == 0.0          # isolated: nothing ever arrives
    assert float(b[0, 1].sum()) > 0.0           # coupled: it does


def test_foi_meta_coupling_starts_at_the_border_graph():
    import models
    fixed = torch.softmax(torch.rand(25, 25), -1)
    net = models.Net(7, 25, head="foi_meta", backbone="gcn")
    c = torch.softmax(torch.log(fixed.clamp_min(1e-9)) + net.c_logit, -1)
    torch.testing.assert_close(c, fixed)
