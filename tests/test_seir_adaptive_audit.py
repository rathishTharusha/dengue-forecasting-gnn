"""The seir-adaptive audit: the SEIR head and adaptive graph after the fixes.

Each test pins one defect found in the audit so it cannot come back:

* S2 -- the spatial import term was centred over the whole batch, so a forecast
  changed with whichever other windows shared its batch.
* S1 -- ``lam_param="anchor"`` and ``"mass"`` must carry the population scale the
  free ``log`` parameterisation leaves to the network.
* S3 -- ``state_fit="encoder"`` must start exactly where ``state_fit=True`` does.
* A1 -- the original adaptive-graph initialisation is uniform at the start.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "seirgnn2"))

RHO, OMEGA, GAMMA = 1.0 / 11.0, 0.7 / 7.0, 1.0 / 7.0
N = 5


def _net(**kw):
    import models

    torch.manual_seed(0)
    kw = {"head": "foi_res", "backbone": "gcn", "state_fit": True, **kw}
    return models.Net(7, N, **kw).eval()


def _steady(c: torch.Tensor, pop: torch.Tensor, s: float = 0.3) -> torch.Tensor:
    """Fractions at the discrete-time steady state that reports ``c`` cases a week."""
    f = c / (7.0 * RHO * pop)  # daily infections
    e = f / (1.0 - math.exp(-OMEGA))
    i = f / (1.0 - math.exp(-GAMMA))
    s_ = torch.full_like(c, s)
    return torch.stack([s_, e, i, 1.0 - s_ - e - i], -1)


def _inputs(b: int = 4):
    g = torch.Generator().manual_seed(1)
    x = torch.rand(b, N, 7, generator=g)
    fixed = torch.softmax(torch.rand(N, N, generator=g), -1)
    c = 5.0 + 200.0 * torch.rand(b, N, generator=g)
    pop = 1e5 + 2e6 * torch.rand(b, N, generator=g)
    p_z = torch.log1p(c).unsqueeze(-1).expand(b, N, 3).contiguous()
    return x, fixed, p_z, _steady(c, pop), pop, c


@pytest.mark.parametrize("lam", ["log", "anchor"])
def test_forecast_does_not_depend_on_the_rest_of_the_batch(lam):
    net = _net(lam_param=lam)
    with torch.no_grad():
        net.beta.fill_(2.0)  # make the import term bite
    x, fixed, p_z, st0, pop, _ = _inputs()
    with torch.no_grad():
        whole = net(x, fixed, p_z, st0, pop)[0]
        alone = net(x[:1], fixed, p_z[:1], st0[:1], pop[:1])[0]
    torch.testing.assert_close(alone, whole[:1])


def test_anchor_at_zero_output_reproduces_last_week():
    net = _net(head="foi", lam_param="anchor", state_fit=False)
    _, fixed, p_z, st0, pop, c = _inputs()
    raw = torch.zeros(p_z.shape)
    with torch.no_grad():
        counts = torch.expm1(net._physics(raw, st0, pop, 0.0, 1.0, fixed, None, p_z))
    torch.testing.assert_close(counts, c.unsqueeze(-1).expand_as(counts), rtol=0.03, atol=0.5)


def test_anchor_scale_follows_population_not_the_network():
    """Same raw output, ten times the population and cases: ten times the counts."""
    net = _net(head="foi", lam_param="anchor", state_fit=False)
    _, fixed, _, _, pop, c = _inputs()
    raw = torch.full((4, N, 3), 0.3)
    out = []
    for k in (1.0, 10.0):
        st0 = _steady(c * k, pop * k)
        p_z = torch.log1p(c * k).unsqueeze(-1).expand(4, N, 3)
        with torch.no_grad():
            out.append(torch.expm1(net._physics(raw, st0, pop * k, 0.0, 1.0, fixed, None, p_z)))
    torch.testing.assert_close(out[1], 10.0 * out[0], rtol=2e-3, atol=0.5)


def test_mass_action_at_replacement_holds_incidence_steady():
    net = _net(head="foi", lam_param="mass", state_fit=False)
    _, fixed, p_z, st0, pop, c = _inputs()
    s, i = st0[..., 0], st0[..., 2]
    f = c / (7.0 * RHO * pop)
    beta = f / (s * i)  # beta I S = daily infections
    raw = (beta.log() - net.mass_bias.detach()).unsqueeze(-1).expand(4, N, 3)
    with torch.no_grad():
        counts = torch.expm1(net._physics(raw, st0, pop, 0.0, 1.0, fixed, None, p_z))
    torch.testing.assert_close(counts, c.unsqueeze(-1).expand_as(counts), rtol=0.03, atol=0.5)


def test_encoder_e0_starts_where_the_global_fit_does():
    x, fixed, p_z, st0, pop, _ = _inputs()
    with torch.no_grad():
        a = _net(lam_param="anchor", state_fit=True)(x, fixed, p_z, st0, pop)[0]
        b = _net(lam_param="anchor", state_fit="encoder")(x, fixed, p_z, st0, pop)[0]
    torch.testing.assert_close(a, b)


def _entropy(a: torch.Tensor) -> float:
    a = a.detach()
    return float((-(a * a.log()).sum(-1)).mean() / math.log(a.shape[0]))


def test_original_adaptive_init_is_uniform_and_gwn_init_is_not():
    import models

    torch.manual_seed(0)
    small = models.learned_adjacency(*models.adaptive_embeddings(25, "small"))
    gwn = models.learned_adjacency(*models.adaptive_embeddings(25, "gwn"))
    assert _entropy(small) > 0.9999
    assert _entropy(gwn) < 0.9


def test_uniform_backbone_mean_pools():
    import models

    layer = models.GraphLayer(4, N, "uniform")
    fixed = torch.softmax(torch.rand(N, N), -1)
    torch.testing.assert_close(layer.adjacency(fixed), torch.full((N, N), 1.0 / N))


def test_shared_adjacency_is_one_matrix_and_trains():
    net = _net(
        head="residual", backbone="adaptive", adj_init="gwn", adj_shared=True, state_fit=False
    ).train()
    assert len(net.learned_adjacencies()) == 1
    x, fixed, p_z, *_ = _inputs()
    net(x, fixed, p_z)[0].sum().backward()
    assert net.adj_e1.grad is not None and net.adj_e1.grad.abs().sum() > 0
    assert all(layer.e1.grad is None for layer in net.graph)


def test_default_network_has_no_new_parameters():
    names = {n for n, _ in _net().named_parameters()}
    assert not names & {"adj_e1", "adj_e2", "mass_bias", "e_head.weight", "e_head.bias"}


def test_gwn_embeddings_are_exempt_from_weight_decay():
    import train

    net = _net(
        head="residual", backbone="adaptive", adj_init="gwn", adj_shared=True, state_fit=False
    )
    opt = train._optimizer(net, 1e-3, 1e-4, "gwn")
    exempt = {id(p) for g in opt.param_groups if g["weight_decay"] == 0.0 for p in g["params"]}
    assert id(net.adj_e1) in exempt and id(net.graph[0].e1) in exempt
    assert id(net.in_proj[0].weight) not in exempt


def test_frozen9_origins_reproduce_the_protocol_check_value():
    import core
    import corrected_data as cd
    import numpy as np

    try:
        data = cd.load()
    except (FileNotFoundError, OSError):
        pytest.skip("corrected dataset not available")
    folds = core.build_folds(
        data.cases, data.missing, core.WINDOW, core.ORIGINS_F9, core.TEST_FRAC_F9
    )
    per = []
    for f in folds:
        pack = core.build_tensors(data, f, "test", False, False, False)
        per.append(core.rmse(pack["p_raw"].numpy(), pack["y_raw"].numpy()))
    assert round(float(np.mean(per)), 4) == 28.5410
