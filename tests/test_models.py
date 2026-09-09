"""Tests for dengue_gnn.models.

Several of these encode findings from docs/PHASE2_REVIEW.md so the issues cannot
silently return: notably that the fixed-graph control shares the propagation path
with the adaptive model (F6), and that the gate starts where the paper says it
does (F11).
"""

import pytest

torch = pytest.importorskip("torch")

from dengue_gnn.models import (  # noqa: E402
    AdaptiveAdjacency,
    AdaptiveGCN,
    build_fixed_adjacency,
    row_normalize,
)

N_NODES = 4
IN_DIM = 6
HIDDEN = 8
HORIZON = 3


@pytest.fixture
def adj():
    a = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    return row_normalize(a + torch.eye(N_NODES))


def make_model(adj, **kw):
    torch.manual_seed(0)
    return AdaptiveGCN(
        in_dim=IN_DIM,
        hidden=HIDDEN,
        horizon=HORIZON,
        n_nodes=N_NODES,
        adj_fixed=adj,
        **kw,
    )


# --------------------------------------------------------------------------
# adjacency construction
# --------------------------------------------------------------------------


def test_row_normalize_rows_sum_to_one():
    a = torch.tensor([[1.0, 1.0, 0.0], [2.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(row_normalize(a).sum(dim=1), torch.ones(3))


def test_row_normalize_survives_isolated_node():
    """An all-zero row must not produce nan."""
    a = torch.zeros(2, 2)
    assert torch.isfinite(row_normalize(a)).all()


def test_build_fixed_adjacency_respects_node_order():
    adj_list = {"A": ["B"], "B": ["A"], "C": []}
    order = ["C", "B", "A"]
    m = build_fixed_adjacency(adj_list, order, self_loops=False, normalize=False)
    # A is index 2, B is index 1
    assert m[2, 1] == 1.0
    assert m[1, 2] == 1.0
    assert m[0].sum() == 0.0


def test_build_fixed_adjacency_adds_self_loops():
    m = build_fixed_adjacency({"A": [], "B": []}, ["A", "B"], normalize=False)
    assert m[0, 0] == 1.0 and m[1, 1] == 1.0


def test_build_fixed_adjacency_rejects_unknown_district():
    with pytest.raises(ValueError, match="missing from node_order"):
        build_fixed_adjacency({"A": ["Z"]}, ["A"])


def test_build_fixed_adjacency_rejects_one_directional_edge():
    """A shared border cannot be directional; a lone entry is an authoring bug.

    Found on the project's own data: sri_lanka_adj_list.json had Kandy -> Ampara
    with no reciprocal entry, which silently made message passing one-way
    between them.
    """
    adj_list = {"A": ["B"], "B": []}
    with pytest.raises(ValueError, match="one-directional"):
        build_fixed_adjacency(adj_list, ["A", "B"])


def test_build_fixed_adjacency_symmetric_check_can_be_disabled():
    adj_list = {"A": ["B"], "B": []}
    m = build_fixed_adjacency(adj_list, ["A", "B"], require_symmetric=False, normalize=False)
    assert m[0, 1] == 1.0 and m[1, 0] == 0.0


# --------------------------------------------------------------------------
# adaptive adjacency
# --------------------------------------------------------------------------


def test_adaptive_adjacency_rows_sum_to_one():
    torch.manual_seed(0)
    a = AdaptiveAdjacency(N_NODES, emb_dim=5)()
    assert a.shape == (N_NODES, N_NODES)
    assert torch.allclose(a.sum(dim=1), torch.ones(N_NODES), atol=1e-6)


def test_adaptive_adjacency_is_trainable():
    torch.manual_seed(0)
    mod = AdaptiveAdjacency(N_NODES, emb_dim=5)
    mod().sum().backward()
    assert mod.e1.grad is not None
    assert mod.e2.grad is not None


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


def test_forward_shape(adj):
    model = make_model(adj)
    out = model(torch.randn(7, N_NODES, IN_DIM))
    assert out.shape == (7, N_NODES, HORIZON)


def test_forward_rejects_wrong_node_count(adj):
    model = make_model(adj)
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(2, N_NODES + 1, IN_DIM))


def test_gate_starts_where_the_paper_says(adj):
    """F11: the paper states initialisation at sigma(g) ~ 0.82."""
    model = make_model(adj, use_adaptive=True, gate_init=1.5)
    assert model.gate_value() == pytest.approx(0.8176, abs=1e-3)


def test_control_uses_fixed_adjacency_exactly(adj):
    """F6: use_adaptive=False must be the pure geographic graph."""
    model = make_model(adj, use_adaptive=False)
    assert torch.allclose(model.blended_adjacency(), adj)
    assert model.gate_value() == 1.0


def test_control_creates_no_graph_parameters(adj):
    """The control must not carry unused embeddings that still receive weight decay."""
    control = make_model(adj, use_adaptive=False)
    names = {n for n, _ in control.named_parameters()}
    assert not any("adaptive" in n or n == "gate" for n in names)

    adaptive = make_model(adj, use_adaptive=True)
    names = {n for n, _ in adaptive.named_parameters()}
    assert "gate" in names
    assert any("adaptive.e1" in n for n in names)


def test_control_and_adaptive_share_the_propagation_path(adj):
    """F6 in one assertion.

    With the gate forced fully open (sigma(g) -> 1), the adaptive model must
    reduce *exactly* to the control. If the two ever diverge here, the ablation
    is confounding the learned graph with a change of operator -- which is the
    mistake the Phase-2 paper made against the PyG GCNConv baseline.
    """
    x = torch.randn(5, N_NODES, IN_DIM)

    control = make_model(adj, use_adaptive=False).eval()
    adaptive = make_model(adj, use_adaptive=True, gate_init=40.0).eval()

    # copy the shared weights so only the adjacency can differ
    adaptive.w1.load_state_dict(control.w1.state_dict())
    adaptive.w2.load_state_dict(control.w2.state_dict())
    adaptive.head.load_state_dict(control.head.state_dict())

    with torch.no_grad():
        assert torch.allclose(control(x), adaptive(x), atol=1e-5)


def test_blend_moves_with_the_gate(adj):
    model = make_model(adj, use_adaptive=True, gate_init=0.0)  # sigma = 0.5
    blend = model.blended_adjacency()
    expected = 0.5 * adj + 0.5 * model.adaptive()
    assert torch.allclose(blend, expected, atol=1e-6)


def test_blended_adjacency_rows_sum_to_one(adj):
    """Both components are row-stochastic, so any convex blend must be too."""
    model = make_model(adj, use_adaptive=True)
    rows = model.blended_adjacency().sum(dim=1)
    assert torch.allclose(rows, torch.ones(N_NODES), atol=1e-5)


def test_adj_fixed_is_a_buffer_not_a_parameter(adj):
    model = make_model(adj)
    assert "adj_fixed" in dict(model.named_buffers())
    assert "adj_fixed" not in dict(model.named_parameters())


def test_rejects_mismatched_adjacency():
    with pytest.raises(ValueError, match="adj_fixed must be"):
        AdaptiveGCN(
            in_dim=IN_DIM,
            hidden=HIDDEN,
            horizon=HORIZON,
            n_nodes=N_NODES,
            adj_fixed=torch.eye(N_NODES + 1),
        )
