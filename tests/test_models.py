"""Unit tests for baseline GCN/GAT architectures and AdaptiveGCN."""

import pytest
import torch

from dengue_gnn.models import (
    AdaptiveGCN,
    DenseGraphConv,
    GNNBaseline,
    create_adaptive_model,
    create_baseline_model,
    edge_index_to_dense_adj,
)


@pytest.fixture
def dummy_graph():
    num_nodes = 25
    in_dim = 33
    horizon = 3
    x = torch.randn(num_nodes, in_dim)
    # Complete self-loops + ring edges for testing
    src = list(range(num_nodes)) + list(range(num_nodes))
    dst = list(range(num_nodes)) + [(i + 1) % num_nodes for i in range(num_nodes)]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return x, edge_index, num_nodes, in_dim, horizon


def test_gcn_forward_and_backward(dummy_graph):
    x, edge_index, num_nodes, in_dim, horizon = dummy_graph
    model = create_baseline_model(in_dim=in_dim, hidden=64, horizon=horizon, kind="GCN")

    out = model(x, edge_index)
    assert out.shape == (num_nodes, horizon)

    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_gat_forward_and_backward(dummy_graph):
    x, edge_index, num_nodes, in_dim, horizon = dummy_graph
    model = create_baseline_model(
        in_dim=in_dim, hidden=64, horizon=horizon, kind="GAT", heads=8, dropout=0.0
    )

    out = model(x, edge_index)
    assert out.shape == (num_nodes, horizon)

    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_dense_graph_conv(dummy_graph):
    x, _, num_nodes, in_dim, _ = dummy_graph
    adj = torch.eye(num_nodes)
    conv = DenseGraphConv(in_features=in_dim, out_features=16)

    out = conv(x, adj)
    assert out.shape == (num_nodes, 16)

    loss = out.sum()
    loss.backward()
    assert conv.weight.grad is not None
    assert conv.bias.grad is not None


def test_edge_index_to_dense_adj(dummy_graph):
    _, edge_index, num_nodes, _, _ = dummy_graph
    adj = edge_index_to_dense_adj(edge_index, n_nodes=num_nodes, self_loops=True, row_normalize=True)

    assert adj.shape == (num_nodes, num_nodes)
    # Check row-normalization: each row sums to 1.0
    row_sums = adj.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones(num_nodes), rtol=1e-5, atol=1e-5)


def test_adaptive_gcn_gated_blend(dummy_graph):
    x, edge_index, num_nodes, in_dim, horizon = dummy_graph
    fixed_adj = edge_index_to_dense_adj(edge_index, n_nodes=num_nodes)

    model = create_adaptive_model(
        in_dim=in_dim,
        hidden=32,
        horizon=horizon,
        n_nodes=num_nodes,
        emb_dim=8,
        use_adaptive=True,
        use_fixed=True,
        fixed_adj=fixed_adj,
        dropout=0.1,
    )

    # Initial gate weight should be ~0.5
    assert abs(model.get_gate_weight() - 0.5) < 1e-4

    # Adaptive adjacency should be valid probability matrix (row sum = 1)
    adp_adj = model.get_adaptive_adj()
    assert adp_adj.shape == (num_nodes, num_nodes)
    assert torch.all(adp_adj >= 0.0)
    torch.testing.assert_close(adp_adj.sum(dim=-1), torch.ones(num_nodes), rtol=1e-4, atol=1e-4)

    # Forward pass
    out = model(x)
    assert out.shape == (num_nodes, horizon)

    # Backward pass
    loss = out.sum()
    loss.backward()
    assert model.nodevec1.grad is not None
    assert model.nodevec2.grad is not None
    assert model.gate.grad is not None


def test_adaptive_gcn_pure_adaptive(dummy_graph):
    x, _, num_nodes, in_dim, horizon = dummy_graph
    model = create_adaptive_model(
        in_dim=in_dim,
        hidden=32,
        horizon=horizon,
        n_nodes=num_nodes,
        emb_dim=8,
        use_adaptive=True,
        use_fixed=False,
        fixed_adj=None,
    )

    out = model(x)
    assert out.shape == (num_nodes, horizon)
    assert model.gate is None
    assert model.get_gate_weight() == 0.0


def test_invalid_kind():
    with pytest.raises(ValueError):
        create_baseline_model(in_dim=33, hidden=64, horizon=3, kind="UNSUPPORTED")
