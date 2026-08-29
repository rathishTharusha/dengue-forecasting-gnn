"""Tests for dengue_gnn.losses.

The space-of-application tests encode finding F5 from docs/PHASE2_REVIEW.md: the
non-negativity constraint is a statement about case counts, and applying it to
the network's residual-space output would penalise the model for forecasting a
decline in cases.
"""

import pytest

torch = pytest.importorskip("torch")

from dengue_gnn.losses import (  # noqa: E402
    nonnegativity_loss,
    smoothness_loss,
    spatial_regularisation,
)

N = 3
H = 2


@pytest.fixture
def chain_adj():
    """A -- B -- C, row-normalised with self-loops."""
    a = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    return a / a.sum(dim=1, keepdim=True)


# --------------------------------------------------------------------------
# smoothness
# --------------------------------------------------------------------------


def test_smoothness_is_zero_when_all_districts_agree(chain_adj):
    pred = torch.full((4, N, H), 42.0)
    assert smoothness_loss(pred, chain_adj) == pytest.approx(0.0)


def test_smoothness_grows_with_disagreement(chain_adj):
    mild = torch.tensor([[[0.0], [1.0], [2.0]]]).float()
    wild = torch.tensor([[[0.0], [100.0], [200.0]]]).float()
    assert smoothness_loss(wild, chain_adj) > smoothness_loss(mild, chain_adj)


def test_smoothness_ignores_unconnected_pairs():
    """With no edges at all, nothing can be penalised."""
    pred = torch.tensor([[[0.0], [500.0], [0.0]]]).float()
    no_edges = torch.zeros(N, N)
    assert smoothness_loss(pred, no_edges) == pytest.approx(0.0)


def test_smoothness_is_differentiable(chain_adj):
    pred = torch.randn(2, N, H, requires_grad=True)
    smoothness_loss(pred, chain_adj).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_smoothness_rejects_wrong_rank(chain_adj):
    with pytest.raises(ValueError, match="batch, n_nodes, horizon"):
        smoothness_loss(torch.randn(N, H), chain_adj)


def test_smoothness_rejects_mismatched_adjacency():
    with pytest.raises(ValueError, match="adj must be"):
        smoothness_loss(torch.randn(1, N, H), torch.eye(N + 1))


# --------------------------------------------------------------------------
# non-negativity
# --------------------------------------------------------------------------


def test_nonnegativity_is_zero_for_valid_counts():
    assert nonnegativity_loss(torch.tensor([0.0, 5.0, 900.0])) == pytest.approx(0.0)


def test_nonnegativity_penalises_negative_counts():
    assert nonnegativity_loss(torch.tensor([-3.0, 0.0])) == pytest.approx(9.0 / 2)


def test_nonnegativity_on_counts_vs_residuals_differ():
    """F5: the same forecast, scored in two spaces, gives opposite verdicts.

    A district going from 100 cases to 60 is a legitimate, desirable forecast.
    On the count scale it is unpenalised. On the residual scale the value is
    negative and would be penalised -- biasing the model against ever predicting
    a decline.
    """
    last_week = torch.tensor([100.0])
    forecast_counts = torch.tensor([60.0])
    forecast_residual = forecast_counts - last_week  # what the network emits

    assert nonnegativity_loss(forecast_counts) == pytest.approx(0.0)
    assert nonnegativity_loss(forecast_residual) > 0.0


# --------------------------------------------------------------------------
# combined
# --------------------------------------------------------------------------


def test_lambda_zero_contributes_nothing(chain_adj):
    pred = torch.randn(2, N, H)
    assert spatial_regularisation(pred, chain_adj, lambda_phys=0.0) == pytest.approx(0.0)


def test_lambda_zero_keeps_the_graph_connected(chain_adj):
    """The zero must still be differentiable, or the lambda=0 row would crash."""
    pred = torch.randn(2, N, H, requires_grad=True)
    spatial_regularisation(pred, chain_adj, lambda_phys=0.0).backward()
    assert pred.grad is not None


def test_regularisation_scales_with_lambda(chain_adj):
    pred = torch.tensor([[[0.0], [50.0], [-10.0]]]).float()
    small = spatial_regularisation(pred, chain_adj, lambda_phys=0.01)
    large = spatial_regularisation(pred, chain_adj, lambda_phys=1.0)
    assert large == pytest.approx(float(small) * 100, rel=1e-4)


def test_cons_weight_applies(chain_adj):
    pred = torch.tensor([[[-5.0], [-5.0], [-5.0]]]).float()  # smoothness is 0 here
    base = spatial_regularisation(pred, chain_adj, lambda_phys=1.0, cons_weight=1.0)
    heavy = spatial_regularisation(pred, chain_adj, lambda_phys=1.0, cons_weight=10.0)
    assert heavy == pytest.approx(float(base) * 10, rel=1e-4)
