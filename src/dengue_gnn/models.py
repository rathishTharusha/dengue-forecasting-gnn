"""Spatio-temporal GNN models for district-level dengue forecasting.

Reconstructed from the specification in ``paper/sections/03_framework.tex`` after
the original implementation was lost (it lived in a git-ignored ``scratch/``
directory). Reconstruction notes and deliberate deviations are marked
``REVIEW:`` below and recorded in ``docs/PHASE2_REVIEW.md``.

The models here all share one propagation path so that ablations differ in
exactly one thing at a time:

    H1 = ReLU(A @ X  @ W1)
    H2 = ReLU(A @ H1 @ W2)
    Y  = Linear(Dropout(H2))

``A`` is what changes between configurations:

* ``AdaptiveGCN(use_adaptive=False)`` -- ``A = A_fixed``, the geographic graph.
* ``AdaptiveGCN(use_adaptive=True)``  -- ``A = σ(g)·A_fixed + (1−σ(g))·A_adp``,
  the Graph WaveNet-style gated blend.

Having both behind one class is the point. The Phase-2 paper compared its dense
adaptive model against a PyTorch Geometric ``GCNConv`` baseline and attributed
the whole RMSE difference to the learned adjacency -- but ``GCNConv`` applies
symmetric normalisation ``D^-1/2 (A+I) D^-1/2`` while the dense path uses
row-normalisation, so the propagation operator changed too. ``use_adaptive=False``
is the missing control that separates those two variables.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "AdaptiveAdjacency",
    "AdaptiveGCN",
    "build_fixed_adjacency",
    "row_normalize",
]


def row_normalize(adj: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale each row of ``adj`` to sum to 1.

    Row-normalisation makes each district's incoming influence a weighted average
    of its neighbours, which keeps activations on a comparable scale regardless of
    how many neighbours a district has.

    REVIEW: The Phase-2 paper does not state how ``A_fixed`` was normalised. Row
    normalisation is chosen here because it matches the row-softmax that
    :class:`AdaptiveAdjacency` applies to the learned graph -- so blending the two
    combines like with like. It is *not* what ``GCNConv`` does; see the module
    docstring.
    """
    return adj / adj.sum(dim=1, keepdim=True).clamp_min(eps)


def build_fixed_adjacency(
    adj_list: dict[str, list[str]],
    node_order: list[str],
    self_loops: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Build the geographic adjacency matrix from a district adjacency list.

    Args:
        adj_list: Mapping from district name to its list of neighbouring districts.
        node_order: District names in the order they appear on axis 1 of the
            feature array. This ordering is load-bearing -- a mismatch silently
            scrambles the graph, so it is passed explicitly rather than inferred
            from ``adj_list`` key order.
        self_loops: Add the identity, so a district's own history informs it.
        normalize: Apply :func:`row_normalize`.

    Returns:
        Float tensor of shape ``(N, N)``.

    Raises:
        ValueError: If a name in ``adj_list`` is absent from ``node_order``.
    """
    index = {name: i for i, name in enumerate(node_order)}
    n = len(node_order)
    adj = torch.zeros(n, n, dtype=torch.float32)

    for district, neighbours in adj_list.items():
        if district not in index:
            raise ValueError(f"district {district!r} missing from node_order")
        i = index[district]
        for neighbour in neighbours:
            if neighbour not in index:
                raise ValueError(f"neighbour {neighbour!r} missing from node_order")
            adj[i, index[neighbour]] = 1.0

    if self_loops:
        adj = adj + torch.eye(n)
    if normalize:
        adj = row_normalize(adj)
    return adj


class AdaptiveAdjacency(nn.Module):
    """Graph WaveNet-style self-adaptive adjacency, ``softmax(ReLU(E1 E2ᵀ))``.

    Two free embedding matrices are learned by gradient descent. Their product
    scores every ordered district pair; ``ReLU`` zeroes the negatives so the graph
    comes out sparse rather than fully connected, and the row softmax makes each
    district's incoming weights sum to 1.

    Nothing ties these weights to geography -- the model is free to discover that
    two districts are epidemiologically coupled despite not sharing a border.

    Args:
        n_nodes: Number of districts.
        emb_dim: Embedding width ``d``. With ``N=25`` this is the whole capacity
            of the learned graph: ``2·N·d`` parameters.
        init_scale: Standard deviation of the embedding initialisation.
    """

    def __init__(self, n_nodes: int, emb_dim: int = 10, init_scale: float = 0.1) -> None:
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * init_scale)
        self.e2 = nn.Parameter(torch.randn(n_nodes, emb_dim) * init_scale)

    def forward(self) -> torch.Tensor:
        """Return the learned adjacency, shape ``(N, N)``, rows summing to 1."""
        return F.softmax(F.relu(self.e1 @ self.e2.t()), dim=1)


class AdaptiveGCN(nn.Module):
    """Two-layer dense GCN with an optional learned, gated adjacency.

    Args:
        in_dim: Input features per node, i.e. ``window × n_features``.
        hidden: Hidden width of both graph layers.
        horizon: Number of future weeks predicted per node.
        n_nodes: Number of districts.
        adj_fixed: Geographic adjacency, shape ``(N, N)``. Registered as a buffer,
            so it moves with ``.to(device)`` but is never trained.
        emb_dim: Embedding width for the adaptive graph.
        dropout: Dropout probability before the prediction head.
        use_adaptive: When ``False`` the model is the fixed-graph control and no
            embedding or gate parameters are created at all.
        gate_init: Initial value of the raw gate ``g``. The default 1.5 gives
            ``σ(g) ≈ 0.82``, so training starts leaning on geography and shifts
            toward the learned graph only if the data supports it.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        horizon: int,
        n_nodes: int,
        adj_fixed: torch.Tensor,
        emb_dim: int = 10,
        dropout: float = 0.1,
        use_adaptive: bool = True,
        gate_init: float = 1.5,
    ) -> None:
        super().__init__()
        if adj_fixed.shape != (n_nodes, n_nodes):
            raise ValueError(
                f"adj_fixed must be ({n_nodes}, {n_nodes}), got {tuple(adj_fixed.shape)}"
            )

        self.n_nodes = n_nodes
        self.horizon = horizon
        self.dropout = dropout
        self.use_adaptive = use_adaptive

        self.register_buffer("adj_fixed", adj_fixed.clone().float())

        if use_adaptive:
            self.adaptive = AdaptiveAdjacency(n_nodes, emb_dim)
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        else:
            self.adaptive = None
            self.gate = None

        self.w1 = nn.Linear(in_dim, hidden)
        self.w2 = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, horizon))

    def blended_adjacency(self) -> torch.Tensor:
        """The adjacency actually used for message passing this forward pass.

        Returns ``A_fixed`` unchanged in the control configuration, and
        ``σ(g)·A_fixed + (1−σ(g))·A_adp`` when the adaptive graph is enabled.
        """
        if not self.use_adaptive:
            return self.adj_fixed
        gate = torch.sigmoid(self.gate)
        return gate * self.adj_fixed + (1.0 - gate) * self.adaptive()

    def gate_value(self) -> float:
        """``σ(g)`` as a plain float, for logging. ``1.0`` in the control."""
        if not self.use_adaptive:
            return 1.0
        return float(torch.sigmoid(self.gate).detach().cpu())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict ``horizon`` future weeks for every node.

        Args:
            x: Node features, shape ``(batch, n_nodes, in_dim)``.

        Returns:
            Predictions of shape ``(batch, n_nodes, horizon)``, in whatever space
            the caller is training in (this project uses log1p residual-over-
            persistence space -- see :mod:`dengue_gnn.losses`).
        """
        if x.dim() != 3 or x.shape[1] != self.n_nodes:
            raise ValueError(f"expected (batch, {self.n_nodes}, in_dim), got {tuple(x.shape)}")

        adj = self.blended_adjacency()
        # einsum keeps the batch axis untouched while mixing across nodes:
        # (N,N) x (B,N,F) -> (B,N,F)
        h = F.relu(torch.einsum("ij,bjf->bif", adj, self.w1(x)))
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(torch.einsum("ij,bjf->bif", adj, self.w2(h)))
        return self.head(h)
