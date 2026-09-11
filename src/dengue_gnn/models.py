"""Graph Neural Network architectures for dengue forecasting.

Contains:
1. Spatio-temporal GNN baselines (GCN, GAT) over fixed sparse edge_index.
2. DenseGraphConv for dense/continuous graph convolutions.
3. AdaptiveGCN with Graph WaveNet-style self-adaptive node embeddings and gated blending.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


def edge_index_to_dense_adj(
    edge_index: torch.Tensor,
    n_nodes: int = 25,
    self_loops: bool = True,
    row_normalize: bool = True,
) -> torch.Tensor:
    """Convert a PyG edge_index tensor to a dense normalized adjacency matrix.

    Args:
        edge_index: Tensor of shape (2, num_edges).
        n_nodes: Total number of nodes (districts).
        self_loops: Ensure diagonal elements are 1.0.
        row_normalize: If True, apply D^-1 A row normalization.

    Returns:
        adj: Dense float32 tensor of shape (n_nodes, n_nodes).
    """
    adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    src, dst = edge_index[0], edge_index[1]
    adj[src, dst] = 1.0

    if self_loops:
        adj.fill_diagonal_(1.0)

    if row_normalize:
        row_sums = adj.sum(dim=-1, keepdim=True)
        row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        adj = adj / row_sums

    return adj


class DenseGraphConv(nn.Module):
    """Dense spatial graph convolution layer: Z = A H W + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features of shape (N, in_features).
            adj: Adjacency matrix of shape (N, N).

        Returns:
            out: Convolved node features of shape (N, out_features).
        """
        support = torch.matmul(x, self.weight)
        out = torch.matmul(adj, support)
        if self.bias is not None:
            out = out + self.bias
        return out


class GNNBaseline(nn.Module):
    """Spatio-temporal GNN baseline (GCN or GAT) for multi-horizon node forecasting.

    Args:
        in_dim: Input feature dimension per node (e.g. Fd * window = 11 * 3 = 33).
        hidden: Hidden layer dimension (default 64).
        horizon: Multi-step forecasting horizon H (default 3).
        kind: Backbone type, either "GCN" or "GAT".
        heads: Number of attention heads for GAT (default 8).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        horizon: int = 3,
        kind: str = "GCN",
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.kind = kind.upper()
        self.dropout = dropout
        self.in_dim = in_dim
        self.hidden = hidden
        self.horizon = horizon

        if self.kind == "GCN":
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        elif self.kind == "GAT":
            if hidden % heads != 0:
                raise ValueError(f"hidden dimension ({hidden}) must be divisible by heads ({heads})")
            self.conv1 = GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout)
            self.conv2 = GATConv(hidden, hidden, heads=1, concat=True, dropout=dropout)
        else:
            raise ValueError(f"Unsupported GNN kind '{kind}'. Expected 'GCN' or 'GAT'.")

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single time-window graph.

        Args:
            x: Node feature tensor of shape (N, in_dim).
            edge_index: Graph edge tensor of shape (2, num_edges).

        Returns:
            out: Forecast tensor of shape (N, horizon).
        """
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        out = self.head(h)
        return out


class AdaptiveGCN(nn.Module):
    """Adaptive Graph Convolutional Network (Contribution c).

    Learns node embeddings E1, E2 to discover latent inter-district connectivity:
        A_adp = Softmax(ReLU(E1 @ E2.T))

    When combined with the physical/geographic graph:
        A_blend = sigma(g) * A_fixed + (1 - sigma(g)) * A_adp
    where g is a learnable scalar gating parameter initialized at 0.0 (50/50 balance).

    Args:
        in_dim: Input feature dimension per node.
        hidden: Hidden layer dimension (default 64).
        horizon: Forecast horizon (default 3).
        n_nodes: Number of nodes / districts (default 25).
        emb_dim: Embedding dimension for adaptive graph (default 10).
        use_adaptive: Whether to compute learned adaptive adjacency.
        use_fixed: Whether to blend with fixed geographic adjacency.
        fixed_adj: Optional pre-normalized fixed adjacency matrix (N, N).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        horizon: int = 3,
        n_nodes: int = 25,
        emb_dim: int = 10,
        use_adaptive: bool = True,
        use_fixed: bool = True,
        fixed_adj: torch.Tensor | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.horizon = horizon
        self.n_nodes = n_nodes
        self.emb_dim = emb_dim
        self.use_adaptive = use_adaptive
        self.use_fixed = use_fixed
        self.dropout = dropout

        # 1. Learned node embeddings for adaptive adjacency
        if self.use_adaptive:
            self.nodevec1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
            self.nodevec2 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        else:
            self.register_parameter("nodevec1", None)
            self.register_parameter("nodevec2", None)

        # 2. Fixed adjacency matrix buffer
        if fixed_adj is not None:
            # Ensure row normalized
            row_sums = fixed_adj.sum(dim=-1, keepdim=True)
            row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
            norm_adj = fixed_adj / row_sums
            self.register_buffer("fixed_adj", norm_adj.float())
        else:
            self.register_buffer("fixed_adj", None)

        # 3. Learnable gating parameter g, initialized to 0.0 -> sigmoid(0) = 0.5
        if self.use_adaptive and self.use_fixed and fixed_adj is not None:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("gate", None)

        # 4. Dense graph convolutions
        self.conv1 = DenseGraphConv(in_dim, hidden)
        self.conv2 = DenseGraphConv(hidden, hidden)

        # 5. Multi-step forecasting head
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )

    def get_adaptive_adj(self) -> torch.Tensor:
        """Compute the normalized self-adaptive adjacency matrix."""
        if not self.use_adaptive:
            raise ValueError("Adaptive graph component is disabled.")
        adp = F.relu(torch.matmul(self.nodevec1, self.nodevec2.t()))
        return F.softmax(adp, dim=-1)

    def get_effective_adj(self) -> torch.Tensor:
        """Compute the blended effective adjacency matrix used in convolutions."""
        if self.use_adaptive and self.use_fixed and self.fixed_adj is not None and self.gate is not None:
            adp = self.get_adaptive_adj()
            alpha = torch.sigmoid(self.gate)
            return alpha * self.fixed_adj + (1.0 - alpha) * adp
        elif self.use_adaptive:
            return self.get_adaptive_adj()
        elif self.fixed_adj is not None:
            return self.fixed_adj
        else:
            raise ValueError("Neither adaptive nor fixed adjacency is available.")

    def get_gate_weight(self) -> float:
        """Return the scalar weight sigma(gate) assigned to the fixed geographic graph."""
        if self.gate is not None:
            return float(torch.sigmoid(self.gate).item())
        return 1.0 if self.use_fixed else 0.0

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features tensor of shape (N, in_dim).
            edge_index: Ignored if fixed_adj was provided at initialization.

        Returns:
            out: Predicted values of shape (N, horizon).
        """
        adj = self.get_effective_adj()
        h = F.relu(self.conv1(x, adj))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, adj))
        out = self.head(h)
        return out


def create_baseline_model(
    in_dim: int,
    hidden: int = 64,
    horizon: int = 3,
    kind: str = "GCN",
    heads: int = 8,
    dropout: float = 0.1,
) -> GNNBaseline:
    """Convenience factory to instantiate a GNNBaseline model."""
    return GNNBaseline(
        in_dim=in_dim,
        hidden=hidden,
        horizon=horizon,
        kind=kind,
        heads=heads,
        dropout=dropout,
    )


def create_adaptive_model(
    in_dim: int,
    hidden: int = 64,
    horizon: int = 3,
    n_nodes: int = 25,
    emb_dim: int = 10,
    use_adaptive: bool = True,
    use_fixed: bool = True,
    fixed_adj: torch.Tensor | None = None,
    dropout: float = 0.1,
) -> AdaptiveGCN:
    """Convenience factory to instantiate an AdaptiveGCN model."""
    return AdaptiveGCN(
        in_dim=in_dim,
        hidden=hidden,
        horizon=horizon,
        n_nodes=n_nodes,
        emb_dim=emb_dim,
        use_adaptive=use_adaptive,
        use_fixed=use_fixed,
        fixed_adj=fixed_adj,
        dropout=dropout,
    )
