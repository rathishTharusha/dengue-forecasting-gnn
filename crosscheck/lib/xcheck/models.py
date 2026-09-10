"""Model implementations for the reproductions that need one.

Two families live here.

**R1 -- Weng et al. (2024).** :class:`STGAT` is transcribed from
``reference_repo/Models/gnn_models.py``. It is the one architecture of their
five that needs no ``torch-geometric-temporal``: the reference builds it from
plain ``torch_geometric.nn.GATConv`` plus two LSTM layers. The other four
(ASTGCN, A3TGCN, DCRNN, AAGCN) come from ``torch-geometric-temporal`` itself; R1
imports them when that package is present and reports them as unavailable when
it is not, rather than substituting a lookalike and calling it a reproduction.

**R2 -- GulMohamed et al. (2026).** :class:`DynamicSTGNN` implements the paper's
Equations (2)-(14): hybrid geographic/gravity-mobility dynamic graph, graph
convolution spatial encoder, attention-augmented LSTM temporal encoder, feature
fusion, and per-horizon heads with an optional Gaussian uncertainty output. The
ablation switches (``dynamic_graph``, ``temporal_attention``, ``use_mobility``)
exist because the paper's Table 6 reports exactly those three ablations.

Nothing here imports ``src/dengue_gnn``.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["STGAT", "DynamicSTGNN", "gaussian_nll"]


# --------------------------------------------------------------------------
# R1 -- Weng et al.
# --------------------------------------------------------------------------


class STGAT(nn.Module):
    """Graph attention encoder followed by two LSTM layers.

    Transcribed from the reference implementation, with two deliberate
    departures, both noted so a reader can tell reproduction from repair:

    * The reference calls ``torch.tensor(x)`` on an existing tensor inside
      ``forward``; that is a no-op copy that newer torch warns about, so the
      cast is dropped.
    * The reference reshapes using ``data.num_graphs`` from a PyG ``Batch``.
      This version takes an explicit ``(batch, nodes, features)`` tensor, which
      is the same arithmetic without the loader dependency.

    Args:
        in_channels: Input features per node (``window`` for disease-only runs).
        n_pred: Forecast horizon.
        n_nodes: Number of graph nodes.
        heads: Attention heads in the ``GATConv``.
        hidden: Hidden width of both LSTM layers.
        dropout: Dropout applied after the attention layer.
    """

    def __init__(
        self,
        in_channels: int,
        n_pred: int,
        n_nodes: int,
        heads: int = 8,
        hidden: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv

        self.n_pred = n_pred
        self.n_nodes = n_nodes
        self.dropout = dropout

        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=in_channels,
            heads=heads,
            dropout=dropout,
            concat=False,
        )
        self.lstm1 = nn.LSTM(input_size=n_nodes, hidden_size=hidden, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=hidden, hidden_size=hidden, num_layers=1)
        self.linear = nn.Linear(hidden, n_nodes * n_pred)

        # Reference initialization: xavier on weights, zeros on biases.
        for lstm in (self.lstm1, self.lstm2):
            for name, param in lstm.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0.0)
                elif "weight" in name:
                    nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.linear.weight)

    @staticmethod
    def _batched_edge_index(edge_index: torch.Tensor, batch: int, nodes: int) -> torch.Tensor:
        """Block-diagonal edge index for ``batch`` copies of one graph.

        The reference feeds a PyG ``DataLoader``, whose ``Batch`` collation
        offsets each graph's node ids by ``g * nodes`` so the graphs stay
        disconnected. Passing a single-graph ``edge_index`` for a batched input
        instead would leave every graph after the first with no edges at all --
        silently, since the shapes still line up.
        """
        if batch == 1:
            return edge_index
        offsets = torch.arange(batch, device=edge_index.device) * nodes
        return (edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)).permute(1, 0, 2).reshape(2, -1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, nodes, features)`` to ``(batch, nodes, n_pred)``.

        ``edge_index`` describes a **single** graph; batching is handled here.
        """
        batch, nodes, feats = x.shape
        h = self.gat(
            x.reshape(batch * nodes, feats),
            self._batched_edge_index(edge_index, batch, nodes),
        )
        h = torch.nn.functional.dropout(h, self.dropout, training=self.training)

        # (B*N, F) -> (B, N, F) -> (F, B, N): features become the sequence axis,
        # exactly as in the reference's movedim(2, 0).
        h = h.reshape(batch, nodes, feats).movedim(2, 0)
        h, _ = self.lstm1(h)
        h, _ = self.lstm2(h)

        out = self.linear(h[-1])
        return out.reshape(batch, self.n_nodes, self.n_pred)


# --------------------------------------------------------------------------
# R2 -- GulMohamed et al.
#
# The dynamic graph itself (Eq. 2-4) lives in xcheck.graph, which needs no
# torch. Only the network is here.
# --------------------------------------------------------------------------


class _TemporalAttention(nn.Module):
    """Additive attention over LSTM outputs, the paper's Eq. (6)-(8).

    Scores each of the ``L`` hidden states and returns their convex combination.
    Disabling it (the Table 6 ablation) means taking the last hidden state
    instead, which is the plain-LSTM behaviour.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """``(batch, L, hidden)`` -> ``(batch, hidden)``."""
        weights = torch.softmax(self.score(h), dim=1)
        return (weights * h).sum(dim=1)


class DynamicSTGNN(nn.Module):
    """The paper's Dynamic Spatio-Temporal GNN, Equations (5)-(14).

    Pipeline: graph convolution per time step (Eq. 5) -> attention-augmented
    LSTM over the window (Eq. 6-8) -> fusion with environmental and mobility
    features (Eq. 9) -> one head per horizon (Eq. 11), optionally emitting a
    mean and a log-variance (Eq. 12).

    Args:
        n_features: Node features per time step.
        n_env: Width of the auxiliary environmental feature vector.
        n_mob: Width of the auxiliary mobility feature vector.
        horizons: Number of separate forecast heads.
        hidden: Width of the spatial and temporal representations.
        dropout: Dropout applied to the fused embedding.
        temporal_attention: ``False`` runs the "no attention" ablation.
        use_mobility: ``False`` runs the "no mobility features" ablation.
        probabilistic: Emit ``(mu, log_var)`` instead of a point forecast.
    """

    def __init__(
        self,
        n_features: int,
        n_env: int = 0,
        n_mob: int = 0,
        horizons: int = 1,
        hidden: int = 64,
        dropout: float = 0.1,
        temporal_attention: bool = True,
        use_mobility: bool = True,
        probabilistic: bool = False,
    ) -> None:
        super().__init__()
        self.use_mobility = use_mobility
        self.probabilistic = probabilistic
        self.horizons = horizons

        self.spatial = nn.Linear(n_features, hidden)  # W_s of Eq. (5)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.attention = _TemporalAttention(hidden) if temporal_attention else None

        fused_in = hidden + n_env + (n_mob if use_mobility else 0)
        self.fuse = nn.Sequential(nn.Linear(fused_in, hidden), nn.ReLU(), nn.Dropout(dropout))

        out_per_head = 2 if probabilistic else 1
        self.heads = nn.ModuleList(nn.Linear(hidden, out_per_head) for _ in range(horizons))

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        env: torch.Tensor | None = None,
        mob: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast from a window of node features.

        Args:
            x: ``(batch, window, nodes, n_features)`` node features.
            adjacency: ``(nodes, nodes)`` normalized adjacency, or
                ``(batch, window, nodes, nodes)`` for a genuinely time-varying
                graph.
            env: Optional ``(batch, nodes, n_env)`` environmental features.
            mob: Optional ``(batch, nodes, n_mob)`` mobility features.

        Returns:
            ``(batch, nodes, horizons)`` point forecasts, or
            ``(batch, nodes, horizons, 2)`` of ``(mu, log_var)`` when
            ``probabilistic``.
        """
        batch, window, nodes, _ = x.shape

        # Eq. (5), applied per time step: H_t = ReLU(A X_t W_s)
        proj = self.spatial(x)
        if adjacency.dim() == 2:
            spatial = torch.relu(torch.einsum("ij,btjf->btif", adjacency, proj))
        elif adjacency.dim() == 4:
            spatial = torch.relu(torch.einsum("btij,btjf->btif", adjacency, proj))
        else:
            raise ValueError(f"adjacency must be 2-D or 4-D, got {adjacency.dim()}-D")

        # Eq. (6)-(8): each node's window is its own sequence.
        seq = spatial.permute(0, 2, 1, 3).reshape(batch * nodes, window, -1)
        out, (h_n, _) = self.lstm(seq)
        temporal = self.attention(out) if self.attention is not None else h_n[-1]
        temporal = temporal.reshape(batch, nodes, -1)

        # Eq. (9): fuse with auxiliary features.
        parts = [temporal]
        if env is not None:
            parts.append(env)
        if self.use_mobility and mob is not None:
            parts.append(mob)
        fused = self.fuse(torch.cat(parts, dim=-1))

        # Eq. (11)/(12): one head per horizon.
        stacked = torch.stack([head(fused) for head in self.heads], dim=2)
        return stacked if self.probabilistic else stacked.squeeze(-1)


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of a Gaussian, the training loss behind Eq. (12).

    ``0.5 * (log_var + (y - mu)^2 / exp(log_var))``, averaged, with the constant
    term dropped. Optimizing this is what makes the Eq. (14) intervals
    calibrated rather than decorative.
    """
    return torch.mean(0.5 * (log_var + (target - mu) ** 2 / torch.exp(log_var)))
