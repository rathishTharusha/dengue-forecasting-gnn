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
    "GatedTCN",
    "GraphAttention",
    "RecurrentTemporal",
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


class GatedTCN(nn.Module):
    r"""Gated dilated causal convolution over the input window.

    This is the temporal operator Graph WaveNet~\cite{wu2019graphwavenet} places
    before every graph-convolution layer, and which the first version of this model
    omitted entirely: it flattened ``(window, features)`` into one vector and applied
    a linear layer, so the "spatio-temporal" model had no temporal component at all.
    That omission is the most likely reason a plain LSTM outperformed it.

    Two parallel convolutions form an LSTM-style gate,

        h = tanh(W_f * x) . sigmoid(W_g * x)

    where the filter branch proposes an update and the gate branch decides how much
    of it passes. Left-padding by ``(kernel - 1) * dilation`` keeps the convolution
    **causal**: the output at week *t* never sees week *t+1*.

    Args:
        n_feat: Input channels, i.e. features per week.
        hidden: Output channels.
        kernel: Convolution width in weeks.
        dilation: Spacing between taps. With ``W=3`` there is little room to
            dilate, so the default is 1; the parameter exists so the receptive
            field can grow if the window is widened.
    """

    def __init__(self, n_feat: int, hidden: int, kernel: int = 2, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.filt = nn.Conv1d(n_feat, hidden, kernel, dilation=dilation)
        self.gate = nn.Conv1d(n_feat, hidden, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``(batch, n_feat, window)``. Returns ``(batch, hidden)``.

        Only the final timestep is returned: it is the one whose receptive field
        covers the whole window, and the prediction head consumes a single vector
        per node.
        """
        x = F.pad(x, (self.pad, 0))
        out = torch.tanh(self.filt(x)) * torch.sigmoid(self.gate(x))
        return out[..., -1]


class GraphAttention(nn.Module):
    r"""Multi-head graph attention, after Velickovic et al.~\cite{velickovic2018gat}.

    Attention coefficients are computed from node features and **masked by the
    fixed adjacency**, as in the original: a node attends only to its graph
    neighbours (plus itself, since ``adj_fixed`` carries self-loops). The result
    is a data-dependent, per-forward-pass adjacency, in contrast to
    :class:`AdaptiveAdjacency`, whose learned graph is a free parameter shared
    across all inputs.

    That distinction is the point of running both: GAT re-weights a *given*
    topology per input, while the adaptive graph *invents* a topology that is
    fixed once trained. They are different hypotheses about what geography gets
    wrong.

    Args:
        in_dim: Input features per node.
        out_dim: Output features per node, per head.
        heads: Number of attention heads; outputs are averaged, not concatenated,
            so ``out_dim`` is the final width.
        dropout: Applied to the attention coefficients.
        negative_slope: LeakyReLU slope in the attention logits, 0.2 in the paper.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 8,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.dropout, self.negative_slope = dropout, negative_slope
        self.proj = nn.Linear(in_dim, heads * out_dim, bias=False)
        # a = [a_src ; a_dst] in the paper's notation, split for efficiency.
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        """Args: ``x`` is ``(batch, n_nodes, in_dim)``, ``adj_mask`` ``(N, N)``.

        Returns ``(batch, n_nodes, out_dim)``, averaged over heads.
        """
        b, n, _ = x.shape
        h = self.proj(x).view(b, n, self.heads, self.out_dim)

        # e_ij = LeakyReLU(a_src . h_i + a_dst . h_j)
        src = (h * self.att_src).sum(-1)  # (b, n, heads)
        dst = (h * self.att_dst).sum(-1)
        logits = F.leaky_relu(src.unsqueeze(2) + dst.unsqueeze(1), self.negative_slope)

        # Mask to graph neighbours before softmax, as GAT prescribes.
        mask = (adj_mask > 0).unsqueeze(0).unsqueeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(logits, dim=2)
        alpha = torch.nan_to_num(alpha)  # an isolated node has an all -inf row
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # alpha (b, i, j, heads) x h (b, j, heads, f) -> (b, i, heads, f)
        out = torch.einsum("bijk,bjkf->bikf", alpha, h)
        return out.mean(dim=2)  # average heads; concatenating would change width


class RecurrentTemporal(nn.Module):
    """GRU or LSTM over the input window, with optional attention over timesteps.

    The recurrent temporal encoder used by A3TGCN and STGAT. Where
    :class:`GatedTCN` is convolutional, this is recurrent; both give the model the
    temporal operator the original design lacked.

    ``attention=True`` reproduces A3TGCN's temporal attention: instead of taking
    the final hidden state, a learned score over all timesteps produces a weighted
    sum, letting the model emphasise whichever weeks matter for the horizon.

    Args:
        n_feat: Input channels per week.
        hidden: Hidden width.
        kind: ``"gru"`` or ``"lstm"``.
        attention: Attention-weighted pooling over timesteps rather than last state.
    """

    def __init__(self, n_feat: int, hidden: int, kind: str = "gru", attention: bool = False):
        super().__init__()
        if kind not in ("gru", "lstm"):
            raise ValueError(f"kind must be 'gru' or 'lstm', got {kind!r}")
        cell = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = cell(n_feat, hidden, batch_first=True)
        self.attention = attention
        self.score = nn.Linear(hidden, 1) if attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``(batch, n_feat, window)``. Returns ``(batch, hidden)``."""
        out, _ = self.rnn(x.transpose(1, 2))  # (batch, window, hidden)
        if not self.attention:
            return out[:, -1]
        w = F.softmax(self.score(out), dim=1)  # (batch, window, 1)
        return (w * out).sum(dim=1)


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
        temporal: str = "none",
        spatial: str = "gcn",
        n_feat: int | None = None,
        window: int | None = None,
        gat_heads: int = 8,
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

        valid_temporal = ("none", "gtcn", "gru", "lstm", "gru_attn")
        if temporal not in valid_temporal:
            raise ValueError(f"temporal must be one of {valid_temporal}, got {temporal!r}")
        if spatial not in ("gcn", "gat"):
            raise ValueError(f"spatial must be 'gcn' or 'gat', got {spatial!r}")
        self.temporal, self.spatial = temporal, spatial

        if temporal != "none":
            if n_feat is None or window is None:
                raise ValueError(f"temporal={temporal!r} requires n_feat and window")
            if n_feat * window != in_dim:
                raise ValueError(f"n_feat*window ({n_feat}*{window}) must equal in_dim ({in_dim})")
            self.n_feat, self.window = n_feat, window
            if temporal == "gtcn":
                self.tcn = GatedTCN(n_feat, hidden)
            else:
                self.tcn = RecurrentTemporal(
                    n_feat,
                    hidden,
                    kind="lstm" if temporal == "lstm" else "gru",
                    attention=(temporal == "gru_attn"),
                )
            spatial_in = hidden
        else:
            self.tcn = None
            spatial_in = in_dim

        if spatial == "gat":
            # GAT computes attention over the masked topology, so the first
            # linear map is folded into the attention layer.
            self.gat1 = GraphAttention(spatial_in, hidden, heads=gat_heads, dropout=dropout)
            self.gat2 = GraphAttention(hidden, hidden, heads=gat_heads, dropout=dropout)
            self.w1 = self.w2 = None
        else:
            self.gat1 = self.gat2 = None
            self.w1 = nn.Linear(spatial_in, hidden)
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

        if self.tcn is not None:
            # (B, N, n_feat*window) -> (B*N, n_feat, window). The reshape matches
            # _build_fold's feature-major layout, transpose((1, 2, 0)).
            b, n, _ = x.shape
            seq = x.reshape(b * n, self.n_feat, self.window)
            x = self.tcn(seq).reshape(b, n, -1)

        if self.spatial == "gat":
            # GAT attends over the topology; the blended adjacency supplies the
            # mask so the learned and attention-based graphs stay comparable.
            h = F.relu(self.gat1(x, adj))
            h = F.dropout(h, self.dropout, training=self.training)
            h = F.relu(self.gat2(h, adj))
        else:
            # einsum keeps the batch axis untouched while mixing across nodes:
            # (N,N) x (B,N,F) -> (B,N,F)
            h = F.relu(torch.einsum("ij,bjf->bif", adj, self.w1(x)))
            h = F.dropout(h, self.dropout, training=self.training)
            h = F.relu(torch.einsum("ij,bjf->bif", adj, self.w2(h)))
        return self.head(h)
