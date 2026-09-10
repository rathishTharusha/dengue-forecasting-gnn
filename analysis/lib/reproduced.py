"""The five reproduced architectures, under this project's frozen protocol.

Why this module exists
----------------------
The Phase-1 and Phase-2 baselines were hand-rolled. The five architectures in
Weng et al. are, by contrast, *verified*: ``reproduction/`` runs their released
code and recovers their published numbers to within 7.3% across twenty
comparisons, two of them to four decimal places. That makes them the credible
starting point for a baseline, and this module is that baseline.

What is taken from them, and what is not
----------------------------------------
**Taken:** the architectures, from ``torch_geometric_temporal`` at the version
their ``requirements.txt`` pins, wired exactly as ``Models/gnn_models.py`` wires
them.

**Not taken:** their evaluation. ``reproduction/REPRODUCIBILITY_MATRIX.md``
establishes that their reported "Cross Validated" column is measured on the
``full`` loader -- training data included -- and that RMSE is averaged per window
rather than pooled, which reports a figure ~39% below the pooled value on this
target. Their training loop also has no early stopping and no validation-based
model selection: the final epoch's weights are scored.

So the architectures run here under ``docs/ROADMAP.md``'s frozen protocol:
rolling-origin over 3 origins x 3 seeds, window 3 -> horizon 3, normalisation
from training weeks only, early stopping on a genuine validation split, and
**pooled** RMSE on genuinely held-out windows. Numbers from this module are
therefore not comparable to Table I, and are comparable to every other row of
this project's ablation table.

One uniform head seam
---------------------
Every architecture here ends at a node-wise feature tensor and hands it to
:func:`improved.make_head`. That seam is what makes ``per_horizon_heads`` and
``probabilistic`` testable at all: without it they could be wired into only some
of the five, and an ablation row that silently covers three architectures out of
five is worse than no row.

With no increments enabled the head is a single ``nn.Linear``, which is the
reference wiring for A3TGCN, DCRNN and AAGCN. Two need a note:

* **STGAT** -- its LSTM collapses the node axis, so the head emits all 25 nodes
  from one graph-level vector. Still a single layer, exactly as the reference's
  ``nn.Linear(hidden, n_nodes * horizon)``.
* **ASTGCN** -- the reference's final convolution *is* the prediction, leaving no
  representation to attach a head to. It is given ``num_for_predict=feat`` and a
  head on top: one extra projection relative to the reference, present in
  **every** arm including ``base``. Comparisons between arms therefore stay
  controlled; comparisons of ASTGCN against ``reproduction/`` do not.

Requires ``torch_geometric_temporal``; see ``reproduction/README.md`` for the
pinned stack and the three forced deviations it needs.
"""

from __future__ import annotations

import improved as imp
import torch
from torch import nn

__all__ = ["ARCHITECTURES", "build", "input_adapter"]

#: The five architectures with released code, in Table I order.
ARCHITECTURES = ("STGAT", "A3TGCN", "ASTGCN", "DCRNN", "AAGCN")


class _STGAT(nn.Module):
    """GATConv over the district graph, then two LSTM layers.

    Transcribed from ``Models/gnn_models.py``. Two deliberate departures, both
    noted so a reader can tell reproduction from repair:

    * The reference calls ``F.dropout(x, self.dropout)`` without ``training=``,
      which defaults to ``True`` -- so dropout stays active at inference even
      after ``model.eval()``. That is a bug; it is fixed here, because this is a
      baseline rather than a reproduction.
    * Batching offsets the edge index the way PyG's ``Batch`` collation does.
      Passing a single-graph edge index for a batched input silently leaves every
      graph after the first with no edges.
    """

    def __init__(self, n_nodes: int, window: int, horizon: int, inc: imp.Increments,
                 heads: int = 8, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv

        self.n_nodes, self.horizon, self.dropout = n_nodes, horizon, dropout
        self.gat = GATConv(window, window, heads=heads, dropout=dropout, concat=False)
        self.lstm1 = nn.LSTM(n_nodes, hidden, num_layers=1)
        self.lstm2 = nn.LSTM(hidden, hidden, num_layers=1)
        # The LSTM has already collapsed the node axis, so the head emits every
        # node from one graph-level vector -- one layer, as in the reference.
        self.head = imp.make_head(hidden, horizon, inc, nodes=n_nodes)
        for lstm in (self.lstm1, self.lstm2):
            for name, param in lstm.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0.0)
                elif "weight" in name:
                    nn.init.xavier_uniform_(param)
        for layer in self.head.modules():      # their init, whatever shape the head is
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    @staticmethod
    def _batched_edges(edge_index: torch.Tensor, batch: int, nodes: int) -> torch.Tensor:
        if batch == 1:
            return edge_index
        offsets = torch.arange(batch, device=edge_index.device) * nodes
        return (edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)).permute(1, 0, 2).reshape(2, -1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch, nodes, window = x.shape
        h = self.gat(x.reshape(batch * nodes, window),
                     self._batched_edges(edge_index, batch, nodes))
        h = torch.nn.functional.dropout(h, self.dropout, training=self.training)
        h = h.reshape(batch, nodes, window).movedim(2, 0)
        h, _ = self.lstm1(h)
        h, _ = self.lstm2(h)
        return self.head(h[-1])


class _A3TGCN(nn.Module):
    """Attention temporal GCN, wired as ``TemporalGCN`` in their repo."""

    def __init__(self, n_nodes: int, window: int, horizon: int, inc: imp.Increments,
                 hidden: int = 32) -> None:
        super().__init__()
        from torch_geometric_temporal import A3TGCN

        self.core = A3TGCN(in_channels=1, out_channels=hidden, periods=window)
        self.head = imp.make_head(hidden, horizon, inc)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # A3TGCN wants (nodes, in_channels, periods) and carries no batch
        # dimension, so the batch is folded into the node dimension against a
        # block-diagonal graph. The graphs are disjoint, so this is equivalent to
        # looping -- and looping made this the second-slowest arm in the sweep
        # (~90 s per fold against ~6 s for STGAT).
        batch, nodes, window = x.shape
        big_edges = _STGAT._batched_edges(edge_index, batch, nodes)
        h = self.core(x.reshape(batch * nodes, 1, window), big_edges)
        return self.head(torch.relu(h).reshape(batch, nodes, -1))


class _ASTGCN(nn.Module):
    """Attention-based spatio-temporal GCN, 2 blocks / 64 filters as they set it."""

    def __init__(self, n_nodes: int, window: int, horizon: int, inc: imp.Increments,
                 feat: int = 16) -> None:
        super().__init__()
        from torch_geometric_temporal import ASTGCN

        # The reference sets num_for_predict=horizon, making the final convolution
        # the prediction itself and leaving nothing to attach a head to. Widening
        # it to `feat` and projecting gives the same seam as the other four; the
        # projection is present in every arm, `base` included.
        self.core = ASTGCN(nb_block=2, in_channels=1, K=3,
                           nb_chev_filter=64, nb_time_filter=64, time_strides=1,
                           num_for_predict=feat, len_input=window,
                           num_of_vertices=n_nodes)
        self.head = imp.make_head(feat, horizon, inc)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # (batch, nodes, window) -> (batch, nodes, in_channels=1, window)
        return self.head(self.core(x.unsqueeze(2), edge_index))


class _DCRNN(nn.Module):
    """Diffusion-convolution recurrent network, K=32 as they set it."""

    def __init__(self, n_nodes: int, window: int, horizon: int, inc: imp.Increments,
                 hidden: int = 64) -> None:
        super().__init__()
        from torch_geometric_temporal.nn.recurrent import DCRNN

        self.core = DCRNN(window, hidden, K=32)
        self.head = imp.make_head(hidden, horizon, inc)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # DCRNN carries no batch dimension, so the batch is folded into the node
        # dimension against a block-diagonal graph -- the graphs are disjoint, so
        # this is equivalent to looping and ~400x faster. Looping made DCRNN the
        # bottleneck of the whole sweep (6 s per batch of 32 against 0.01-0.37 s
        # for the others).
        batch, nodes, window = x.shape
        big_edges = _STGAT._batched_edges(edge_index, batch, nodes)
        h = torch.relu(self.core(x.reshape(batch * nodes, window), big_edges))
        return self.head(h.reshape(batch, nodes, -1))


class _AAGCN(nn.Module):
    """Two-stream adaptive GCN.

    ``adaptive`` is exposed here, unlike in their repo where it is hard-wired
    False. It cannot be enabled at their ``out_channels=1``: PGT computes
    ``inter_c = out_channels // 4``, which is 0, and the adaptive branch builds a
    zero-width convolution. Raising ``out_channels`` and projecting back is what
    makes the switch reachable -- and the projection is present in **both** arms
    so that enabling adaptivity is the only difference between them.
    """

    def __init__(self, n_nodes: int, window: int, horizon: int, inc: imp.Increments,
                 edge_index: torch.Tensor, channels: int = 8,
                 adaptive: bool = False, attention: bool = True) -> None:
        super().__init__()
        from torch_geometric_temporal import AAGCN

        # AAGCN builds its adjacency inside __init__, so the graph has to arrive
        # here rather than being attached afterwards.
        self.core = AAGCN(in_channels=1, out_channels=channels, edge_index=edge_index,
                          num_nodes=n_nodes, stride=1, residual=True,
                          adaptive=adaptive, attention=attention)
        self.head = imp.make_head(channels * window, horizon, inc)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # (batch, nodes, window) -> (batch, in_channels=1, window, nodes)
        out = self.core(x.permute(0, 2, 1).unsqueeze(1))
        batch, channels, window, nodes = out.shape
        return self.head(out.permute(0, 3, 1, 2).reshape(batch, nodes, channels * window))


def build(name: str, n_nodes: int, window: int, horizon: int,
          edge_index: torch.Tensor | None = None,
          inc: imp.Increments | None = None, **kwargs) -> nn.Module:
    """Construct one architecture by name.

    Args:
        name: One of :data:`ARCHITECTURES`.
        n_nodes: Districts.
        window: Input weeks.
        horizon: Output weeks.
        edge_index: Required by AAGCN, which bakes the graph into its constructor.
        inc: Which head increments to enable; ``None`` means the plain baseline.
        **kwargs: Passed through (e.g. ``adaptive=True`` for AAGCN).

    Returns:
        A module mapping ``(batch, nodes, window)`` to ``(batch, nodes, horizon)``,
        or to ``(batch, nodes, horizon, 2)`` when ``inc.probabilistic``.

    Raises:
        ValueError: On an unknown name.
    """
    inc = inc or imp.Increments()
    if name == "STGAT":
        return _STGAT(n_nodes, window, horizon, inc, **kwargs)
    if name == "A3TGCN":
        return _A3TGCN(n_nodes, window, horizon, inc, **kwargs)
    if name == "ASTGCN":
        return _ASTGCN(n_nodes, window, horizon, inc, **kwargs)
    if name == "DCRNN":
        return _DCRNN(n_nodes, window, horizon, inc, **kwargs)
    if name == "AAGCN":
        if edge_index is None:
            raise ValueError("AAGCN needs edge_index at construction time")
        return _AAGCN(n_nodes, window, horizon, inc, edge_index, **kwargs)
    raise ValueError(f"unknown architecture {name!r}; expected one of {ARCHITECTURES}")


def input_adapter(x: torch.Tensor) -> torch.Tensor:
    """Common input contract: ``(batch, nodes, window)`` float32."""
    return x.to(torch.float32)
