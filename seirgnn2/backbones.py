"""The five published architectures, made available to the seirgnn2 heads.

Why this module exists
----------------------
``models.GraphLayer`` is a single dense matmul -- ``relu(lin(A @ h))`` -- and its
``gat`` mode is a literal alias for ``gcn``, returning the same fixed adjacency.
That is a control, not a proposal. The proposal is Liu et al.'s SEIR-LSTM with
the LSTM replaced by a *spatio-temporal graph network*, so the force of infection
has to come out of A3TGCN, STGAT, ASTGCN, AAGCN or DCRNN for the comparison to
mean anything.

Those five are already transcribed and verified in ``analysis/lib/reproduced.py``
-- ``reproduction/`` recovers their published numbers to within 7.3% across
twenty comparisons. This module reuses that wiring rather than re-transcribing
it, so nothing here can drift from the verified reference.

Two adaptations, both deliberate
--------------------------------
**Representation, not prediction.** ``reproduced.build`` ends at
``imp.make_head(..., horizon)``. Passing ``horizon = hidden`` makes that final
layer emit a representation of width ``hidden`` instead of a forecast, which is
the seam the seirgnn2 heads attach to. No layer is added or removed.

**Exogenous channels bypass the backbone.** The five take ``(batch, nodes,
window)`` -- one channel, the case history. ``run_s5_seir_gnn.py`` fitted climate
and season into that contract with ``nn.Linear(in_dim, 1)``, collapsing every
covariate to a scalar before the graph saw it, which is why its input-level
factor moved results by under 1% RMSE. Here the backbone receives the case
window it was designed for and the extra channels are concatenated to its output
at the head seam. They reach the head undiminished, and the backbone runs at its
reference contract.

The ``torch_sparse`` shim
-------------------------
``torch_geometric_temporal`` imports ``SparseTensor`` from ``torch_sparse`` in
exactly one module, ``nn/recurrent/evolvegcno.py``, reached only through the
package ``__init__``. ``torch_sparse`` has no wheel for this torch and needs a
source build. EvolveGCN is not one of the five and is never constructed, so the
symbol is supplied from ``torch_geometric.typing`` instead. Nothing this module
builds touches it; :func:`check` proves that by running a forward pass through
all five.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))

#: The five architectures with released code, in the reproduction's order.
PUBLISHED = ("STGAT", "A3TGCN", "ASTGCN", "DCRNN", "AAGCN")

#: Liu et al.'s SEIR-LSTM encoder, run in this harness so that "did the graph
#: beat the LSTM" is a controlled comparison. ``analysis/lib/seir_lstm.py`` and
#: ``run_s4_seir_lstm.py`` train it in a *different* loop with a different
#: objective and different folds, so the paper's 23.79 / 62.61 is not comparable
#: to any SEIR-GNN number. Here both sit behind the same heads, the same loss,
#: the same folds and the same early stopping, and differ only by the encoder.
REAL = (*PUBLISHED, "LSTM")


def install_shim() -> None:
    """Satisfy ``evolvegcno``'s import so the other five can be reached."""
    if "torch_sparse" in sys.modules:
        return
    try:
        from torch_geometric.typing import SparseTensor
    except Exception:  # pragma: no cover - only if PyG drops the alias
        class SparseTensor:
            """Placeholder; EvolveGCN is never constructed here."""

    stub = types.ModuleType("torch_sparse")
    stub.SparseTensor = SparseTensor
    sys.modules["torch_sparse"] = stub


class _LSTM(nn.Module):
    """Per-district LSTM over the case window -- no message passing.

    The encoder of Liu et al.'s SEIR-LSTM, at the seam the other five use. Each
    district is rolled independently, which is the point: it is the control that
    says what the graph adds.
    """

    def __init__(self, window: int, hidden: int, layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch, nodes, window = x.shape
        h, _ = self.lstm(x.reshape(batch * nodes, window, 1))
        return h[:, -1].reshape(batch, nodes, -1)


class Real(nn.Module):
    """One published architecture as an encoder for the seirgnn2 heads.

    Maps ``(B, N, window + extra)`` to ``(B, N, hidden + extra)``: the case
    window goes through the architecture, the exogenous channels go around it.
    """

    def __init__(self, name: str, n_nodes: int, window: int, extra: int, hidden: int,
                 edge_index: torch.Tensor, **kwargs) -> None:
        super().__init__()
        install_shim()
        import reproduced as arch

        if name not in REAL:
            raise ValueError(f"unknown architecture {name!r}; expected one of {REAL}")
        self.name, self.window, self.extra = name, window, extra
        if name == "LSTM":
            self.core = _LSTM(window, hidden, **kwargs)
        else:
            if name == "AAGCN":
                kwargs = {"adaptive": False, "channels": 8, **kwargs}
            self.core = arch.build(name, n_nodes, window, hidden, edge_index=edge_index, **kwargs)
        self.out_dim = hidden + extra

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.core(x[..., : self.window].contiguous(), edge_index))
        if self.extra:
            h = torch.cat([h, x[..., self.window :]], dim=-1)
        return h


def check(n_nodes: int = 25, window: int = 3, hidden: int = 32, batch: int = 4) -> dict[str, str]:
    """Build and run every architecture once; report shape or the failure."""
    install_shim()
    ring = torch.tensor([[k for k in range(n_nodes)] + [(k + 1) % n_nodes for k in range(n_nodes)],
                         [(k + 1) % n_nodes for k in range(n_nodes)] + list(range(n_nodes))])
    x = torch.randn(batch, n_nodes, window + 2)
    out = {}
    for name in REAL:
        try:
            net = Real(name, n_nodes, window, extra=2, hidden=hidden, edge_index=ring)
            with torch.no_grad():
                h = net(x, ring)
            params = sum(p.numel() for p in net.parameters())
            out[name] = f"ok  {tuple(h.shape)}  ({params:,} params)"
        except Exception as exc:
            out[name] = f"FAIL  {type(exc).__name__}: {exc}"
    return out


if __name__ == "__main__":
    for name, status in check().items():
        print(f"{name:8s} {status}")
