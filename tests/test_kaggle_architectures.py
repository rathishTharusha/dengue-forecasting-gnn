"""The Kaggle notebook's plain-PyTorch architectures equal the reference implementations.

``full_paper/kaggle/src/40_architectures.py`` transcribes the five published
architectures without any graph library, so the notebook runs on a stock Kaggle
image. This loads each reference module's weights (``analysis/lib/reproduced.py``
over ``torch_geometric_temporal`` and PyG) into the transcription and requires the
same output on the district graph, batched as the harness batches it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "seirgnn2"))


def _edge_index():
    adj = json.loads((REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json").read_text())
    names = sorted(adj)
    pos = {d: k for k, d in enumerate(names)}
    a = np.eye(len(names), dtype=np.float32)
    for d, nb in adj.items():
        for o in nb:
            if o in pos:
                a[pos[d], pos[o]] = 1.0
    src, dst = np.nonzero(a)
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


@pytest.fixture(scope="module")
def ns():
    import backbones

    backbones.install_shim()
    pytest.importorskip("torch_geometric_temporal")
    code = (REPO / "full_paper" / "kaggle" / "src" / "40_architectures.py").read_text(
        encoding="utf-8"
    )
    space: dict = {}
    exec(compile(code, "40_architectures.py", "exec"), space)
    return space


@pytest.mark.parametrize(
    "name,kw",
    [("STGAT", {}), ("A3TGCN", {}), ("ASTGCN", {}), ("AAGCN", {"adaptive": False, "channels": 8})],
)
def test_transcription_matches_reference(ns, name, kw):
    import reproduced

    torch.manual_seed(0)
    ei = _edge_index()
    ref = reproduced.build(name, 25, 3, 64, edge_index=ei, **kw)
    mine = ns["ENCODERS"][name](25, 3, 64, ei)
    missing, unexpected = mine.load_state_dict(ref.state_dict(), strict=False)
    assert not unexpected, unexpected
    # Only the precomputed graph operators, which the reference rebuilds each call.
    assert all(
        k.split(".")[-1] in {"incoming", "m", "lt", "p_out", "p_in", "A"} for k in missing
    ), missing
    x = torch.randn(7, 25, 3)
    modes = ("eval",) if name == "STGAT" else ("eval", "train")  # STGAT's dropout draws differently
    for mode in modes:
        ref.train(mode == "train")
        mine.train(mode == "train")
        with torch.no_grad():
            want = ref(x, ei)
            got = mine(x)
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5, msg=f"{name} ({mode})")


def test_dcrnn_diffusion_operators_match_exactly(ns):
    """Both K = 32 diffusion operators, on the batched graph the reference uses.

    The whole K = 32 network is compared through its operators rather than its
    output: its recursion builds gate pre-activations of ~1e6 from nearly
    cancelling terms, and the reference computes its normalisation in float32, so
    rounding alone moves the output by ~1e-3 on some inputs. The operators carry
    all of the graph logic, including the position-based pairing of the reverse
    weights, and agree to the last bit.
    """
    import reproduced
    from torch_geometric_temporal.nn.recurrent.dcrnn import DConv as RefDConv

    ei = _edge_index()
    p_out, p_in = ns["DConv"].transitions(ei, 25)
    for batch in (1, 7):
        big = reproduced._STGAT._batched_edges(ei, batch, 25)
        ref = RefDConv(1, 1, 2)
        with torch.no_grad():
            ref.bias.zero_()
            for which, op in ((0, p_out), (1, p_in)):
                ref.weight.zero_()
                ref.weight[which, 1, 0, 0] = 1.0
                eye = torch.eye(25 * batch)
                got = torch.stack(
                    [ref(eye[:, k : k + 1], big, None)[:, 0] for k in range(25 * batch)], 1
                )
                torch.testing.assert_close(got, torch.block_diag(*[op] * batch), rtol=0, atol=0)


def test_dcrnn_cell_matches_reference_at_small_k(ns):
    """The full GRU cell, at a K where the recursion stays well conditioned."""
    from torch_geometric_temporal.nn.recurrent import DCRNN as RefDCRNN

    torch.manual_seed(0)
    ei = _edge_index()
    ref = RefDCRNN(3, 64, K=3)
    mine = ns["DCRNNCell"](3, 64, 3)
    mine.load_state_dict(ref.state_dict())
    p_out, p_in = ns["DConv"].transitions(ei, 25)
    x = torch.randn(25, 3)
    with torch.no_grad():
        torch.testing.assert_close(mine(x, p_out, p_in), ref(x, ei), rtol=1e-5, atol=1e-6)


def test_lstm_encoder_matches_seirgnn2(ns):
    import backbones

    torch.manual_seed(0)
    ref = backbones._LSTM(3, 64)
    mine = ns["LSTMEncoder"](25, 3, 64)
    mine.load_state_dict(ref.state_dict())
    x = torch.randn(4, 25, 3)
    with torch.no_grad():
        torch.testing.assert_close(mine(x), ref(x, None))
