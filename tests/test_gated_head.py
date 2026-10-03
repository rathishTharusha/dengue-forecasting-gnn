"""The ``gated`` head is ``foi_res`` with the SEIR simulator taken out.

docs/RESCUE_PLAN.md attributes the gap between the two heads to the physics,
which is only valid if everything else is shared: the same anchor, the same
gate, the same initialisation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "seirgnn2"))


def _net(head):
    import models

    torch.manual_seed(0)
    return models.Net(7, 5, head=head, backbone="gcn")


def test_gate_matches_foi_res_initialisation():
    assert float(_net("gated").alpha) == float(_net("foi_res").alpha)


def test_closed_gate_returns_persistence_and_open_gate_the_direct_forecast():
    gated, direct = _net("gated"), _net("direct")
    gated.eval(), direct.eval()
    x, fixed = torch.rand(2, 5, 7), torch.softmax(torch.rand(5, 5), -1)
    p_z = torch.rand(2, 5, 3)
    raw, _, _ = direct(x, fixed, p_z)
    with torch.no_grad():
        gated.alpha.fill_(-50.0)
    torch.testing.assert_close(gated(x, fixed, p_z)[0], p_z)
    with torch.no_grad():
        gated.alpha.fill_(50.0)
    torch.testing.assert_close(gated(x, fixed, p_z)[0], raw)
