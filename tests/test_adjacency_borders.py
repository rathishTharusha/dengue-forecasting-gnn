"""The model's district graph must be exactly the set of shared borders in the GADM 4.1 polygons.

Two edges of the original benchmark file (Kandy to Ampara and Kegalle to Kalutara) were
one-directional and not real borders; both were removed. This fails if the adjacency list
is not symmetric, lists a pair that does not share a border, or omits one that does.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _edges(adj: dict) -> set[frozenset]:
    return {frozenset((a, b)) for a, nbrs in adj.items() for b in nbrs if a != b}


def test_adjacency_equals_gadm_shared_borders():
    adj = json.loads(
        (REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json").read_text(encoding="utf-8")
    )
    borders = json.loads(
        (REPO / "data" / "external" / "district_borders_gadm41.json").read_text(encoding="utf-8")
    )
    truth = {frozenset(p) for p in borders["shared_borders"]}
    mine = _edges(adj)
    assert sorted(adj) == borders["districts"]
    assert len(truth) == 57
    assert mine - truth == set(), f"listed but not a border: {sorted(map(sorted, mine - truth))}"
    assert truth - mine == set(), (
        f"border missing from the list: {sorted(map(sorted, truth - mine))}"
    )


def test_adjacency_is_symmetric():
    adj = json.loads(
        (REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json").read_text(encoding="utf-8")
    )
    one_way = [(a, b) for a, nbrs in adj.items() for b in nbrs if a not in adj.get(b, [])]
    assert one_way == [], one_way
