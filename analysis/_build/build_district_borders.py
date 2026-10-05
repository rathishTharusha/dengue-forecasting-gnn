"""List the pairs of Sri Lankan districts that share a border, from the GADM 4.1 level-1 polygons.

Two districts count as neighbours when their polygons intersect after a 0.0005 degree (about
50 m) buffer, which absorbs the digitising gaps along shared boundaries. The result is committed
as ``data/external/district_borders_gadm41.json`` so a test can check the model's adjacency list
without the git-ignored GADM file.

Source: GADM version 4.1, https://gadm.org/ (``gadm41_LKA_1.json``, the file the original
benchmark pipeline used; data/raw/disease_modeling_MLOS2/Data/Countries).

    python analysis/_build/build_district_borders.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

from shapely.geometry import shape

REPO = Path(__file__).resolve().parent.parent.parent
GADM = REPO / "data" / "raw" / "disease_modeling_MLOS2" / "Data" / "Countries" / "gadm41_LKA_1.json"
OUT = REPO / "data" / "external" / "district_borders_gadm41.json"
BUFFER_DEG = 0.0005


def main() -> None:
    raw = GADM.read_bytes()
    gj = json.loads(raw.decode("utf-8"))
    polys = {f["properties"]["NAME_1"].replace(" ", "").replace("-", ""): shape(f["geometry"])
             for f in gj["features"]}
    pairs = sorted([a, b] for a, b in itertools.combinations(sorted(polys), 2)
                   if polys[a].buffer(BUFFER_DEG).intersects(polys[b]))
    OUT.write_text(json.dumps({
        "source": "GADM 4.1 level 1, gadm41_LKA_1.json",
        "gadm_sha256": hashlib.sha256(raw).hexdigest(),
        "buffer_degrees": BUFFER_DEG,
        "districts": sorted(polys),
        "shared_borders": pairs,
    }, indent=1), encoding="utf-8")
    print(f"{len(polys)} districts, {len(pairs)} shared borders -> {OUT.name}")


if __name__ == "__main__":
    main()
