"""Build a movement-weighted district graph from population and distance.

Why: the project's spatial structure is "two districts are connected if they
share a border", and that assumption has been tested twice and found to
contribute nothing measurable -- the adaptive-graph experiment (EXP-016) and the
frozen-encoder control (EXP-052). Dengue is carried between districts by people,
while *Aedes* mosquitoes travel only a few hundred metres, so a border is the
wrong proxy for the thing that matters.

A gravity model is the standard substitute in spatial epidemiology:

    flow(i -> j)  proportional to  (P_i^a * P_j^b) / d(i, j)^k

with `d` the great-circle distance between district interior points. Nothing new
has to be collected: the populations are already in `data/external/` and the
boundaries are the GADM level-1 polygons the project already uses for ERA5.

Outputs `data/external/district_gravity_matrix.csv` -- a 25 x 25 row-normalised
matrix, zero diagonal, districts in the same order as `sri_lanka_adj_list.json`.

Usage::

    python analysis/_build/build_gravity_graph.py              # k = 2
    python analysis/_build/build_gravity_graph.py --decay 1.0  # k = 1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
GADM = REPO / "data" / "raw" / "disease_modeling_MLOS2" / "Data" / "Countries" / "gadm41_LKA_1.json"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
POP = REPO / "data" / "external" / "district_population.csv"
OUT = REPO / "data" / "external" / "district_gravity_matrix.csv"
POINTS_OUT = REPO / "data" / "external" / "district_points.csv"


def rings(geom: dict) -> list[list[list[float]]]:
    """Every exterior ring in a Polygon or MultiPolygon."""
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    return [poly[0] for poly in geom["coordinates"]]


def ring_centroid(ring: list[list[float]]) -> tuple[float, float, float]:
    """Planar centroid and |area| of one ring, by the shoelace formula.

    Degrees are treated as a plane here. Over a single Sri Lankan district that
    is accurate to well under a kilometre, which is far below the precision a
    gravity model needs.
    """
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if a == 0:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys), 0.0
    a *= 0.5
    return cx / (6 * a), cy / (6 * a), abs(a)


def district_points() -> dict[str, tuple[float, float]]:
    """(lat, lon) per district: the centroid of its largest ring."""
    if not GADM.exists():
        raise SystemExit(
            f"missing {GADM.relative_to(REPO)} -- see docs/DATA_PROVENANCE.md for the source")
    gadm = json.loads(GADM.read_text(encoding="utf-8"))
    wanted = set(json.loads(ADJ.read_text(encoding="utf-8")))
    pts: dict[str, tuple[float, float]] = {}
    for ft in gadm["features"]:
        name = ft["properties"]["NAME_1"]
        if name not in wanted:
            continue
        best = max((ring_centroid(r) for r in rings(ft["geometry"])), key=lambda t: t[2])
        pts[name] = (round(best[1], 4), round(best[0], 4))     # lat, lon
    missing = wanted - set(pts)
    if missing:
        raise SystemExit(f"no polygon for {sorted(missing)}")
    return pts


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def populations(names: list[str]) -> np.ndarray:
    """The most recent population per district, in the given order."""
    df = pd.read_csv(POP)
    col = next((c for c in df.columns if c.lower() in ("district", "name")), df.columns[0])
    year_cols = [c for c in df.columns if str(c).strip().isdigit()]
    if year_cols:
        latest = max(year_cols, key=lambda c: int(c))
        series = df.set_index(col)[latest]
    else:
        value = next(c for c in df.columns
                     if "pop" in c.lower() and df[c].dtype.kind in "if")
        year = next((c for c in df.columns if "year" in c.lower()), None)
        if year is not None:
            df = df.sort_values(year).groupby(col).tail(1)
        series = df.set_index(col)[value]
    missing = [n for n in names if n not in series.index]
    if missing:
        raise SystemExit(f"no population for {missing}")
    return np.array([float(series[n]) for n in names])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decay", type=float, default=2.0, help="distance exponent k")
    ap.add_argument("--alpha", type=float, default=1.0, help="exponent on the origin population")
    ap.add_argument("--beta", type=float, default=1.0, help="exponent on the destination population")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    names = sorted(json.loads(ADJ.read_text(encoding="utf-8")))
    pts = district_points()
    pop = populations(names)
    n = len(names)

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = haversine_km(pts[names[i]], pts[names[j]])

    flow = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            flow[i, j] = (pop[i] ** args.alpha * pop[j] ** args.beta) / dist[i, j] ** args.decay
    row = flow.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    hat = flow / row

    pd.DataFrame(hat, index=names, columns=names).to_csv(args.out)
    pd.DataFrame({"district": names,
                  "lat": [pts[x][0] for x in names],
                  "lon": [pts[x][1] for x in names],
                  "population": pop.astype(int)}).to_csv(POINTS_OUT, index=False)

    adj = json.loads(ADJ.read_text(encoding="utf-8"))
    geo = np.zeros((n, n))
    for i, a in enumerate(names):
        for b in adj.get(a, []):
            if b in names:
                geo[i, names.index(b)] = 1.0
    geo_hat = geo / np.maximum(geo.sum(axis=1, keepdims=True), 1.0)

    print(f"wrote {Path(args.out).relative_to(REPO)}  ({n} x {n}, k = {args.decay})")
    print(f"wrote {POINTS_OUT.relative_to(REPO)}")
    print(f"\nmean distance between districts: {dist[dist > 0].mean():.0f} km")
    print(f"agreement with the border graph: corr = "
          f"{np.corrcoef(hat.ravel(), geo_hat.ravel())[0, 1]:.3f}")
    print(f"share of gravity weight on bordering districts: "
          f"{100 * hat[geo > 0].sum() / hat.sum():.1f}%")
    print("\nstrongest links")
    pairs = [(hat[i, j], names[i], names[j]) for i in range(n) for j in range(n) if i != j]
    for w, a, b in sorted(pairs, reverse=True)[:8]:
        border = "border" if geo[names.index(a), names.index(b)] else "      "
        print(f"  {a:14s} -> {b:14s} {w:.3f}  {border}  {dist[names.index(a), names.index(b)]:5.0f} km")
    return 0


if __name__ == "__main__":
    sys.exit(main())
