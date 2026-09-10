"""One table: baselines, adaptive-graph arms, and physics-informed arms.

Everything is on the same scale and the same windows -- pooled RMSE, artifact-free,
so the week-395 reporting backlog is excluded from every row including the floor.
An all-windows number is 15 RMSE higher and is not comparable to anything here.

Two columns matter and they answer different questions:

``vs floor``
    Absolute standing. Does this arm beat naive persistence? This is the question a
    reader asks first and the one almost nothing on this dataset passes.
``paired dRMSE``
    Does the *change* help, holding the architecture, origin and seed fixed? A
    change can be reliably positive while its architecture still loses to
    persistence -- which is exactly what the physics constraint turns out to be.

Sample sizes differ by design and are printed, because at n=9 two of these arms
were indistinguishable and at n=60 one holds and the other is a coin flip.

Run::

    python analysis/_build/compare_all.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "analysis" / "results"

#: Persistence, artifact-free, per origin.
FLOOR = {0.55: 26.882223151529477, 0.7: 38.88532915547214, 0.85: 22.79547681977414}
FLOOR_MEAN = float(np.mean(list(FLOOR.values())))


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def summarise(vals, floors):
    vals, floors = np.asarray(vals), np.asarray(floors)
    _, p = stats.ttest_rel(vals, floors)
    return float(vals.mean()), float((vals - floors).mean()), float(p), len(vals)


def paired(rows, arm, base_arm, key):
    def k(r):
        return (r["origin"], r["seed"])

    base = {k(r): r["RMSE_clean"] for r in rows if r[key] == base_arm}
    pairs = [(r["RMSE_clean"], base[k(r)]) for r in rows if r[key] == arm and k(r) in base]
    if len(pairs) < 3:
        return None
    d = np.array([a - b for a, b in pairs])
    if np.allclose(d, 0.0):
        return 0.0, len(d), 0, float("nan")
    _, p = stats.ttest_rel([a for a, _ in pairs], [b for _, b in pairs])
    return float(d.mean()), len(d), int((d < 0).sum()), float(p)


def line(label, mean, delta, p_floor, n, pair=None, note=""):
    pf = "  n/a" if np.isnan(p_floor) else f"{p_floor:7.3f}"
    if pair is None:
        pd = " " * 26
    else:
        d, np_, better, pp = pair
        ps = "  n/a" if np.isnan(pp) else f"{pp:7.4f}"
        pd = f"{d:+9.3f}{better:5d}/{np_:<4d}{ps}"
    print(f"{label:34s}{mean:9.2f}{delta:+9.3f}{pf}{n:5d}   {pd}{note}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Pooled RMSE, ARTIFACT-FREE. Persistence floor = "
          f"{FLOOR_MEAN:.2f} (26.88 / 38.89 / 22.80 by origin).\n")
    print(f"{'':34s}{'RMSE':>9s}{'vs floor':>9s}{'p':>7s}{'n':>5s}   "
          f"{'paired d':>9s}{'better':>10s}{'p':>7s}")
    print("=" * 108)

    # ---------------------------------------------------------- baselines
    print("\nBASELINE  --  the five reproduced architectures, no modification")
    sweep = load("improved_sweep.json")
    base_rows = [r for r in sweep if r["arch"] != "persistence" and r["increment"] == "base"]
    for arch in ("A3TGCN", "STGAT", "AAGCN", "ASTGCN", "DCRNN"):
        sub = [r for r in base_rows if r["arch"] == arch]
        if sub:
            line(f"  {arch}", *summarise([r["RMSE_clean"] for r in sub],
                                         [FLOOR[r["origin"]] for r in sub]))

    # ---------------------------------------------- adaptive graph
    print("\nADAPTIVE GRAPH  --  learned adjacency instead of, or blended with, geography")
    repro = load("reproduced_baseline.json")
    ad = [r for r in repro if r["arm"] in ("AAGCN", "AAGCN+adaptive")]
    for arm in ("AAGCN", "AAGCN+adaptive"):
        sub = [r for r in ad if r["arm"] == arm]
        pair = paired(ad, arm, "AAGCN", "arm") if arm != "AAGCN" else None
        line(f"  {arm}", *summarise([r["RMSE_clean"] for r in sub],
                                    [FLOOR[r["origin"]] for r in sub]), pair=pair)
    # EXP-022, artifact-free, self-path restored. Not in a results JSON because
    # it was a controlled encoder run rather than part of a sweep.
    print("  STGNN encoder, 4 graph modes (EXP-022, n=9, 60 epochs, self-path on)")
    print("    none 28.90    fixed 28.97    adaptive 29.16    hybrid 28.55")
    print("    -- learned-only is worst; blended is best; spread 0.42 at n=9, unsettled")

    # ---------------------------------------------- physics informed
    print("\nPHYSICS-INFORMED  --  SEIR-SEI constraints added to the objective")
    ext: list[dict] = []
    for f in sorted((RESULTS / "physics_extended").glob("physics_envelope_*.json")):
        ext.extend(json.loads(f.read_text(encoding="utf-8")))
    ext = [r for r in ext if r["arch"] != "persistence"]
    for arch, arm, note in (("STGAT", "base", ""), ("STGAT", "spatial", "  <- holds"),
                            ("AAGCN", "base", ""),
                            ("AAGCN", "envelope", "  <- dissolves at n=75")):
        sub = [r for r in ext if r["arch"] == arch and r["increment"] == arm]
        if not sub:
            continue
        rows = [r for r in ext if r["arch"] == arch]
        pair = paired(rows, arm, "base", "increment") if arm != "base" else None
        line(f"  {arch} {arm}", *summarise([r["RMSE_clean"] for r in sub],
                                           [FLOOR[r["origin"]] for r in sub]),
             pair=pair, note=note)

    log = [r for r in json.loads((RESULTS / "physics_logsmooth"
                                 / "physics_envelope_STGAT.json").read_text("utf-8"))
           if r["arch"] == "STGAT"]
    sub = [r for r in log if r["increment"] == "spatial_log"]
    if sub:
        line("  STGAT spatial_log", *summarise([r["RMSE_clean"] for r in sub],
                                               [FLOOR[r["origin"]] for r in sub]),
             pair=paired(log, "spatial_log", "base", "increment"),
             note="  <- our variant, worse")

    print("\n" + "=" * 108)
    print("Reading it: `vs floor` is absolute standing against naive persistence;")
    print("`paired d` is the effect of the change itself, architecture held fixed.")
    print("Negative is better in both. They disagree, and that disagreement is the")
    print("finding: the physics constraint reliably improves its model without")
    print("moving that model past the baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
