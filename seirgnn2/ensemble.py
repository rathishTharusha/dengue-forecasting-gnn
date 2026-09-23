"""Equal-weight ensembles from saved forecasts -- remedy R5.

Every multi-team forecasting evaluation finds the ensemble most consistently
accurate (Johansson et al. 2019; Cramer et al. 2022), and Cramer et al. are
precise that it wins on consistency, *"although it made the single best
forecast less frequently than any other model"*. That only happens when the
members err differently. EXP-039 found ASTGCN, AAGCN and the LSTM correlate at
0.98-0.99 -- effectively one model -- so the ensembles worth testing pair a
neural arm with a structurally different one.

Weights are equal and fixed in advance. Fitting them, even on validation, would
turn a pre-registered comparison into another search.

Reads ``results/<grid>_preds.pkl`` from ``sweep.py --keep`` and writes
``results/<grid>+ens.json``: the grid's own rows plus one row per ensemble per
(origin, seed), in the same schema, so ``stats.py <grid>+ens`` tests ensembles
exactly like any other arm.

    python seirgnn2/ensemble.py remedies "B" "R4b knn"
    python seirgnn2/ensemble.py remedies "B" "R4a nbglm" "R4b knn"
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import core

RESULTS = Path(__file__).resolve().parent / "results"


def build(grid: str, members: list[str]) -> list[dict]:
    preds = pickle.loads((RESULTS / f"{grid}_preds.pkl").read_bytes())
    have = {(o, s) for (n, o, s) in preds}
    name = "ENS[" + " + ".join(members) + "]"
    rows = []
    for o, s in sorted(have):
        runs = [preds.get((m, o, s)) for m in members]
        if any(r is None for r in runs):
            continue
        test = np.mean([r["test"]["pred"] for r in runs], axis=0)
        val = np.mean([r["val"]["pred"] for r in runs], axis=0)
        row = core.score(test, runs[0]["test"]["truth"])
        row.update(val_RMSE=core.rmse(val, runs[0]["val"]["truth"]), origin=o, seed=s,
                   name=name, backbone="ensemble", head="-", loss="-", members=members)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("members", nargs="+", help="arm names exactly as in the grid")
    args = ap.parse_args()

    base_path = RESULTS / f"{args.grid}+ens.json"
    rows = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else \
        json.loads((RESULTS / f"{args.grid}.json").read_text(encoding="utf-8"))
    new = build(args.grid, args.members)
    if not new:
        raise SystemExit(f"no (origin, seed) has forecasts for all of {args.members}")
    rows = [r for r in rows if r["name"] != new[0]["name"]] + new
    base_path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    val = np.mean([r["val_RMSE"] for r in new])
    print(f"{new[0]['name']}: {len(new)} rows, mean val RMSE {val:.2f} -> {base_path.name}")


if __name__ == "__main__":
    main()
