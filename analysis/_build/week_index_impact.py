"""How many forecast windows read a week whose date was wrong, per arm family and protocol.

The eight rows below carried a wrong ``week_start`` (see ``fix_week_index.py``). A window
whose forecast week is ``i`` reads covariate rows at fixed offsets:

  season                row i                      (day of year of week_start[i])
  climate lags 2-4      rows i-4 .. i-2
  climate blocks 2-13   rows i-13 .. i-2
  climate blocks 2-25   rows i-25 .. i-2
  COVID policy/mobility row i-2
  stringency prior      row i-1

Counts are per split (train / validation / test) for the three- and nine-origin protocols
of ``seirgnn2/core.py``. Writes ``analysis/results/week_index_impact.json``. No training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "seirgnn2"))

import core  # noqa: E402
import corrected_data as cd  # noqa: E402

BAD = [0, 209, 297, 346, 378, 388, 537, 554]
FAMILIES = {"season": (0, 0), "climate lags 2-4": (2, 4), "climate blocks 2-13": (2, 13),
            "climate blocks 2-25": (2, 25), "COVID policy and mobility (lag 2)": (2, 2),
            "stringency prior (lag 1)": (1, 1)}
PROTOCOLS = {"three origins": (core.ORIGINS, core.TEST_FRAC),
             "nine origins, wide": (core.ORIGINS_9, core.TEST_FRAC_9),
             "nine origins, frozen": (core.ORIGINS_F9, core.TEST_FRAC_F9)}


def touches(idx: np.ndarray, lo: int, hi: int) -> np.ndarray:
    bad = np.array(BAD)
    return np.array([bool(((bad >= i - hi) & (bad <= i - lo)).any()) for i in idx])


def main() -> None:
    data = cd.load()
    out: dict = {"bad_rows": BAD, "families": {}}
    for fam, (lo, hi) in FAMILIES.items():
        out["families"][fam] = {}
        for pname, (origins, frac) in PROTOCOLS.items():
            rows = []
            for f in core.build_folds(data.cases, data.missing, core.WINDOW, origins, frac):
                rows.append({"origin": f.origin, **{
                    s: [int(touches(f.idx[s], lo, hi).sum()), int(len(f.idx[s]))]
                    for s in ("train", "val", "test")}})
            out["families"][fam][pname] = rows
    path = REPO / "analysis" / "results" / "week_index_impact.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for fam in out["families"]:
        for pname, rows in out["families"][fam].items():
            tr = sum(r["train"][0] for r in rows)
            va = sum(r["val"][0] for r in rows)
            te = sum(r["test"][0] for r in rows)
            n = sum(r["train"][1] for r in rows), sum(r["val"][1] for r in rows), sum(r["test"][1] for r in rows)
            print(f"{fam:36s} {pname:20s} affected windows train {tr}/{n[0]}  val {va}/{n[1]}  test {te}/{n[2]}")


if __name__ == "__main__":
    main()
