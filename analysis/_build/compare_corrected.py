"""Compare the benchmark on the original array against the corrected case series.

Reads the ``corrected_<arch>_<dataset>.json`` files the Kaggle kernels write and
answers three questions, each per dataset:

1. **Reproduction** -- does ``original`` recover the numbers the paper reports?
   If not, nothing else in this table can be trusted.
2. **Standing** -- does each architecture beat persistence? Seeds are averaged
   within an origin first, so ``wins`` counts independent folds.
3. **The physics effect** -- ``spatial`` minus ``base`` on matched
   (origin, seed), the paper's paired claim.

Run::

    python analysis/_build/compare_corrected.py --dir analysis/results/corrected_benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent.parent

#: Base-arm artifact-free RMSE from the earlier 3-origin physics sweep, for the
#: reproduction check (analysis/results/beat_baseline/physics_envelope_*.json).
EARLIER_BASE = {"A3TGCN": 29.022, "AAGCN": 29.966, "STGAT": 29.517}


def load(directory: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(directory.glob("corrected_*.json")):
        rows.extend(json.loads(f.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no corrected_*.json under {directory}")
    return pd.DataFrame(rows).drop_duplicates(["dataset", "arch", "increment", "origin", "seed"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(REPO / "analysis" / "results" / "corrected_benchmark"))
    ap.add_argument("--out", default=None, help="Write the summary table as Markdown here")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = load(Path(args.dir))

    floor = df[df.arch == "persistence"].groupby(["dataset", "origin"])["RMSE_clean"].mean()
    model = df[df.arch != "persistence"]

    print("=== 1. floor (artifact-free, mean of origins) ===")
    print(floor.groupby("dataset").mean().round(3).to_string())

    print("\n=== 2. standing against persistence, clustered by origin ===")
    rows = []
    for (ds, arch, inc), g in model.groupby(["dataset", "arch", "increment"]):
        per = g.groupby("origin")["RMSE_clean"].mean()
        f = np.array([floor[(ds, o)] for o in per.index])
        d = per.to_numpy() - f
        rows.append(dict(dataset=ds, arch=arch, arm=inc, n_origins=len(per),
                         runs=len(g), RMSE=per.mean(), floor=f.mean(), vs_floor=d.mean(),
                         wins=int((d < 0).sum())))
    stand = pd.DataFrame(rows)
    order = {"original": 0, "reordered": 1, "rebuilt": 2}
    stand = stand.sort_values(["arch", "arm", "dataset"], key=lambda s: s.map(order)
                              if s.name == "dataset" else s)
    print(stand.round(3).to_string(index=False))

    print("\n=== 3. physics: spatial - base, paired on (origin, seed) ===")
    for (ds, arch), g in model.groupby(["dataset", "arch"]):
        b = g[g.increment == "base"].set_index(["origin", "seed"])["RMSE_clean"]
        s = g[g.increment == "spatial"].set_index(["origin", "seed"])["RMSE_clean"]
        common = b.index.intersection(s.index)
        if len(common) < 3:
            continue
        dd = s.loc[common] - b.loc[common]
        p = stats.ttest_rel(s.loc[common], b.loc[common])[1] if dd.std() > 0 else float("nan")
        print(f"  {ds:9s} {arch:7s} n={len(dd):2d}  d={dd.mean():+.4f}  "
              f"better {int((dd < 0).sum())}/{len(dd)}  p={p:.4g}")

    print("\n=== reproduction check (original, base) ===")
    for arch, ref in EARLIER_BASE.items():
        hit = stand[(stand.dataset == "original") & (stand.arch == arch) & (stand.arm == "base")]
        if len(hit):
            got = float(hit.RMSE.iloc[0])
            print(f"  {arch:7s} now {got:.3f}  earlier {ref:.3f}  diff {got - ref:+.3f}")

    if args.out:
        try:
            content = stand.round(3).to_markdown(index=False)
        except ImportError:
            content = stand.round(3).to_string(index=False)
        Path(args.out).write_text(content, encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
