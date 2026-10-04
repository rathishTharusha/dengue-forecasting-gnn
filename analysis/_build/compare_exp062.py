"""EXP-062: compare the week-index re-run of the Kaggle notebook with EXP-050, run by run.

Usage::

    python analysis/_build/compare_exp062.py <new runs.jsonl> [--old full_paper/outputs/kaggle_run/runs.jsonl]

Two tables are printed and written next to the new file (``exp062_comparison.json``):

* **Reproducibility check.** Configurations that read no date-joined input (no seasonal
  features, no climate) should reproduce EXP-050. For each one the largest absolute
  difference in test RMSE and validation RMSE over its (origin, seed) runs, next to the
  seed-level noise (the mean standard deviation across seeds within an origin in EXP-050).
  A difference above that noise is flagged; stop and explain it before using any new number.
* **Affected arms.** Mean test and validation RMSE, old and new, per configuration.

The affected set is read from the notebook's own configuration source, not typed here.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent


def affected_configs() -> dict[str, set[str]]:
    src = (REPO / "full_paper" / "kaggle" / "src" / "70_experiments.py").read_text(encoding="utf-8")
    ns: dict = {}
    exec(src[src.index("ENC = ["): src.index("SEEDS = (0, 1, 2)")], ns)  # noqa: S102 - our own config block
    out = {}
    for key, cfgs in (("three", ns["CONFIGS3"]), ("nine", ns["CONFIGS9"])):
        out[key] = {k for k, v in cfgs.items()
                    if v.get("use_season") or v.get("use_climate") or v.get("clim_blocks") or v.get("use_ndvi")}
    return out


def load(path: Path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    by = defaultdict(dict)
    for r in rows:
        by[(r["origin_set"], r["name"])][(round(r["origin"], 6), r["seed"])] = r
    return by


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("new", type=Path)
    ap.add_argument("--old", type=Path, default=REPO / "full_paper/outputs/kaggle_run/runs.jsonl")
    a = ap.parse_args()
    old, new = load(a.old), load(a.new)
    aff = affected_configs()
    report = {"unaffected": [], "affected": [], "flagged": []}
    print("REPRODUCIBILITY CHECK (configurations with no date-joined input)")
    print(f"{'config':44s} {'runs':>4s} {'max |d test|':>12s} {'max |d val|':>12s} {'seed noise':>10s}")
    for key in sorted(old):
        oset, name = key
        if name in aff[{"three": "three", "nine": "nine"}[oset]] or key not in new:
            continue
        keys = sorted(set(old[key]) & set(new[key]))
        dt = [abs(new[key][k]["RMSE"] - old[key][k]["RMSE"]) for k in keys]
        dv = [abs(new[key][k]["val_RMSE"] - old[key][k]["val_RMSE"]) for k in keys]
        per_origin = defaultdict(list)
        for (o, _), r in old[key].items():
            per_origin[o].append(r["RMSE"])
        noise = float(np.mean([np.std(v, ddof=1) for v in per_origin.values() if len(v) > 1]))
        row = {"config": name, "origin_set": oset, "runs": len(keys), "max_test": max(dt), "max_val": max(dv),
               "seed_noise": noise}
        report["unaffected"].append(row)
        flag = max(dt) > max(noise, 1e-9)
        if flag:
            report["flagged"].append(row)
        print(f"{name[:44]:44s} {len(keys):4d} {max(dt):12.4f} {max(dv):12.4f} {noise:10.4f}{'  <-- above noise' if flag else ''}")
    print("\nAFFECTED ARMS (mean over runs)")
    print(f"{'config':44s} {'set':>5s} {'val old':>8s} {'val new':>8s} {'test old':>9s} {'test new':>9s}")
    for key in sorted(old):
        oset, name = key
        if name not in aff[oset] or key not in new:
            continue
        m = lambda d, f: float(np.mean([r[f] for r in d.values()]))  # noqa: E731
        row = {"config": name, "origin_set": oset, "val_old": m(old[key], "val_RMSE"), "val_new": m(new[key], "val_RMSE"),
               "test_old": m(old[key], "RMSE"), "test_new": m(new[key], "RMSE")}
        report["affected"].append(row)
        print(f"{name[:44]:44s} {oset:>5s} {row['val_old']:8.2f} {row['val_new']:8.2f} {row['test_old']:9.2f} {row['test_new']:9.2f}")
    out = a.new.parent / "exp062_comparison.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nflagged unaffected configurations: {len(report['flagged'])}; wrote {out}")
    return 1 if report["flagged"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
