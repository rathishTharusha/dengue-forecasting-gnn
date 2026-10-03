"""One command that tells two people whether they are running the same experiment.

Different RMSE numbers between collaborators are almost never a disagreement
about models. They are a disagreement about the protocol -- dataset build,
origin count, window, horizon, how seeds are aggregated, and how the paired test
is clustered. This script prints a fingerprint of all of that, plus the
persistence floor it implies, so the comparison can be checked in one line before
anyone argues about a model.

Persistence is the ideal check: it has no parameters and no randomness, so if two
people report a different persistence RMSE they are provably not running the same
evaluation, whatever their models say.

Usage::

    python analysis/_build/protocol_check.py                 # the frozen protocol
    python analysis/_build/protocol_check.py --all           # every combination
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "analysis" / "lib", REPO / "analysis" / "_build", REPO / "src"):
    sys.path.insert(0, str(_p))

import run_corrected_benchmark as rcb  # noqa: E402
import run_s5_seir_gnn_v2 as v2  # noqa: E402

# ---------------------------------------------------------------- the protocol
FROZEN = dict(
    dataset="rebuilt",
    protocol="9origin",
    origins=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    window=3,
    horizon=3,
    seeds=[0, 1, 2],
    normalisation="train_only",
    metric="mean of per-origin split RMSE, raw counts",
    seed_aggregation="average seeds WITHIN an origin before any test",
    cluster_unit="origin",
    n_clusters=9,
    test="exact paired sign-flip, two-sided",
    multiplicity="Holm across arms compared to the same baseline",
)


def persistence(cases, folds, horizon):
    """RMSE of 'repeat the last observed week', per origin. No parameters, no seed."""
    out = []
    for f in folds:
        idx = np.asarray(f.test_index)
        idx = idx[idx + horizon <= len(cases)]
        if len(idx) == 0:
            return []
        truth = np.stack([cases[i:i + horizon].T for i in idx])
        pers = np.repeat(cases[idx - 1][:, :, None], horizon, axis=2)
        out.append(float(np.sqrt(((pers - truth) ** 2).mean())))
    return out


def min_attainable_p(n_clusters: int) -> float:
    """Smallest two-sided p an exact sign-flip test can return with n clusters."""
    flips = np.array(list(itertools.product([-1, 1], repeat=n_clusters)))
    x = np.ones(n_clusters)
    return float(np.mean(np.abs((flips * x).mean(1)) >= abs(x.mean())))


def fingerprint(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def report(dataset, protocol, horizon, label=""):
    cases, _adj, _artifact, missing, _ = rcb.prepare(dataset)
    folds = v2.build_folds(cases, missing, protocol)
    per = persistence(cases, folds, horizon)
    if not per:
        print(f"  {dataset:10s} {protocol:9s} H={horizon:<3d} not evaluable "
              f"(test window runs past the end of the series)")
        return
    n = len(folds)
    print(f"  {dataset:10s} {protocol:9s} H={horizon:<3d} folds={n}  "
          f"persistence={np.mean(per):8.4f}  min p={min_attainable_p(n):.5f} {label}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="print every combination")
    args = ap.parse_args()

    print("=" * 78)
    print("FROZEN PROTOCOL   fingerprint", fingerprint(FROZEN))
    print("=" * 78)
    for k, v in FROZEN.items():
        print(f"  {k:18s} {v}")

    cases, _adj, _artifact, missing, _ = rcb.prepare(FROZEN["dataset"])
    folds = v2.build_folds(cases, missing, FROZEN["protocol"])
    per = persistence(cases, folds, FROZEN["horizon"])
    print(f"\n  origins found      {[f.origin for f in folds]}")
    print(f"  train/val/test     {[(len(f.train_index), len(f.val_index), len(f.test_index)) for f in folds][:3]} ...")
    print("\n  PERSISTENCE FLOOR (no parameters, no randomness -- the check value)")
    for f, v_ in zip(folds, per):
        print(f"      origin {f.origin:<5} {v_:9.4f}")
    print(f"      {'MEAN':<12} {np.mean(per):9.4f}   <-- both sides must print this")

    n = len(folds)
    print(f"\n  CLUSTERING: {n} origins. Seeds are averaged within an origin first,")
    print(f"  because three seeds share one test window and are not independent.")
    print(f"  Smallest p this test can ever return: {min_attainable_p(n):.5f}")
    for bad in (3, 6):
        if bad != n:
            print(f"      with {bad} clusters it would be {min_attainable_p(bad):.5f} "
                  f"-- nothing below that is attainable")

    if args.all:
        print("\n" + "=" * 78)
        print("EVERY COMBINATION -- find which one a differing number came from")
        print("=" * 78)
        for ds in ("rebuilt", "reordered", "original"):
            for proto in ("9origin", "3origin"):
                for H in (3, 6, 12):
                    tag = "  <== FROZEN" if (ds, proto, H) == (
                        FROZEN["dataset"], FROZEN["protocol"], FROZEN["horizon"]) else ""
                    try:
                        report(ds, proto, H, tag)
                    except Exception as exc:
                        print(f"  {ds:10s} {proto:9s} H={H:<3d} unavailable ({exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
