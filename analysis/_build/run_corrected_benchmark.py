"""Re-run the core benchmark on the original and the corrected case series.

``docs/ARRAY_AUDIT.md`` found the processed array out of date order, with a 2023
block inside every training split. This measures what that did to the results,
with **one training loop across all datasets** -- the physics sweep's own
``run_fold`` -- so the data is the only thing that changes.

Datasets
--------
``original``   the array as the benchmark used it (459 rows, artifact at row 395)
``reordered``  the same rows in true date order, duplicate removed (451 weeks)
``rebuilt``    every source report on a regular weekly grid (559 weeks)

Protocol
--------
The frozen protocol, unchanged: 3 rolling origins (0.55 / 0.70 / 0.85), window 3,
horizon 3, train-only normalisation, early stopping on validation, pooled RMSE.
Two things are made dataset-aware, because hard-coding them would silently score
the corrected series wrongly:

* the artifact is located **by report** (2021, no. 2), not by row 395;
* nothing is ever filled. ``rebuilt`` leaves its 7 weeks with no source report
  missing, and any window whose input or target touches one is dropped, so no
  model reads or is scored against a value nobody reported.

Arms: ``base`` (the reproduced architectures) and ``spatial`` (the physics result
the paper reports), plus persistence.

Run::

    python analysis/_build/run_corrected_benchmark.py --arch A3TGCN --datasets original reordered
    python analysis/_build/run_corrected_benchmark.py --persistence-only
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "analysis" / "_build"))
sys.path.insert(0, str(REPO / "src"))

import adaptive as base  # noqa: E402
import improved as imp  # noqa: E402
import physics as phys  # noqa: E402

NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
CORRECTED = REPO / "data" / "corrected"
OUT_DIR = REPO / "analysis" / "results" / "corrected_benchmark"

WINDOW, HORIZON = 3, 3
DATASETS = ("original", "reordered", "rebuilt")
_ARTIFACT_WINDOWS = imp.artifact_windows


def load(dataset: str):
    """Return ``(cases, adjacency, artifact_row, missing_rows)`` for one dataset.

    ``cases`` may contain NaN: ``rebuilt`` leaves weeks with no source report
    missing rather than filling them.
    """
    _, adjacency, _ = base.load_dataset(NPY, ADJ)
    if dataset == "original":
        cases, _, _ = base.load_dataset(NPY, ADJ)
        return cases, adjacency, imp.ARTIFACT_WEEK, np.array([], dtype=int)
    arr = np.load(CORRECTED / f"{dataset}_cases.npy")
    index = pd.read_csv(CORRECTED / f"{dataset}_index.csv")
    # `rebuilt` repairs the artifact report from the published PDF, so it may have
    # none left; a row far outside the series then flags no window.
    flagged = index.index[index["is_artifact"]]
    artifact = int(flagged[0]) if len(flagged) else -10**6
    missing = index.index[index["status"].eq("missing")].to_numpy()
    cases = arr[..., 0].astype(np.float64)
    if not np.array_equal(np.flatnonzero(np.isnan(cases).any(axis=1)), missing):
        raise SystemExit(f"{dataset}: NaN rows do not match the rows flagged missing")
    return cases, adjacency, artifact, missing


def build_folds_masked(cases: np.ndarray, missing: np.ndarray):
    """The frozen protocol's folds, on a series with missing weeks and no filling.

    Split boundaries are exactly ``adaptive.build_folds``'s -- computed on the
    full list of window positions -- so a missing week never shifts a fold. Then:

    * normalisation statistics ignore missing weeks (``nanmean``/``nanstd``);
    * any window whose input ``[i - W, i)`` or target ``[i, i + H)`` touches a
      missing week is dropped from its split. Nothing is ever filled, so no
      window can read an invented value or a value borrowed from a later week.
    """
    if missing.size == 0:
        return base.build_folds(cases, WINDOW, HORIZON)
    bad = set(int(m) for m in missing)
    n_weeks = cases.shape[0]
    ids = list(range(WINDOW, n_weeks - HORIZON))
    clean = lambda i: not any(t in bad for t in range(i - WINDOW, i + HORIZON))  # noqa: E731
    to = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    folds = []
    for origin in (0.55, 0.70, 0.85):
        cut = int(origin * len(ids))
        end = int(min(origin + 0.15, 1.0) * len(ids))
        train_ids = [i for i in ids[: cut - 30] if clean(i)]
        val_ids = [i for i in ids[cut - 30: cut] if clean(i)]
        test_ids = [i for i in ids[cut:end] if clean(i)]
        history = np.log1p(cases[: ids[: cut - 30][-1] + 1])
        mean, std = float(np.nanmean(history)), float(np.nanstd(history) + 1e-8)
        z = (np.log1p(cases) - mean) / std

        def pack(chosen, scaled=z):
            x = np.stack([scaled[i - WINDOW: i].T for i in chosen])
            y = np.stack([scaled[i: i + HORIZON].T for i in chosen])
            p = np.stack([np.repeat(scaled[i - 1][:, None], HORIZON, axis=1) for i in chosen])
            if np.isnan(x).any() or np.isnan(y).any():
                raise AssertionError("a kept window reads a missing week")
            return to(x), to(y), to(p)

        folds.append(base.Fold(origin, *pack(train_ids), *pack(val_ids), *pack(test_ids),
                               mean, std, np.asarray(test_ids), np.asarray(train_ids),
                               np.asarray(val_ids)))
    return folds


def prepare(dataset: str):
    cases, adjacency, artifact, missing = load(dataset)
    return cases, adjacency, artifact, missing, build_folds_masked(cases, missing)


def persistence_records(dataset: str) -> list[dict]:
    cases, _, artifact, missing, folds = prepare(dataset)
    out = []
    for f in folds:
        mask = _ARTIFACT_WINDOWS(f.test_index, WINDOW, HORIZON, week=artifact)
        sc = base.persistence_scores(f, cases, HORIZON, mask)
        out.append(dict(dataset=dataset, arch="persistence", increment="base", origin=f.origin,
                        seed=-1, weeks=int(len(cases)), test_windows=int(len(f.test_index)),
                        artifact_windows=int(mask.sum()), missing_weeks=int(missing.size), **sc))
    return out


def run_dataset(dataset: str, archs, arms, seeds, epochs, out_dir: Path) -> list[dict]:
    import run_physics_experiments as rpe

    cases, adjacency, artifact, missing, folds = prepare(dataset)
    # run_fold scores with imp.artifact_windows; point it at this dataset's artifact.
    imp.artifact_windows = functools.partial(_ARTIFACT_WINDOWS, week=artifact)
    src, dst = np.nonzero(adjacency)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    adj_dense = torch.tensor(adjacency, dtype=torch.float32)
    hist = {f.origin: {s: phys.window_history(cases, getattr(f, f"{s}_index"))
                       for s in ("train", "val", "test")} for f in folds}

    if missing.size and set(arms) - {"base", "spatial"}:
        # Other arms read a 9-week history that can include missing weeks.
        raise SystemExit(f"arms {sorted(set(arms) - {'base', 'spatial'})} need gap-aware history")
    records = persistence_records(dataset)
    for arch_name in archs:
        path = out_dir / f"corrected_{arch_name}_{dataset}.json"
        for arm, fold, seed in itertools.product(arms, folds, seeds):
            t0 = time.time()
            scales = torch.tensor(cases[fold.train_index].mean(axis=0), dtype=torch.float32)
            try:
                sc = rpe.run_fold(arch_name=arch_name, mode=arm, fold=fold, cases=cases,
                                  hist=hist[fold.origin], edge_index=edge_index,
                                  adj_dense=adj_dense, district_scales=scales,
                                  seed=seed, epochs=epochs)
            except Exception as exc:  # keep the sweep alive; the failure is logged
                print(f"{dataset:9s} {arch_name:7s} {arm:8s} o{fold.origin} s{seed} "
                      f"FAILED {type(exc).__name__}: {exc}", flush=True)
                continue
            records.append(dict(dataset=dataset, arch=arch_name, increment=arm,
                                origin=fold.origin, seed=seed, **sc))
            print(f"{dataset:9s} {arch_name:7s} {arm:8s} o{fold.origin} s{seed} "
                  f"RMSE {sc['RMSE']:7.2f} clean {sc['RMSE_clean']:6.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            path.write_text(json.dumps(records, indent=1), encoding="utf-8")
    imp.artifact_windows = _ARTIFACT_WINDOWS
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", nargs="*", default=["A3TGCN"])
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--arms", nargs="*", default=["base", "spatial"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--persistence-only", action="store_true")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.persistence_only:
        rows = [r for d in args.datasets for r in persistence_records(d)]
        (out_dir / "persistence.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        df = pd.DataFrame(rows)
        print(df[["dataset", "origin", "weeks", "test_windows", "artifact_windows",
                  "RMSE", "RMSE_clean"]].to_string(index=False))
        print("\nfloor (mean of origins):")
        print(df.groupby("dataset", sort=False)[["RMSE", "RMSE_clean"]].mean().round(3).to_string())
        return 0

    for dataset in args.datasets:
        run_dataset(dataset, args.arch, args.arms, tuple(range(args.seeds)), args.epochs, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
