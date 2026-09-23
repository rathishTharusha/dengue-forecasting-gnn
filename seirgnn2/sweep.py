"""Run a list of configurations across every (origin, seed) and save one row each.

Rows are never pre-aggregated: one row per (config, origin, seed) is what makes
the paired tests in ``stats.py`` possible, and the repo's logging convention
asks for exactly that.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import core  # noqa: E402
import corrected_data as cd  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
SEEDS = (0, 1, 2)


def _job(spec: dict) -> dict:
    import torch

    import train
    torch.set_num_threads(1)
    data = cd.load()
    cfg = {k: v for k, v in spec.items() if k not in ("origin", "seed", "name", "window")}
    folds = core.build_folds(data.cases, data.missing, spec.get("window", core.WINDOW))
    fold = next(f for f in folds if f.origin == spec["origin"])
    edge, fixed = core.adjacency(data.names)
    t0 = time.time()
    row, _, _ = train.run_fold(data, fold, seed=spec["seed"], edge=edge, fixed=fixed, **cfg)
    row["name"] = spec["name"]
    row["elapsed"] = round(time.time() - t0, 1)
    return row


def persistence_rows(windows: tuple[int, ...] = (core.WINDOW,)) -> list[dict]:
    """One persistence row per (window, origin).

    A longer window shifts the fold boundaries, so each window needs its own
    baseline; pairing an arm against the wrong one would compare across
    different evaluation windows.
    """
    data = cd.load()
    rows = []
    for w, fold in [(w, f) for w in windows
                    for f in core.build_folds(data.cases, data.missing, w)]:
        pack = core.build_tensors(data, fold, "test", False, False, False)
        val = core.build_tensors(data, fold, "val", False, False, False)
        row = core.score(pack["p_raw"].numpy(), pack["y_raw"].numpy())
        name = "persistence" if w == core.WINDOW else f"persistence w={w}"
        row.update(name=name, backbone="-", head="-", loss="-", origin=fold.origin,
                   seed=-1, window=w,
                   val_RMSE=core.rmse(val["p_raw"].numpy(), val["y_raw"].numpy()))
        rows.append(row)
    return rows


def run(configs: list[dict], out_name: str, workers: int = 6) -> list[dict]:
    jobs = [dict(c, origin=o, seed=s) for c, (o, s)
            in itertools.product(configs, itertools.product(core.ORIGINS, SEEDS))]
    print(f"{len(configs)} configs x {len(core.ORIGINS)} origins x {len(SEEDS)} seeds "
          f"= {len(jobs)} runs on {workers} workers", flush=True)
    windows = tuple(sorted({c.get("window", core.WINDOW) for c in configs}))
    rows, done, t0 = persistence_rows(windows), 0, time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, j) for j in jobs]
        for fut in as_completed(futures):
            try:
                row = fut.result()
                rows.append(row)
                done += 1
                print(f"[{done:3d}/{len(jobs)}] {row['name']:26s} o{row['origin']} s{row['seed']} "
                      f"val {row['val_RMSE']:7.2f} test {row['RMSE']:7.2f} ({row['elapsed']}s)",
                      flush=True)
            except Exception as exc:  # a dead worker must not silently shrink the grid
                done += 1
                print(f"[{done:3d}/{len(jobs)}] FAILED: {exc!r}", flush=True)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{out_name}.json"
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {path} ({len(rows)} rows) in {time.time() - t0:.0f}s", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    import grids
    run(getattr(grids, args.grid)(args.epochs), args.grid, args.workers)
