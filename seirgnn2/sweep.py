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


def _origins(spec: dict) -> tuple[tuple[float, ...], float]:
    """Which origin set a config runs on -- the frozen three or the confirmatory nine."""
    if spec.get("origins") == "nine":
        return core.ORIGINS_9, core.TEST_FRAC_9
    return core.ORIGINS, core.TEST_FRAC


def _job(spec: dict, keep: bool = False):
    import torch

    import knn
    import train
    torch.set_num_threads(1)
    data = cd.load()
    cfg = {k: v for k, v in spec.items()
           if k not in ("origin", "seed", "name", "window", "origins")}
    origins, test_frac = _origins(spec)
    folds = core.build_folds(data.cases, data.missing, spec.get("window", core.WINDOW),
                             origins, test_frac)
    fold = next(f for f in folds if f.origin == spec["origin"])
    edge, fixed = core.adjacency(data.names)
    t0 = time.time()
    # k-NN analogue forecasting (remedy R4b) has no network to train, but it
    # honours the same folds, row schema and --keep contract.
    import gbm
    runner = {"knn": knn.run_fold, "gbm": gbm.run_fold}.get(cfg.get("backbone"), train.run_fold)
    res = runner(data, fold, seed=spec["seed"], edge=edge, fixed=fixed, keep=keep, **cfg)
    row, extra = (res[0], res[1]) if keep else (res[0], None)
    row["name"] = spec["name"]
    row["origins"] = spec.get("origins", "three")
    row["elapsed"] = round(time.time() - t0, 1)
    return row, extra


def persistence_rows(windows: tuple[int, ...] = (core.WINDOW,),
                     origin_sets: tuple[str, ...] = ("three",)) -> list[dict]:
    """One persistence row per (window, origin-set, origin).

    A longer window shifts the fold boundaries and the confirmatory nine evaluate
    different weeks than the frozen three, so each combination needs its own
    baseline; pairing an arm against the wrong one compares across different
    evaluation windows.
    """
    data = cd.load()
    rows = []
    combos = [(w, o, f) for w in windows for o in origin_sets
              for f in core.build_folds(data.cases, data.missing, w,
                                        *_origins({"origins": o}))]
    for w, oset, fold in combos:
        pack = core.build_tensors(data, fold, "test", False, False, False)
        val = core.build_tensors(data, fold, "val", False, False, False)
        row = core.score(pack["p_raw"].numpy(), pack["y_raw"].numpy())
        name = "persistence" if w == core.WINDOW else f"persistence w={w}"
        row.update(name=name, backbone="-", head="-", loss="-", origin=fold.origin,
                   seed=-1, window=w, origins=oset,
                   val_RMSE=core.rmse(val["p_raw"].numpy(), val["y_raw"].numpy()))
        rows.append(row)
    return rows


def run(configs: list[dict], out_name: str, workers: int = 6, keep: bool = False) -> list[dict]:
    """Run every config over its origins x seeds.

    With ``keep`` every run's forecasts for train / val / test are also written
    to ``results/<grid>_preds.pkl`` keyed by ``(name, origin, seed)``, so
    ensembles (remedy R5) and forecast-level diagnoses can be built afterwards
    without retraining.
    """
    jobs = [dict(c, origin=o, seed=s) for c in configs
            for o, s in itertools.product(_origins(c)[0], SEEDS)]
    print(f"{len(configs)} configs -> {len(jobs)} runs on {workers} workers", flush=True)
    windows = tuple(sorted({c.get("window", core.WINDOW) for c in configs}))
    osets = tuple(sorted({c.get("origins", "three") for c in configs}))
    rows, done, t0 = persistence_rows(windows, osets), 0, time.time()
    preds = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, j, keep) for j in jobs]
        for fut in as_completed(futures):
            try:
                row, extra = fut.result()
                rows.append(row)
                if keep:
                    preds[(row["name"], row["origin"], row["seed"])] = extra
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
    if keep:
        import pickle
        pp = OUT / f"{out_name}_preds.pkl"
        pp.write_bytes(pickle.dumps(preds))
        print(f"wrote {pp} ({len(preds)} runs)", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--keep", action="store_true", help="also save every run's forecasts")
    args = ap.parse_args()
    import grids
    run(getattr(grids, args.grid)(args.epochs), args.grid, args.workers, args.keep)
