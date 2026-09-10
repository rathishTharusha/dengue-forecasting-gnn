"""Head-to-head: the repo's Phase-2 baseline against the reproduction-derived one.

The Phase-2 conclusions rest on a baseline that loses to persistence. Before any
of that work is deleted, the premise has to be checked by *running* it, not by
reading its notes.

Both models are run on the **same folds, same seeds, same protocol** --
`src.dengue_gnn.experiment.Config`'s own defaults, which happen to match the
frozen protocol in `docs/ROADMAP.md` exactly:

    window 3 -> horizon 3, origins (0.55, 0.70, 0.85), test_frac 0.15,
    val_weeks 30, seeds (0, 1, 2), residual over persistence, log1p,
    hidden 64, dropout 0.1

so any difference is the model, not the harness.

Run::

    python analysis/_build/bench_baselines.py            # full
    python analysis/_build/bench_baselines.py --quick    # 1 seed, fewer epochs

Writes `analysis/results/baseline_benchmark.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "analysis" / "lib"))

from dengue_gnn.experiment import Config, load_dataset, rolling_origin  # noqa: E402

import adaptive as ours  # noqa: E402

NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
OUT = REPO / "analysis" / "results" / "baseline_benchmark.json"


def theirs(seeds: tuple[int, ...], adaptive: bool) -> list[dict]:
    """Run the repo's own Phase-2 model through its own harness."""
    raw, adj_fixed, _ = load_dataset(NPY, ADJ)
    cfg = Config(seeds=seeds, use_adaptive=adaptive, temporal="none", spatial="gcn")
    rows = rolling_origin(raw, adj_fixed, cfg, include_persistence=True, verbose=False)
    return rows


def summarise_theirs(rows: list[dict], tag: str) -> dict:
    """Pooled-across-horizon records only, split model vs persistence."""
    pooled = [r for r in rows if r.get("horizon") == 0]
    # Their harness labels these "Persistence" / "DenseGCN" / "AdaptiveGCN".
    # A case-sensitive mismatch here silently averages the persistence rows into
    # the model's score, which is exactly the kind of error this script exists
    # to avoid making about someone else's work.
    model = [r for r in pooled if str(r.get("model", "")).lower() != "persistence"]
    persist = [r for r in pooled if str(r.get("model", "")).lower() == "persistence"]
    assert model and persist, f"unexpected model labels: {set(r.get('model') for r in pooled)}"
    out = {
        "arm": tag,
        "rmse": float(np.mean([r["rmse"] for r in model])),
        "rmse_sd": float(np.std([r["rmse"] for r in model])),
        "mae": float(np.mean([r["mae"] for r in model])),
        "n": len(model),
    }
    if persist:
        out["persistence_rmse"] = float(np.mean([r["rmse"] for r in persist]))
    return out


def ourbench(seeds: tuple[int, ...], mode: str, epochs: int) -> dict:
    """Run the reproduction-derived STGNN on the same folds."""
    cases, adjacency, _ = ours.load_dataset(NPY, ADJ)
    fixed = torch.tensor(adjacency, dtype=torch.float32)
    folds = ours.build_folds(cases)
    scores, per_fold = [], []
    for fold in folds:
        for seed in seeds:
            _, sc = ours.train_one(fold, fixed, mode, seed=seed, epochs=epochs)
            scores.append(sc)
            per_fold.append({"origin": fold.origin, "seed": seed, **sc})
    persist = [ours.persistence_scores(f, cases)["RMSE"] for f in folds]
    return {
        "arm": f"ours:{mode}",
        "rmse": float(np.mean([s["RMSE"] for s in scores])),
        "rmse_sd": float(np.std([s["RMSE"] for s in scores])),
        "mae": float(np.mean([s["MAE"] for s in scores])),
        "n": len(scores),
        "persistence_rmse": float(np.mean(persist)),
        "runs": per_fold,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="1 seed, 30 epochs")
    args = ap.parse_args()

    seeds = (0,) if args.quick else (0, 1, 2)
    epochs = 30 if args.quick else 150

    results = []
    started = time.time()

    for tag, adaptive in (("theirs:dense_fixed", False), ("theirs:adaptive", True)):
        t0 = time.time()
        rows = theirs(seeds, adaptive)
        rec = summarise_theirs(rows, tag)
        rec["seconds"] = round(time.time() - t0, 1)
        results.append(rec)
        print(f"{tag:22s} RMSE {rec['rmse']:7.2f} +/- {rec['rmse_sd']:5.2f} "
              f"MAE {rec['mae']:6.2f}  n={rec['n']}  ({rec['seconds']}s)", flush=True)

    for mode in ("fixed", "adaptive"):
        t0 = time.time()
        rec = ourbench(seeds, mode, epochs)
        rec["seconds"] = round(time.time() - t0, 1)
        results.append(rec)
        print(f"{'ours:' + mode:22s} RMSE {rec['rmse']:7.2f} +/- {rec['rmse_sd']:5.2f} "
              f"MAE {rec['mae']:6.2f}  n={rec['n']}  ({rec['seconds']}s)", flush=True)

    persist = next((r["persistence_rmse"] for r in results if "persistence_rmse" in r), None)
    print(f"\npersistence floor on these folds: {persist:.2f}")
    print(f"total {time.time() - started:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
