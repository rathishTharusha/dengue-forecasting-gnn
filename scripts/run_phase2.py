"""Re-run the Phase-2 ablation under the corrected protocol.

Writes one row per (config, fold, seed, horizon) to results/phase2_runs.csv --
nothing pre-aggregated, so mean +/- std and paired significance tests stay
possible after the fact.

Configurations, in the order the ablation table needs them:

    dense_fixed       fixed geographic graph, dense propagation, no regulariser.
                      This is the control the Phase-2 comparison lacked: it
                      isolates the learned adjacency from the change of
                      propagation operator (review F6).
    adaptive          + learned gated adjacency (the EXP-003 configuration).
    adaptive_lam*     + spatial regularisation at four weights (EXP-004 sweep).

Persistence is emitted alongside every fold and is the bar that matters. It is
written by the ``dense_fixed`` configuration only, so that one is always needed.

Usage:
    python scripts/run_phase2.py [--quick]

    # sharded -- the configurations are independent, so on CPU this is the
    # difference between roughly four hours and one. Concatenate the shard CSVs.
    python scripts/run_phase2.py --only adaptive --out results/_shard_adaptive.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dengue_gnn.experiment import Config, load_dataset, rolling_origin  # noqa: E402

OUT = REPO / "results" / "phase2_runs.csv"
NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"


def configs(quick: bool) -> list[Config]:
    common = dict(epochs=3, patience=3, seeds=(0,)) if quick else {}
    return [
        Config(label="dense_fixed", use_adaptive=False, lambda_phys=0.0, **common),
        Config(label="adaptive", use_adaptive=True, lambda_phys=0.0, **common),
        Config(label="adaptive_lam0.01", use_adaptive=True, lambda_phys=0.01, **common),
        Config(label="adaptive_lam0.1", use_adaptive=True, lambda_phys=0.1, **common),
        Config(label="adaptive_lam1.0", use_adaptive=True, lambda_phys=1.0, **common),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 epochs, 1 seed -- pipeline check only")
    ap.add_argument(
        "--only",
        metavar="LABEL",
        help="run a single configuration by label. The configurations are independent, "
        "so the ablation can be sharded across processes and the shard CSVs concatenated; "
        "on CPU that is the difference between four hours and one. Persistence is emitted "
        "with 'dense_fixed' only, so that shard must always be run.",
    )
    ap.add_argument("--out", type=Path, default=OUT, help=f"output CSV (default {OUT.name})")
    args = ap.parse_args()
    out_path = Path(args.out).resolve()

    selected = configs(args.quick)
    if args.only:
        labels = [c.label for c in selected]
        if args.only not in labels:
            ap.error(f"unknown config {args.only!r}; choose from {labels}")
        selected = [c for c in selected if c.label == args.only]

    raw, adj, districts = load_dataset(NPY, ADJ)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"data {raw.shape} | {len(districts)} districts | device {device}", flush=True)

    all_rows: list[dict] = []
    started = time.time()

    for i, cfg in enumerate(selected, start=1):
        print(
            f"\n[{i}/{len(selected)}] {cfg.label}  "
            f"(adaptive={cfg.use_adaptive}, lambda={cfg.lambda_phys})",
            flush=True,
        )
        t0 = time.time()
        # persistence is identical across configs; emit it once, with dense_fixed --
        # keyed on the label rather than position so sharded runs stay consistent
        rows = rolling_origin(
            raw,
            adj,
            cfg,
            device=device,
            include_persistence=(cfg.label == "dense_fixed"),
            verbose=True,
        )
        for r in rows:
            r["config"] = cfg.label
            r["use_adaptive"] = cfg.use_adaptive
            r["hidden"] = cfg.hidden
            r["epochs"] = cfg.epochs
        all_rows.extend(rows)
        print(f"    {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"\nwrote {len(all_rows)} rows to {os.path.relpath(out_path, REPO)} "
        f"in {time.time() - started:.0f}s total",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
