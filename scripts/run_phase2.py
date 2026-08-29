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

Persistence is emitted alongside every fold and is the bar that matters.

Usage:
    python scripts/run_phase2.py [--quick]
"""

from __future__ import annotations

import argparse
import csv
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
    args = ap.parse_args()

    raw, adj, districts = load_dataset(NPY, ADJ)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"data {raw.shape} | {len(districts)} districts | device {device}", flush=True)

    all_rows: list[dict] = []
    started = time.time()

    for i, cfg in enumerate(configs(args.quick), start=1):
        print(
            f"\n[{i}/5] {cfg.label}  (adaptive={cfg.use_adaptive}, lambda={cfg.lambda_phys})",
            flush=True,
        )
        t0 = time.time()
        # persistence is identical across configs; emit it once, with the first
        rows = rolling_origin(
            raw, adj, cfg, device=device, include_persistence=(i == 1), verbose=True
        )
        for r in rows:
            r["config"] = cfg.label
            r["use_adaptive"] = cfg.use_adaptive
            r["hidden"] = cfg.hidden
            r["epochs"] = cfg.epochs
        all_rows.extend(rows)
        print(f"    {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0].keys())
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"\nwrote {len(all_rows)} rows to {OUT.relative_to(REPO)} "
        f"in {time.time() - started:.0f}s total",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
