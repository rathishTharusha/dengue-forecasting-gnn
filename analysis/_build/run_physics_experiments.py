"""Run experiments evaluating the biological envelope and spatial physics loss.

Evaluates arms:
- base: unconstrained GNN
- envelope: biological growth ceiling (r_max = 2.3884) & host clearance bounds
- spatial: district-normalized spatial flux smoothness
- composite: envelope + spatial + conservation
- outbreak_aware: composite + asymmetric outbreak weighting

Usage:
    python analysis/_build/run_physics_experiments.py --quick
    python analysis/_build/run_physics_experiments.py --arch STGAT AAGCN
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "src"))

import adaptive as base
import improved as imp
import physics as phys
import physics_loss as ploss
import physics_net as pnet
import reproduced as arch

NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
OUT_DIR = REPO / "analysis" / "results"

WINDOW, HORIZON = 3, 3


def run_fold(
    arch_name: str,
    mode: str,
    fold,
    cases: np.ndarray,
    hist,
    edge_index: torch.Tensor,
    adj_dense: torch.Tensor,
    district_scales: torch.Tensor,
    seed: int,
    epochs: int,
    patience: int = 30,
    batch_size: int = 32,
) -> dict:
    """Train one architecture with specified physics arm on one fold."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    kwargs = {"adaptive": False, "channels": 8} if arch_name == "AAGCN" else {}
    backbone = arch.build(arch_name, 25, WINDOW, HORIZON, edge_index=edge_index, **kwargs)
    net = pnet.RelaxedPhysicsNet(
        backbone=backbone,
        edge_index=edge_index,
        adj_dense=adj_dense,
        district_scales=district_scales,
        mode=mode,
        mean=fold.mean,
        std=fold.std,
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)

    def get_data(split: str, idx=None):
        x = getattr(fold, f"x_{split}")
        p = getattr(fold, f"p_{split}")
        y = getattr(fold, f"y_{split}")
        h = hist[split]
        if idx is not None:
            x, p, y, h = x[idx], p[idx], y[idx], h[idx]
        return x, p, y, h

    n = len(fold.x_train)
    best_weights = None
    best_val = float("inf")
    waited = 0

    for epoch in range(epochs):
        net.train()
        order = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            x, p, y, h = get_data("train", idx)
            counts_true = torch.tensor(fold.inverse(y.numpy()), dtype=torch.float32)

            opt.zero_grad()
            z_pred, counts_pred = net(x, p)
            loss, _ = net.compute_loss(z_pred, counts_pred, y, counts_true, h)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

        # Validation
        net.eval()
        with torch.no_grad():
            x_val, p_val, _, _ = get_data("val")
            _, val_counts = net(x_val, p_val)
            val_rmse = base.rmse(val_counts.numpy(), fold.inverse(fold.y_val.numpy()))

        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            waited = 0
            best_weights = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    if best_weights is not None:
        net.load_state_dict(best_weights)

    # Test evaluation
    net.eval()
    with torch.no_grad():
        x_test, p_test, _, _ = get_data("test")
        _, test_counts = net(x_test, p_test)

    pred = test_counts.numpy()
    truth = fold.inverse(fold.y_test.numpy())
    artifact_idx = imp.artifact_windows(fold.test_index, WINDOW, HORIZON)
    scores = base.pooled_scores(pred, truth, artifact_idx)

    # Calculate growth statistics
    last_obs = cases[fold.test_index - 1]  # (n_windows, 25)
    log_p = np.log1p(np.clip(pred, 0.0, None))
    log_h = np.log1p(np.clip(last_obs[:, :, None], 0.0, None))
    log_traj = np.concatenate([log_h, log_p], axis=-1)
    dlog = np.diff(log_traj, axis=-1)

    scores["growth_max"] = float(np.max(dlog))
    scores["growth_p99"] = float(np.percentile(dlog, 99))
    scores["growth_min"] = float(np.min(dlog))

    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", nargs="*", default=["STGAT", "AAGCN"])
    ap.add_argument("--arms", nargs="*", default=None, help="Subset of arms to run")
    ap.add_argument("--quick", action="store_true", help="Quick smoke test (1 seed, 25 epochs, 2 arms)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory for result files")
    args = ap.parse_args()

    epochs = 25 if args.quick else args.epochs
    seeds = (0,) if args.quick else tuple(range(args.seeds))
    if args.arms:
        arms = args.arms
    else:
        arms = ["base", "composite"] if args.quick else list(pnet.PHYSICS_ARMS)

    cases, adjacency, _ = base.load_dataset(NPY, ADJ)
    src, dst = np.nonzero(adjacency)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    adj_dense = torch.tensor(adjacency, dtype=torch.float32)
    folds = base.build_folds(cases, WINDOW, HORIZON)

    histories = {
        f.origin: {s: phys.window_history(cases, getattr(f, f"{s}_index")) for s in ("train", "val", "test")}
        for f in folds
    }

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for arch_name in args.arch:
        print(f"\n{'='*30} Running {arch_name} {'='*30}")
        records = [
            dict(
                arch="persistence",
                increment="base",
                origin=f.origin,
                seed=-1,
                **base.persistence_scores(f, cases, HORIZON, imp.artifact_windows(f.test_index, WINDOW, HORIZON)),
            )
            for f in folds
        ]

        out = out_dir / f"physics_envelope_{arch_name}.json"
        started = time.time()

        for arm, fold, seed in itertools.product(arms, folds, seeds):
            t0 = time.time()
            district_scales = torch.tensor(cases[fold.train_index].mean(axis=0), dtype=torch.float32)
            try:
                sc = run_fold(
                    arch_name=arch_name,
                    mode=arm,
                    fold=fold,
                    cases=cases,
                    hist=histories[fold.origin],
                    edge_index=edge_index,
                    adj_dense=adj_dense,
                    district_scales=district_scales,
                    seed=seed,
                    epochs=epochs,
                )
            except Exception as exc:
                print(f"{arch_name:8s} {arm:14s} o{fold.origin} s{seed} FAILED {type(exc).__name__}: {exc}", flush=True)
                continue

            records.append(dict(arch=arch_name, increment=arm, origin=fold.origin, seed=seed, **sc))
            print(
                f"{arch_name:8s} {arm:14s} o{fold.origin} s{seed} "
                f"RMSE {sc['RMSE']:7.2f}  clean {sc['RMSE_clean']:6.2f}  "
                f"gmax {sc['growth_max']:5.2f}  ({time.time()-t0:.0f}s)",
                flush=True,
            )
            out.write_text(json.dumps(records, indent=1), encoding="utf-8")
            try:
                import pandas as pd
                pd.DataFrame(records).to_csv(out.with_suffix(".csv"), index=False)
            except Exception:
                pass

        print(f"\n{arch_name} complete in {time.time()-started:.0f}s -> {out}")
        floor = np.mean([r["RMSE_clean"] for r in records if r["arch"] == "persistence"])
        print(f"{'Arm':16s}{'RMSE (all)':>12s}{'RMSE (clean)':>14s}{'gmax':>8s}   [Persistence Floor: {floor:.2f}]")
        for arm in arms:
            rows = [r for r in records if r["increment"] == arm and r["arch"] == arch_name]
            if rows:
                mean_rmse = np.mean([r["RMSE"] for r in rows])
                mean_clean = np.mean([r["RMSE_clean"] for r in rows])
                mean_gmax = np.mean([r["growth_max"] for r in rows])
                diff_floor = mean_clean - floor
                sign = "+" if diff_floor > 0 else ""
                print(f"{arm:16s}{mean_rmse:12.2f}{mean_clean:14.2f} ({sign}{diff_floor:.2f}){mean_gmax:8.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
