"""Physics-informed arms on the reproduced architectures, under the frozen protocol.

Three arms per architecture -- ``base``, ``penalty`` (soft renewal term at a given
lambda) and ``decoder`` (renewal reconstruction) -- over 3 origins x 3 seeds, scored
with and without the week-395 artifact windows.

The same file drives the local smoke test and the Kaggle kernel, so what runs at
scale is what was checked here.

    C:/Users/tharu/rp/v/Scripts/python.exe analysis/_build/run_physics.py --quick
    C:/Users/tharu/rp/v/Scripts/python.exe analysis/_build/run_physics.py

Writes ``analysis/results/physics_<ARCH>.json`` in the same record schema the
sweep kernels use, so ``merge_sweep.py``-style joins work on it.
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

import adaptive as base  # noqa: E402
import improved as imp  # noqa: E402
import physics as phys  # noqa: E402
import renewal  # noqa: E402
import reproduced as arch  # noqa: E402
from dengue_gnn import seir  # noqa: E402

NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
OUT_DIR = REPO / "analysis" / "results"

WINDOW, HORIZON = 3, 3

#: Weights for the soft penalty. 0.0 is `base` and is run as its own arm, so the
#: sweep contains its own control rather than borrowing one from another job.
LAMBDAS = (0.1, 0.3, 1.0)


def build_arms(lambdas=LAMBDAS) -> list[tuple[str, str, float]]:
    """``(label, mode, lambda)`` for every arm, in reporting order."""
    arms = [("base", "base", 0.0)]
    arms += [(f"penalty_{lam:g}", "penalty", lam) for lam in lambdas]
    arms += [("decoder", "decoder", 0.0)]
    return arms


def run(arch_name: str, mode: str, lam: float, fold, hist, edge_index, w,
        seed: int, epochs: int, patience: int = 30, batch_size: int = 32) -> dict:
    """Train one (architecture, arm) on one fold; return held-out pooled scores.

    Model selection is on validation RMSE in raw counts -- the same criterion as
    every other row, and deliberately *not* the training objective, so the
    penalty arms are selected on forecast accuracy rather than on how well they
    satisfy their own constraint.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    kwargs = {"adaptive": False, "channels": 8} if arch_name == "AAGCN" else {}
    backbone = arch.build(arch_name, 25, WINDOW, HORIZON, edge_index=edge_index, **kwargs)
    net = phys.PhysicsNet(backbone, edge_index, w, mode, fold.mean, fold.std)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    data_loss = nn.MSELoss()

    def forward(split: str, idx=None):
        x = getattr(fold, f"x_{split}")
        p = getattr(fold, f"p_{split}")
        h = hist[split]
        if idx is not None:
            x, p, h = x[idx], p[idx], h[idx]
        return net(x, p, h), h

    n = len(fold.x_train)
    best, best_val, waited = None, float("inf"), 0
    for _ in range(epochs):
        net.train()
        order = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            opt.zero_grad()
            (z, counts), h = forward("train", idx)
            loss = data_loss(z, fold.y_train[idx]) + lam * net.physics_loss(counts, h)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

        net.eval()
        with torch.no_grad():
            (_, counts), _ = forward("val")
            val = base.rmse(counts.numpy(), fold.inverse(fold.y_val.numpy()))
        if val < best_val - 1e-6:
            best_val, waited = val, 0
            best = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    if best:
        net.load_state_dict(best)
    net.eval()
    with torch.no_grad():
        (_, counts), _ = forward("test")
    pred = counts.numpy()
    truth = fold.inverse(fold.y_test.numpy())
    scores = base.pooled_scores(pred, truth, imp.artifact_windows(fold.test_index,
                                                                  WINDOW, HORIZON))
    # The measurement that motivated all of this: does the arm still under-react?
    last = hist["test"][..., -1].numpy()
    growth = np.diff(np.log1p(np.maximum(
        np.concatenate([last[:, :, None], pred], axis=2), 0.0)), axis=2)
    scores["growth_p99"] = float(np.percentile(np.abs(growth), 99))
    scores["growth_max"] = float(np.abs(growth).max())
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", nargs="*", default=list(arch.ARCHITECTURES))
    ap.add_argument("--quick", action="store_true", help="1 seed, 20 epochs, one lambda")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    epochs = 20 if args.quick else args.epochs
    seeds = (0,) if args.quick else tuple(range(args.seeds))
    arms = build_arms((0.3,) if args.quick else LAMBDAS)

    cases, adjacency, _ = base.load_dataset(NPY, ADJ)
    src, dst = np.nonzero(adjacency)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    folds = base.build_folds(cases, WINDOW, HORIZON)

    w_np = renewal.generation_interval(seir.PAPER_SECTION4)
    w = torch.tensor(w_np, dtype=torch.float32)
    print(f"generation interval (weeks 1..{len(w_np)}): "
          + " ".join(f"{v:.3f}" for v in w_np)
          + f"   mean {(np.arange(1, len(w_np) + 1) * w_np).sum():.2f}\n", flush=True)

    histories = {
        f.origin: {s: phys.window_history(cases, getattr(f, f"{s}_index"))
                   for s in ("train", "val", "test")}
        for f in folds
    }

    for arch_name in args.arch:
        records = [dict(arch="persistence", increment="base", origin=f.origin, seed=-1,
                        **base.persistence_scores(f, cases, HORIZON,
                                                  imp.artifact_windows(f.test_index,
                                                                       WINDOW, HORIZON)))
                   for f in folds]
        out = OUT_DIR / f"physics_{arch_name}.json"
        started = time.time()
        for (label, mode, lam), fold, seed in itertools.product(arms, folds, seeds):
            t0 = time.time()
            try:
                sc = run(arch_name, mode, lam, fold, histories[fold.origin],
                         edge_index, w, seed, epochs)
            except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
                print(f"{arch_name:8s} {label:12s} o{fold.origin} s{seed} FAILED "
                      f"{type(exc).__name__}: {str(exc)[:70]}", flush=True)
                continue
            records.append(dict(arch=arch_name, increment=label, origin=fold.origin,
                                seed=seed, **sc))
            print(f"{arch_name:8s} {label:12s} o{fold.origin} s{seed} "
                  f"RMSE {sc['RMSE']:7.2f}  clean {sc['RMSE_clean']:6.2f}  "
                  f"gmax {sc['growth_max']:5.2f}  ({time.time()-t0:.0f}s)", flush=True)
            out.write_text(json.dumps(records, indent=1), encoding="utf-8")
        print(f"{arch_name}: {time.time()-started:.0f}s -> {out}\n", flush=True)

        floor = np.mean([r["RMSE_clean"] for r in records if r["arch"] == "persistence"])
        print(f"{'arm':14s}{'RMSE':>9s}{'clean':>9s}{'gmax':>7s}   floor {floor:.2f}")
        for label, _, _ in arms:
            # Persistence rows also carry increment="base"; including them would
            # average the floor into the base arm's score.
            rows = [r for r in records
                    if r["increment"] == label and r["arch"] == arch_name]
            assert all("growth_max" in r for r in rows), "model rows must carry growth"
            if rows:
                print(f"{label:14s}{np.mean([r['RMSE'] for r in rows]):9.2f}"
                      f"{np.mean([r['RMSE_clean'] for r in rows]):9.2f}"
                      f"{np.mean([r['growth_max'] for r in rows]):7.2f}")
        print(flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
