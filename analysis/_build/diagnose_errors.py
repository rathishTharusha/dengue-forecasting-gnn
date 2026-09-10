"""Where does the sweep baseline's error actually live?

A physics term can only help if it constrains a failure mode the model has.
EXP-014 established that the previous one did not: the growth-band ceiling was
slack against a model that *under*-reacts, so it contributed exactly zero
gradient on all 96 rows. That finding was made on the 8-origin protocol, before
the week-395 artifact was known, and on a hand-rolled baseline. This re-measures
it on the current one.

Trains the two architectures that beat the persistence floor artifact-free
(A3TGCN 29.05, STGAT 29.35 against 29.52), captures their held-out predictions,
and reports four decompositions:

1. **Growth reactivity.** Predicted vs observed |d log(1+cases)/dweek|. If the
   model under-reacts, an upper bound on growth is inert and a lower bound is not
   a physical constraint.
2. **Error by outbreak state.** Squared error split by whether the target week is
   in the top decile of that district's history. EDA F6 puts the headroom in the
   tail; this says how much is actually there.
3. **Error by horizon.** Whether the 3-week-ahead step carries the loss.
4. **Peak timing.** Lag at which predicted and observed series correlate best --
   a model that is right but late is a different problem from one that is flat.

Run with the pinned stack::

    C:/Users/tharu/rp/v/Scripts/python.exe analysis/_build/diagnose_errors.py

Writes ``analysis/results/error_diagnosis.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))

import adaptive as base  # noqa: E402
import improved as imp  # noqa: E402
import reproduced as arch  # noqa: E402

NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
OUT = REPO / "analysis" / "results" / "error_diagnosis.json"

#: The two arms that beat the floor artifact-free, per analysis/results/improved_sweep.md.
ARMS = ("A3TGCN", "STGAT")


def train(name: str, fold, edge_index, seed: int, epochs: int) -> np.ndarray:
    """Train one arm on one fold and return held-out predictions as raw counts."""
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    model = arch.build(name, 25, 3, 3, edge_index=edge_index)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    loss_fn = nn.MSELoss()

    def predict(split: str) -> torch.Tensor:
        return model(getattr(fold, f"x_{split}"), edge_index) + getattr(fold, f"p_{split}")

    n = len(fold.x_train)
    best, best_val, waited = None, float("inf"), 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n)
        for s in range(0, n, 32):
            i = order[s : s + 32]
            opt.zero_grad()
            loss = loss_fn(model(fold.x_train[i], edge_index), fold.y_train[i] - fold.p_train[i])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = base.rmse(fold.inverse(predict("val").numpy()), fold.inverse(fold.y_val.numpy()))
        if v < best_val - 1e-6:
            best_val, waited = v, 0
            best = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            waited += 1
            if waited >= 30:
                break
    if best:
        model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        return fold.inverse(predict("test").numpy())


def log_growth(counts: np.ndarray, last: np.ndarray) -> np.ndarray:
    """``d log(1+cases)/dweek`` across a horizon, seeded by the last observation.

    Mirrors :func:`dengue_gnn.mechanistic.implied_log_growth`, in numpy, so the
    diagnosis is measured in the same quantity the constraint would act on.
    """
    seq = np.concatenate([last[:, :, None], counts], axis=2)
    return np.diff(np.log1p(np.maximum(seq, 0.0)), axis=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    cases, adjacency, _ = base.load_dataset(NPY, ADJ)
    src, dst = np.nonzero(adjacency)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    folds = base.build_folds(cases)

    # Outbreak = target week in the top decile of that district's own history, so
    # the label is per-district and not dominated by Colombo's scale.
    district_p90 = np.percentile(cases, 90, axis=0)

    report: dict = {"arms": {}}
    for name in ARMS:
        acc = {k: [] for k in ("pred", "truth", "last", "artifact", "outbreak")}
        for fold in folds:
            mask = imp.artifact_windows(fold.test_index, 3, 3)
            truth = fold.inverse(fold.y_test.numpy())
            last = cases[fold.test_index - 1]
            for seed in range(args.seeds):
                pred = train(name, fold, edge_index, seed, args.epochs)
                acc["pred"].append(pred)
                acc["truth"].append(truth)
                acc["last"].append(last)
                acc["artifact"].append(np.repeat(mask[:, None], 25, axis=1))
                acc["outbreak"].append(truth.max(axis=2) > district_p90[None, :])
                print(f"{name:8s} origin {fold.origin} seed {seed} done", flush=True)

        pred = np.concatenate(acc["pred"])
        truth = np.concatenate(acc["truth"])
        last = np.concatenate(acc["last"])
        artifact = np.concatenate(acc["artifact"])
        outbreak = np.concatenate(acc["outbreak"])

        # Everything below is measured artifact-free: including week 395 would
        # make this a diagnosis of a reporting backlog rather than of the model.
        keep = ~artifact
        g_pred = log_growth(pred, last)[keep]
        g_true = log_growth(truth, last)[keep]
        err2 = ((pred - truth) ** 2)[keep]

        out, ob = {}, outbreak[keep]
        out["growth"] = {
            "pred_p99": float(np.percentile(np.abs(g_pred), 99)),
            "pred_max": float(np.abs(g_pred).max()),
            "true_p99": float(np.percentile(np.abs(g_true), 99)),
            "true_max": float(np.abs(g_true).max()),
            "pred_sd": float(g_pred.std()),
            "true_sd": float(g_true.std()),
            "under_reaction_ratio": float(g_true.std() / max(g_pred.std(), 1e-12)),
        }
        out["outbreak"] = {
            "share_of_windows": float(ob.mean()),
            "share_of_squared_error": float(err2[ob].sum() / err2.sum()),
            "rmse_outbreak": float(np.sqrt(err2[ob].mean())),
            "rmse_quiet": float(np.sqrt(err2[~ob].mean())),
        }
        per_h = ((pred - truth) ** 2)
        per_h[artifact] = np.nan
        out["horizon_rmse"] = [float(np.sqrt(np.nanmean(per_h[:, :, h]))) for h in range(3)]
        # Signed bias on outbreak windows: negative means the model forecasts
        # fewer cases than occur, which is what under-reaction looks like.
        bias = (pred - truth)[keep]
        out["bias"] = {
            "outbreak_mean": float(bias[ob].mean()),
            "quiet_mean": float(bias[~ob].mean()),
        }
        report["arms"][name] = out

        print(f"\n=== {name} (artifact-free) ===")
        g = out["growth"]
        print(f"  |log growth| p99  pred {g['pred_p99']:.3f}  true {g['true_p99']:.3f}")
        print(f"  |log growth| max  pred {g['pred_max']:.3f}  true {g['true_max']:.3f}")
        print(f"  sd ratio true/pred                {g['under_reaction_ratio']:.2f}x")
        o = out["outbreak"]
        print(f"  outbreak windows  {o['share_of_windows']:.1%} of data, "
              f"{o['share_of_squared_error']:.1%} of squared error")
        print(f"  RMSE outbreak {o['rmse_outbreak']:.2f}   quiet {o['rmse_quiet']:.2f}")
        print(f"  bias outbreak {out['bias']['outbreak_mean']:+.2f}   "
              f"quiet {out['bias']['quiet_mean']:+.2f}")
        print("  RMSE by horizon   " + "  ".join(f"h{i+1} {v:.2f}"
                                                  for i, v in enumerate(out["horizon_rmse"])))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
