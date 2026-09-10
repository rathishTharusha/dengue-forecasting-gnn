"""Run the five reproduced architectures under this project's frozen protocol.

This establishes the baseline the ablation table should be measured against:
architectures that are *verified* (``reproduction/`` recovers their published
numbers), evaluated the way this project evaluates everything else.

Also runs the AAGCN adaptive-graph arm, which their repository could not reach --
``out_channels=1`` makes PGT's adaptive branch build a zero-width convolution.
Both AAGCN arms carry the same projection head, so the only difference between
them is the flag.

Requires the pinned stack (torch 2.1.2 / PyG 2.4.0 / PGT 0.54.0):

    python reproduction/verify_local.py --env-only
    C:/Users/tharu/rp/v/Scripts/python.exe analysis/_build/run_reproduced_baseline.py

Writes ``analysis/results/reproduced_baseline.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
OUT = REPO / "analysis" / "results" / "reproduced_baseline.json"

#: name -> kwargs. AAGCN appears twice so the adaptive switch is a matched pair.
ARMS: dict[str, dict] = {
    "STGAT": {},
    "A3TGCN": {},
    "ASTGCN": {},
    "DCRNN": {},
    "AAGCN": {"adaptive": False, "channels": 8},
    "AAGCN+adaptive": {"adaptive": True, "channels": 8},
}


def train_and_score(name: str, fold, edge_index, artifact, seed: int, epochs: int,
                    lr: float = 1e-3, weight_decay: float = 5e-4,
                    patience: int = 30, batch_size: int = 32) -> dict:
    """Train one architecture on one fold; return held-out pooled scores.

    Predicts the residual over persistence in log1p space, matching
    ``docs/decisions/0001`` and every other row of the ablation table. Model
    selection is on a genuine validation split -- the authors' loop has none.

    Scores are reported twice, with and without the test windows touching the
    week-395 reporting artifact: on the origin-0.85 fold those six windows carry
    ~90% of the squared error, so a single pooled number there is mostly a
    measurement of a data-entry backlog.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    kwargs = dict(ARMS[name])
    model = arch.build(name.split("+")[0], fold.x_train.shape[1],
                       fold.x_train.shape[2], fold.y_train.shape[2],
                       edge_index=edge_index, **kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def predict(split: str) -> torch.Tensor:
        x = getattr(fold, f"x_{split}")
        return model(arch.input_adapter(x), edge_index) + getattr(fold, f"p_{split}")

    n = fold.x_train.shape[0]
    best_state, best_val, waited = None, float("inf"), 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            opt.zero_grad()
            out = model(arch.input_adapter(fold.x_train[idx]), edge_index)
            loss = loss_fn(out, fold.y_train[idx] - fold.p_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val = base.rmse(fold.inverse(predict("val").numpy()),
                            fold.inverse(fold.y_val.numpy()))
        if val < best_val - 1e-6:
            best_val, waited = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = fold.inverse(predict("test").numpy())
    truth = fold.inverse(fold.y_test.numpy())
    return base.pooled_scores(pred, truth, artifact)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="1 seed, 20 epochs")
    ap.add_argument("--only", nargs="*", help="subset of arms to run")
    args = ap.parse_args()

    seeds = (0,) if args.quick else (0, 1, 2)
    epochs = 20 if args.quick else 150
    arms = args.only or list(ARMS)

    cases, adjacency, _ = base.load_dataset(NPY, ADJ)
    src, dst = np.nonzero(adjacency)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    folds = base.build_folds(cases)

    artifact = {f.origin: imp.artifact_windows(f.test_index, 3, 3) for f in folds}
    for fold in folds:
        print(f"origin {fold.origin}: {int(artifact[fold.origin].sum())}/"
              f"{len(fold.test_index)} test windows touch week {imp.ARTIFACT_WEEK}")

    records = []
    for fold in folds:
        records.append({"arm": "persistence", "origin": fold.origin, "seed": -1,
                        **base.persistence_scores(fold, cases, 3, artifact[fold.origin])})

    started = time.time()
    for name in arms:
        for fold in folds:
            for seed in seeds:
                t0 = time.time()
                try:
                    scores = train_and_score(name, fold, edge_index,
                                             artifact[fold.origin], seed, epochs)
                except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
                    print(f"{name:15s} origin {fold.origin} seed {seed}  FAILED "
                          f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
                    continue
                records.append({"arm": name, "origin": fold.origin, "seed": seed, **scores})
                print(f"{name:15s} origin {fold.origin} seed {seed}  "
                      f"RMSE {scores['RMSE']:7.2f}  clean {scores['RMSE_clean']:6.2f}  "
                      f"MAE {scores['MAE']:6.2f}  ({time.time() - t0:.0f}s)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"\n{time.time() - started:.0f}s -> {OUT}")

    print(f"\n{'arm':16s}{'RMSE':>9s}{'sd':>7s}{'MAE':>9s}{'n':>4s}")
    for name in ["persistence", *arms]:
        rows = [r for r in records if r["arm"] == name]
        if not rows:
            continue
        rmse = np.array([r["RMSE"] for r in rows])
        mae = np.array([r["MAE"] for r in rows])
        print(f"{name:16s}{rmse.mean():9.2f}{rmse.std():7.2f}{mae.mean():9.2f}{len(rows):4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
