"""Does the model respond to an outbreak it can already see?

The distinction this draws
--------------------------
EXP-024 closed off *anticipating* outbreaks: `R_t` is 26% predictable, climate is
inert, susceptible depletion is 500x too slow. But anticipation and **response**
are different failures, and only the first has been ruled out.

Response means: by the time an outbreak is already visible in the input window --
recent cases elevated, back-solved `R_hat` clearly above 1 -- does the forecast
follow it? If the model under-predicts even then, there is headroom that requires
no prediction at all, only that the model stop discounting evidence it already
holds.

Three measurements, each of which decides a piece of the loss design:

1. **Damping.** Regress the forecast's implied log-growth on the truth's. A slope
   far below 1 means the model is shrinking every signal toward zero -- the
   signature of MSE over a mostly-quiet target. The slope *is* the under-reaction
   factor, stated as one number.
2. **Conditional response.** Split test windows by the `R_hat` visible at forecast
   time. If bias stays negative on windows where `R_hat > 1.5`, the model is
   ignoring evidence in its own input.
3. **Where smoothness belongs.** The retired `losses.smoothness_loss` penalised
   differences in predicted *cases* between neighbouring districts. Case counts
   differ by orders of magnitude between Colombo and rural districts, so that term
   penalises geography. `R` is scale-free and therefore comparable across
   districts. Moran's I on both says whether a Laplacian penalty belongs on `R`
   rather than on counts.

Run with the pinned stack::

    C:/Users/tharu/rp/v/Scripts/python.exe analysis/_build/response_diagnosis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "src"))

import adaptive as base  # noqa: E402
import improved as imp  # noqa: E402
import renewal  # noqa: E402
import reproduced as arch  # noqa: E402

from dengue_gnn import seir  # noqa: E402

NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
OUT = REPO / "analysis" / "results" / "response_diagnosis.json"


def train(name, fold, edge_index, seed, epochs):
    """Train one arm, return held-out predictions as raw counts."""
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    model = arch.build(name, 25, 3, 3, edge_index=edge_index)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    loss_fn = nn.MSELoss()

    def predict(split):
        return model(getattr(fold, f"x_{split}"), edge_index) + getattr(fold, f"p_{split}")

    n = len(fold.x_train)
    best, best_val, waited = None, float("inf"), 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n)
        for s in range(0, n, 32):
            i = order[s : s + 32]
            opt.zero_grad()
            loss_fn(model(fold.x_train[i], edge_index),
                    fold.y_train[i] - fold.p_train[i]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = base.rmse(fold.inverse(predict("val").numpy()),
                          fold.inverse(fold.y_val.numpy()))
        if v < best_val - 1e-6:
            best_val, waited, best = v, 0, {k: t.clone() for k, t in model.state_dict().items()}
        else:
            waited += 1
            if waited >= 30:
                break
    if best:
        model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        return fold.inverse(predict("test").numpy())


def morans_i(values: np.ndarray, adjacency: np.ndarray) -> float:
    """Spatial autocorrelation over districts, averaged across weeks.

    Args:
        values: ``(weeks, districts)``; NaNs are dropped week by week.
        adjacency: Symmetric binary neighbour matrix, diagonal ignored.

    Returns:
        Mean Moran's I. 0 is spatial randomness, 1 is perfect clustering.
    """
    a = adjacency.copy()
    np.fill_diagonal(a, 0.0)
    out = []
    for row in values:
        ok = np.isfinite(row)
        if ok.sum() < 5:
            continue
        x = row[ok] - row[ok].mean()
        sub = a[np.ix_(ok, ok)]
        denom = (x ** 2).sum() * sub.sum()
        if denom <= 0:
            continue
        out.append(len(x) * (x @ sub @ x) / denom)
    return float(np.mean(out))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cases, adjacency, _ = base.load_dataset(NPY, ADJ)
    src, dst = np.nonzero(adjacency)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    folds = base.build_folds(cases)
    w = renewal.generation_interval(seir.PAPER_SECTION4)
    K = len(w)

    pred_all, true_all, last_all, rhat_all = [], [], [], []
    for fold in folds:
        keep = ~imp.artifact_windows(fold.test_index, 3, 3)
        idx = fold.test_index[keep]
        truth = fold.inverse(fold.y_test.numpy())[keep]
        last = cases[idx - 1]
        force = np.stack([sum(w[lag] * cases[i - 1 - lag] for lag in range(K)) for i in idx])
        rhat = last / np.maximum(force, 1.0)
        for seed in (0, 1):
            pred_all.append(train("A3TGCN", fold, edge_index, seed, 150)[keep])
            true_all.append(truth)
            last_all.append(last)
            rhat_all.append(rhat)
        print(f"origin {fold.origin} done", flush=True)

    pred = np.concatenate(pred_all)
    truth = np.concatenate(true_all)
    last = np.concatenate(last_all)
    rhat = np.concatenate(rhat_all)

    def growth(arr):
        seq = np.concatenate([last[:, :, None], arr], axis=2)
        return np.diff(np.log1p(np.maximum(seq, 0.0)), axis=2)

    gp, gt = growth(pred).ravel(), growth(truth).ravel()
    slope = float(np.polyfit(gt, gp, 1)[0])
    log_r = np.log(np.maximum(rhat, 1e-3))
    slope_r = float(np.polyfit(np.repeat(log_r[:, :, None], 3, axis=2).ravel(), gp, 1)[0])

    report = {"damping_slope_vs_truth": slope, "damping_slope_vs_log_rhat": slope_r}
    print("\n=== 1. Damping ===")
    print(f"  slope of predicted log-growth on TRUE log-growth : {slope:.3f}")
    print(f"  slope of predicted log-growth on log R_hat       : {slope_r:.3f}")
    print("  1.0 would be an undamped response; 0.0 would be a flat forecast.")

    print("\n=== 2. Response, conditional on what the model can already see ===")
    print(f"{'R_hat at forecast time':>26s}{'n':>9s}{'bias':>9s}{'RMSE':>9s}{'true growth':>13s}")
    bands = [("R_hat < 0.8", rhat < 0.8), ("0.8-1.2", (rhat >= 0.8) & (rhat < 1.2)),
             ("1.2-1.5", (rhat >= 1.2) & (rhat < 1.5)), ("1.5-2.5", (rhat >= 1.5) & (rhat < 2.5)),
             ("R_hat >= 2.5", rhat >= 2.5)]
    report["bands"] = {}
    for name, mask in bands:
        if mask.sum() == 0:
            continue
        m3 = np.repeat(mask[:, :, None], 3, axis=2)
        bias = float((pred - truth)[m3].mean())
        rmse = float(np.sqrt(((pred - truth)[m3] ** 2).mean()))
        tg = float(growth(truth)[m3].mean())
        report["bands"][name] = {"n": int(mask.sum()), "bias": bias, "rmse": rmse,
                                 "true_growth": tg}
        print(f"{name:>26s}{int(mask.sum()):9d}{bias:9.2f}{rmse:9.2f}{tg:13.3f}")
    print("  A negative bias in the high-R_hat rows means the model is discounting")
    print("  evidence already present in its own input -- a response failure, not a")
    print("  prediction failure, and fixable without predicting anything.")

    print("\n=== 3. Where does a smoothness penalty belong? ===")
    weeks = cases.shape[0]
    force_full = np.zeros_like(cases)
    for lag, weight in enumerate(w, start=1):
        force_full[lag:] += weight * cases[: weeks - lag]
    r_full = np.where(force_full >= 5.0, cases / np.maximum(force_full, 1e-9), np.nan)
    binary = (adjacency > 0).astype(float)
    i_cases = morans_i(np.log1p(cases), binary)
    i_r = morans_i(np.log(np.maximum(r_full, 1e-3)), binary)
    report["morans_i_log_cases"] = i_cases
    report["morans_i_log_r"] = i_r
    print(f"  Moran's I, log1p(cases) : {i_cases:.3f}")
    print(f"  Moran's I, log R        : {i_r:.3f}")
    print("  The retired smoothness_loss penalised neighbouring differences in")
    print("  predicted COUNTS. Counts differ by orders of magnitude across districts,")
    print("  so that term penalised geography. R is scale-free and comparable.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
