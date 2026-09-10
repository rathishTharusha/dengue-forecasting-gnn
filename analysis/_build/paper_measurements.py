"""Every number the paper quotes that no other results file already holds.

Most of the paper's figures read from the JSON that the experiment which produced
them wrote: ``improved_sweep.json``, ``error_diagnosis.json``,
``r_predictability.json`` and so on. A handful of quantities were originally
measured in throwaway scripts during EXP-023 – EXP-025, which is fine for deciding
what to build next and not fine for a paper. This script recomputes them properly
and saves them, so every figure and every table in the write-up traces to a file
that can be regenerated.

Produces ``analysis/results/paper_measurements.json`` with:

``anchors``
    RMSE of each causally available renewal anchor against persistence, by origin.
    The evidence that the physics loses as a *forecasting* mechanism.
``lead_lag``
    Correlation of ``log R_hat`` with past and future growth. The single number
    that explains why a renewal penalty pushes a forecast the wrong way.
``variance``
    Decomposition of ``log R`` into a common weekly factor, a district effect and
    a residual, plus the autocorrelation of each.
``episodes``
    Outbreak episode counts and durations, which bound anything generative.
``detection``
    Incremental AUC of adding the mechanistic quantity and the graph to a trivial
    level baseline, logistic regression fitted on training weeks only.

Run with the pinned stack::

    C:/Users/tharu/rp/v/Scripts/python.exe analysis/_build/paper_measurements.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "analysis" / "_build"))

import adaptive as base  # noqa: E402
import improved as imp  # noqa: E402
import physics as phys  # noqa: E402
import renewal  # noqa: E402
import torch  # noqa: E402
from outbreak_signal import auc_roc, average_precision  # noqa: E402

from dengue_gnn import seir  # noqa: E402

NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
OUT = REPO / "analysis" / "results" / "paper_measurements.json"

HORIZON = 3


def ratio_forecast(hist: torch.Tensor, w: torch.Tensor, horizon: int, damp: float) -> np.ndarray:
    """Renewal in *ratio* form: physics supplies momentum, persistence the level.

    ``pred_t = cases_{t-1} * (force_t / force_{t-1}) ** damp``. At ``damp=0`` this
    is exactly persistence, so the family contains the baseline it is competing
    with -- which is what makes the comparison meaningful rather than rhetorical.
    """
    kernel = w.shape[0]
    flipped = torch.flip(w, dims=(0,))
    h = hist.clone()
    out = []
    for _ in range(horizon):
        f_now = (h[..., -kernel:] * flipped).sum(-1)
        f_prev = (h[..., -kernel - 1 : -1] * flipped).sum(-1)
        ratio = (f_now / f_prev.clamp_min(1e-3)).clamp(0.2, 5.0)
        nxt = h[..., -1] * ratio.pow(damp)
        out.append(nxt)
        h = torch.cat([h, nxt.unsqueeze(-1)], dim=-1)
    return torch.stack(out, dim=-1).numpy()


def logistic(x: np.ndarray, y: np.ndarray, iters: int = 4000, lr: float = 0.5):
    """Standardised logistic regression by gradient descent.

    Standardisation is not cosmetic: unscaled columns of differing magnitude do
    not converge at a single step size, and the failure presents as a *below
    chance* AUC, which is impossible for a fitted model and therefore reads as a
    finding if you are not watching for it.
    """
    mu, sd = x.mean(0), x.std(0) + 1e-9
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    weights = np.zeros(z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ weights, -30, 30)))
        weights -= lr * (z.T @ (p - y)) / len(y)
    return weights, mu, sd


def apply_logistic(x: np.ndarray, fit) -> np.ndarray:
    weights, mu, sd = fit
    return np.column_stack([np.ones(len(x)), (x - mu) / sd]) @ weights


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raw = np.nan_to_num(np.load(NPY, allow_pickle=True)).astype(np.float64)
    cases, adjacency, _ = base.load_dataset(NPY, ADJ)
    w_np = renewal.generation_interval(seir.PAPER_SECTION4)
    w = torch.tensor(w_np, dtype=torch.float32)
    kernel = len(w_np)
    folds = base.build_folds(cases, 3, HORIZON)
    report: dict = {}

    # ---------------------------------------------------------------- anchors
    rows = {k: [] for k in ("persistence", "force", "rhat_force", "ratio_0.5", "ratio_1.0")}
    per_origin = []
    for fold in folds:
        mask = imp.artifact_windows(fold.test_index, 3, HORIZON)
        hist = phys.window_history(cases, fold.test_index)
        truth = fold.inverse(fold.y_test.numpy())
        seed_hist = hist[..., -kernel:]

        preds = {
            "force": renewal.renewal_forecast(torch.zeros(len(hist), 25, HORIZON),
                                              seed_hist, w).numpy(),
            "rhat_force": renewal.renewal_forecast(
                torch.log(renewal.estimate_r(hist, w).clamp_min(1e-3))
                .unsqueeze(-1).expand(-1, -1, HORIZON), seed_hist, w).numpy(),
            "ratio_0.5": ratio_forecast(hist, w, HORIZON, 0.5),
            "ratio_1.0": ratio_forecast(hist, w, HORIZON, 1.0),
        }
        entry = {"origin": fold.origin}
        floor = base.persistence_scores(fold, cases, HORIZON, mask)["RMSE_clean"]
        rows["persistence"].append(floor)
        entry["persistence"] = floor
        for name, pred in preds.items():
            value = base.pooled_scores(pred, truth, mask)["RMSE_clean"]
            rows[name].append(value)
            entry[name] = value
        per_origin.append(entry)
    report["anchors"] = {"per_origin": per_origin,
                         "mean": {k: float(np.mean(v)) for k, v in rows.items()}}

    print("Causally available renewal anchors, artifact-free RMSE")
    for name, value in report["anchors"]["mean"].items():
        print(f"  {name:14s}{value:8.2f}")

    # --------------------------------------------------------------- lead/lag
    weeks = cases.shape[0]
    force_full = np.zeros_like(cases)
    for lag, weight in enumerate(w_np, start=1):
        force_full[lag:] += weight * cases[: weeks - lag]
    # Same usable mask as mechanistic_r.py: enough recent transmission for the
    # ratio to mean anything, AND a non-zero numerator. Clipping a zero-case week
    # to log(1e-3) does not measure a low R, it measures the clip -- and it
    # inflates sd(log R) from 0.749 to 1.137, which then silently corrupts any
    # arithmetic pairing that sd with an r^2 computed on the other subset.
    usable = (force_full >= 5.0) & (cases > 0)
    log_r = np.full_like(force_full, np.nan)
    log_r[usable] = np.log(cases[usable] / force_full[usable])

    future = np.full_like(cases, np.nan)
    future[:-HORIZON] = np.log1p(cases[HORIZON:]) - np.log1p(cases[:-HORIZON])
    past = np.full_like(cases, np.nan)
    past[HORIZON:] = np.log1p(cases[HORIZON:]) - np.log1p(cases[:-HORIZON])

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1])

    report["lead_lag"] = {
        "log_r_vs_past_growth": corr(log_r, past),
        "log_r_vs_future_growth": corr(log_r, future),
        "log_r_vs_abs_future_growth": corr(log_r, np.abs(future)),
        "sd_log_r": float(np.nanstd(log_r)),
    }
    print("\nlog R_hat vs growth: past {log_r_vs_past_growth:+.3f}, "
          "future {log_r_vs_future_growth:+.3f}, "
          "|future| {log_r_vs_abs_future_growth:+.3f}".format(**report["lead_lag"]))

    # --------------------------------------------------------------- variance
    weekly = np.nanmean(log_r, axis=1)
    district = np.nanmean(log_r, axis=0)
    total = float(np.nanvar(log_r))
    ok = np.isfinite(weekly)
    pair = ok[:-1] & ok[1:]
    m = np.isfinite(log_r[:-1]) & np.isfinite(log_r[1:])
    report["variance"] = {
        "total": total,
        "weekly_factor_share": float(np.nanvar(weekly) / total),
        "district_effect_share": float(np.nanvar(district) / total),
        "weekly_factor_autocorr": float(np.corrcoef(weekly[:-1][pair], weekly[1:][pair])[0, 1]),
        "district_log_r_autocorr": float(np.corrcoef(log_r[:-1][m], log_r[1:][m])[0, 1]),
    }
    print("\nlog R variance: weekly factor {weekly_factor_share:.1%}, "
          "district {district_effect_share:.1%}, "
          "weekly autocorr {weekly_factor_autocorr:.3f}".format(**report["variance"]))

    # --------------------------------------------------------------- episodes
    thresh = np.percentile(cases, 90, axis=0)
    above = cases > thresh
    lengths = []
    for d in range(cases.shape[1]):
        run = 0
        for t in range(cases.shape[0]):
            if above[t, d]:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
    lengths = np.array(lengths)
    report["episodes"] = {
        "n_episodes": len(lengths),
        "median_weeks": float(np.median(lengths)),
        "mean_weeks": float(lengths.mean()),
        "max_weeks": int(lengths.max()),
        "at_least_3_weeks": int((lengths >= 3).sum()),
        "district_weeks_above": int(above.sum()),
        "share_above": float(above.mean()),
    }
    print("\noutbreak episodes: {n_episodes}, {at_least_3_weeks} lasting >=3 weeks, "
          "{share_above:.1%} of district-weeks above threshold".format(**report["episodes"]))

    # -------------------------------------------------------------- detection
    temp_idx = 0
    combos = {
        "level": ["level"],
        "level+growth": ["level", "growth"],
        "level+rhat": ["level", "rhat"],
        "level+neighbours": ["level", "neigh"],
        "level+rhat+neighbours": ["level", "rhat", "neigh"],
        "level+rhat+neighbours+climate": ["level", "rhat", "neigh", "temp"],
    }
    scores = {k: [] for k in combos}

    def features(idx, threshold):
        idx = np.asarray(idx)
        last = cases[idx - 1]
        # max(0, ...) or an early index wraps to an empty slice and yields NaN,
        # which then ranks arbitrarily and reads as a below-chance AUC.
        prev4 = np.stack([cases[max(0, i - 4) : i].mean(axis=0) for i in idx])
        force = np.stack([sum(w_np[lag] * cases[i - 1 - lag] for lag in range(kernel))
                          for i in idx])
        rel = last / np.maximum(threshold, 1.0)
        binary = (adjacency > 0).astype(float)
        neigh = (rel @ binary.T) / np.maximum(binary.sum(axis=1), 1)
        temp = np.stack([raw[i - 1, :, temp_idx] for i in idx])
        return {
            "level": np.log1p(rel),
            "growth": np.log1p(last) - np.log1p(prev4),
            "rhat": np.log(np.maximum(last / np.maximum(force, 1.0), 1e-3)),
            "neigh": np.log1p(neigh),
            "temp": (temp - temp.mean()) / (temp.std() + 1e-9),
        }

    for fold in folds:
        threshold = np.percentile(cases[: int(fold.train_index[-1])], 90, axis=0)

        def labelled(idx, threshold=threshold):
            return np.stack([(cases[i : i + HORIZON] > threshold).any(axis=0)
                             for i in idx]).ravel().astype(float)

        ftr, ytr = features(fold.train_index, threshold), labelled(fold.train_index)
        fte, yte = features(fold.test_index, threshold), labelled(fold.test_index)
        for name, keys in combos.items():
            xtr = np.column_stack([ftr[k].ravel() for k in keys])
            xte = np.column_stack([fte[k].ravel() for k in keys])
            s = apply_logistic(xte, logistic(xtr, ytr))
            scores[name].append((auc_roc(yte, s), average_precision(yte, s)))

    # ROC points for the figure, pooled across folds. Saved here so the paper's
    # figure module stays a pure plotter with no torch dependency.
    def roc_curve(labels, values, points=120):
        order = np.argsort(-values)
        lab = labels[order]
        tpr = np.cumsum(lab) / max(lab.sum(), 1)
        fpr = np.cumsum(1 - lab) / max((1 - lab).sum(), 1)
        take = np.unique(np.linspace(0, len(lab) - 1, points).astype(int))
        return [[0.0, *fpr[take].tolist(), 1.0], [0.0, *tpr[take].tolist(), 1.0]]

    pooled_labels, pooled_scores = [], {k: [] for k in ("level", "level+rhat", "climate")}
    for fold in folds:
        threshold = np.percentile(cases[: int(fold.train_index[-1])], 90, axis=0)
        ftr = features(fold.train_index, threshold)
        fte = features(fold.test_index, threshold)
        ytr = np.stack([(cases[i : i + HORIZON] > threshold).any(axis=0)
                        for i in fold.train_index]).ravel().astype(float)
        pooled_labels.append(np.stack([(cases[i : i + HORIZON] > threshold).any(axis=0)
                                       for i in fold.test_index]).ravel().astype(float))
        for name, keys in (("level", ["level"]), ("level+rhat", ["level", "rhat"]),
                           ("climate", ["temp"])):
            xtr = np.column_stack([ftr[k].ravel() for k in keys])
            xte = np.column_stack([fte[k].ravel() for k in keys])
            pooled_scores[name].append(apply_logistic(xte, logistic(xtr, ytr)))
    labels_all = np.concatenate(pooled_labels)
    report["roc"] = {name: roc_curve(labels_all, np.concatenate(v))
                     for name, v in pooled_scores.items()}
    report["roc_base_rate"] = float(labels_all.mean())

    report["detection"] = {
        name: {"auc": float(np.mean([a for a, _ in v])),
               "ap": float(np.mean([p for _, p in v]))}
        for name, v in scores.items()
    }
    base_auc = report["detection"]["level"]["auc"]
    print(f"\n{'detection model':32s}{'AUC':>8s}{'AvgPrec':>10s}{'dAUC':>8s}")
    for name, v in report["detection"].items():
        print(f"{name:32s}{v['auc']:8.3f}{v['ap']:10.3f}{v['auc'] - base_auc:+8.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
