"""Is an outbreak detectable 3 weeks ahead, and is the target worth changing to?

Why ask this
------------
EXP-021 localised the whole problem: outbreak windows are 12.6% of the data and
61.7% of the squared error, forecast with a -12.9 case bias. EXP-024 closed off
the mechanistic route to fixing it -- `R_t` is 26% predictable, climate explains
nothing, susceptible depletion is 500x too slow.

But every one of those tests asked the *regression* question: how many cases next
week. That is a question persistence is close to optimal on, because cases at
*t-1* explain r^2 = 0.85 of cases at *t*. The question a health ministry acts on
is different and binary: **is this district about to exceed its alert threshold?**

Those are not the same problem, and being bad at one does not imply being bad at
the other. A model can be systematically 13 cases low during outbreaks -- fatal
for RMSE -- while still ranking which districts are about to break out correctly.
This script measures whether that ranking signal exists, before anything is built
on the assumption that it does.

Discipline: same as EXP-024. Rolling origin, no shuffling, thresholds fitted on
training weeks only, and a stupid baseline that has to be beaten.

Run::

    C:/Users/tharu/rp/v/Scripts/python.exe analysis/_build/outbreak_signal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "src"))

import adaptive as base  # noqa: E402
import renewal  # noqa: E402

from dengue_gnn import seir  # noqa: E402

NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
OUT = REPO / "analysis" / "results" / "outbreak_signal.json"

HORIZON = 3
TEMP, PRECIP, NDVI = 0, 7, 10


def auc_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC, ties averaged. No sklearn in the pinned stack."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tied groups, or a constant scorer reports AUC != 0.5.
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    pos, neg = labels.sum(), (~labels.astype(bool)).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[labels.astype(bool)].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under precision-recall. The honest metric when positives are rare."""
    order = np.argsort(-scores, kind="mergesort")
    hits = labels[order].astype(float)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    total = hits.sum()
    return float((precision * hits).sum() / total) if total else float("nan")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raw = np.nan_to_num(np.load(NPY, allow_pickle=True)).astype(np.float64)
    cases, adjacency, _names = base.load_dataset(NPY, ADJ)
    _weeks, _districts = cases.shape
    w = renewal.generation_interval(seir.PAPER_SECTION4)

    folds = base.build_folds(cases, 3, HORIZON)
    report: dict = {"folds": []}

    print("Outbreak = district-week above that district's 90th percentile,")
    print("threshold fitted on TRAINING weeks only. Predicted 3 weeks ahead.\n")

    agg = {}
    for fold in folds:
        train_end = int(fold.train_index[-1])
        thresh = np.percentile(cases[:train_end], 90, axis=0)

        idx = fold.test_index
        # Label: does the district exceed its threshold anywhere in the horizon?
        label = np.stack([(cases[i : i + HORIZON] > thresh).any(axis=0) for i in idx])

        last = cases[idx - 1]
        prev4 = np.stack([cases[i - 4 : i].mean(axis=0) for i in idx])
        growth = np.log1p(last) - np.log1p(prev4)

        force = np.stack([
            sum(w[lag] * cases[i - 1 - lag] for lag in range(len(w))) for i in idx
        ])
        r_hat = last / np.maximum(force, 1.0)

        # Neighbour pressure: mean of adjacent districts' current level, relative
        # to their own thresholds, so Colombo does not dominate every neighbour.
        rel = last / np.maximum(thresh, 1.0)
        neighbour = rel @ (adjacency > 0).astype(float).T
        neighbour /= np.maximum((adjacency > 0).sum(axis=1), 1)

        climate = np.stack([raw[i - 1, :, [TEMP, PRECIP, NDVI]].T for i in idx])

        scorers = {
            "current level / threshold": rel,
            "recent growth": growth,
            "R_hat (renewal)": r_hat,
            "neighbour pressure": neighbour,
            "temperature": climate[..., 0],
            "precipitation": climate[..., 1],
            "level x growth": rel * np.exp(growth),
        }

        flat_label = label.ravel()
        row = {"origin": fold.origin, "positives": int(flat_label.sum()),
               "n": int(flat_label.size), "scores": {}}
        for name, score in scorers.items():
            row["scores"][name] = {
                "auc": auc_roc(flat_label, score.ravel()),
                "ap": average_precision(flat_label, score.ravel()),
            }
            agg.setdefault(name, []).append(row["scores"][name])
        report["folds"].append(row)

        print(f"origin {fold.origin}: {row['positives']}/{row['n']} district-weeks "
              f"are outbreaks ({flat_label.mean():.1%} base rate)")

    print(f"\n{'scorer':30s}{'AUC':>8s}{'AvgPrec':>10s}{'lift vs base':>14s}")
    print("-" * 62)
    base_rate = np.mean([r["positives"] / r["n"] for r in report["folds"]])
    ranked = sorted(agg.items(), key=lambda kv: -np.nanmean([d["auc"] for d in kv[1]]))
    for name, vals in ranked:
        auc = float(np.nanmean([d["auc"] for d in vals]))
        ap = float(np.nanmean([d["ap"] for d in vals]))
        report.setdefault("summary", {})[name] = {"auc": auc, "ap": ap}
        print(f"{name:30s}{auc:8.3f}{ap:10.3f}{ap / base_rate:13.2f}x")

    print(f"\nbase rate {base_rate:.1%}; AUC 0.5 and AvgPrec = base rate mean no signal.")
    print("An AUC well above 0.5 here means outbreaks ARE rankable three weeks out,")
    print("even though EXP-021 showed the same models cannot put a number on them.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
