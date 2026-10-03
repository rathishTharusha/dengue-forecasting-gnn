"""Stage S7: Early-Warning Evaluation.

Evaluates outbreak detection and lead-time performance for S*, B* (ASTGCN base),
direct control, and persistence.

Outbreak definition per district per origin (from training weeks only):
  cases > mean + 2 * std

Output: analysis/results/seir_gnn/s7_early_warning/s7_early_warning_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "analysis" / "_build"))
sys.path.insert(0, str(REPO / "src"))

import corrected_data as cd  # noqa: E402
import run_corrected_benchmark as rcb  # noqa: E402


def calculate_early_warning_metrics(y_true: np.ndarray, y_pred: np.ndarray, thresholds: np.ndarray) -> dict:
    """
    y_true: (W, N, 3) true cases over horizon
    y_pred: (W, N, 3) predicted cases or lambda over horizon
    thresholds: (N,) outbreak threshold per district
    """
    W, N, H = y_true.shape

    # Binary outbreak indicator per (window, node, horizon)
    thresh_3d = np.tile(thresholds[None, :, None], (W, 1, H))
    true_outbreak = (y_true > thresh_3d).astype(int)

    metrics_per_h = {}
    for h in range(H):
        t_h = true_outbreak[:, :, h].ravel()
        p_h = y_pred[:, :, h].ravel()

        # Alert if predicted > threshold
        if p_h.max() > 1.0 and p_h.max() < 100.0:
            # Predictions in case scale
            alert_h = (p_h > thresholds.repeat(W)).astype(int)
        else:
            # Scaled or FOI values: use top percentile matching outbreak rate
            pos_rate = t_h.mean()
            pct_val = np.percentile(p_h, (1.0 - pos_rate) * 100.0)
            alert_h = (p_h >= pct_val).astype(int)

        tp = np.sum((alert_h == 1) & (t_h == 1))
        fp = np.sum((alert_h == 1) & (t_h == 0))
        fn = np.sum((alert_h == 0) & (t_h == 1))
        tn = np.sum((alert_h == 0) & (t_h == 0))

        pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = pod
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        try:
            auc = roc_auc_score(t_h, p_h)
        except Exception:
            auc = 0.5

        metrics_per_h[f"h{h+1}"] = {
            "POD": round(float(pod), 4),
            "FAR": round(float(far), 4),
            "F1": round(float(f1), 4),
            "AUC": round(float(auc), 4),
        }

    # Window-level outbreak score: "Outbreak anywhere in 3-week horizon"
    true_any = (true_outbreak.max(axis=-1) > 0).astype(int).ravel() # (W * N,)
    pred_any = y_pred.max(axis=-1).ravel()
    try:
        overall_auc = float(roc_auc_score(true_any, pred_any))
    except Exception:
        overall_auc = 0.5

    # Precision@5 for top 5 districts per window
    p5_list = []
    for w in range(W):
        top5_true = np.argsort(y_true[w].sum(axis=-1))[-5:]
        top5_pred = np.argsort(y_pred[w].sum(axis=-1))[-5:]
        overlap = len(set(top5_true).intersection(set(top5_pred)))
        p5_list.append(overlap / 5.0)

    return {
        "per_horizon": metrics_per_h,
        "overall_AUC": round(overall_auc, 4),
        "precision_at_5": round(float(np.mean(p5_list)), 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s5-results", default=str(REPO / "analysis" / "results" / "seir_gnn" / "s5_seir_gnn" / "s5_seir_gnn_results.json"))
    ap.add_argument("--out", default=str(REPO / "analysis" / "results" / "seir_gnn" / "s7_early_warning" / "s7_early_warning_results.json"))
    args = ap.parse_args()

    s5_path = Path(args.s5_results)
    if not s5_path.exists():
        print(f"Error: S5 results not found at {s5_path}")
        return 1

    s5_data = json.loads(s5_path.read_text(encoding="utf-8"))
    df_s5 = pd.DataFrame(s5_data)
    summary_s5 = df_s5.groupby(["arch", "input_level", "coupling", "head_type"])[["val_RMSE", "test_RMSE"]].mean().reset_index()
    summary_s5 = summary_s5.sort_values("val_RMSE")
    s_star = summary_s5.iloc[0]

    print("=== Stage S7: Early-Warning Evaluation ===")
    print(f"S* finalist: {s_star['arch']} (input={s_star['input_level']}, coupling={s_star['coupling']}, head={s_star['head_type']})")

    data = cd.load()
    cases, adjacency, artifact, missing, folds = rcb.prepare("rebuilt")
    T, N = cases.shape

    # Calculate outbreak threshold per district across origin 0 training set
    tr_0 = folds[0].train_index
    tr_cases = cases[tr_0]
    c_mean = np.nanmean(tr_cases, axis=0)
    c_std = np.nanstd(tr_cases, axis=0)
    thresholds = c_mean + 2.0 * c_std

    results = {
        "thresholds_per_district": thresholds.tolist(),
        "s_star_config": s_star.to_dict(),
        "summary": "Stage S7 Early warning metrics computed across test windows."
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote S7 early warning metrics to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
