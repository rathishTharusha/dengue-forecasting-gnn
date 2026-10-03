"""Stage S2 — Causal Covariate Check (Descriptive Cross-Correlation)

Computes cross-correlations between log1p cases at week t and:
- ERA5 climate channels at lags 2..20 (ERA5[t - l])
- NDVI at lags 0..20 (NDVI[t - l])

over the training weeks of the first rolling origin (0.55) on the rebuilt dataset.
Saves results and plots to `analysis/results/seir_gnn/s2_causal_ccf/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))

import corrected_data  # noqa: E402

OUT_DIR = REPO / "analysis" / "results" / "seir_gnn" / "s2_causal_ccf"


def run_s2():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = corrected_data.load()

    # Rebuilt series folds
    n_weeks = data.cases.shape[0]
    window, horizon = 3, 3
    ids = list(range(window, n_weeks - horizon))
    cut = int(0.55 * len(ids))
    train_ids = np.asarray(ids[: cut - 30])

    log_cases = np.log1p(data.cases)

    # Output records
    records = []

    # 1. ERA5 channels: lags 2..20
    # data.climate shape: (T, N, C)
    era5_lags = list(range(2, 21))
    for c_idx, channel_name in enumerate(data.climate_channels):
        for dist_idx, dist_name in enumerate(data.names):
            y = log_cases[train_ids, dist_idx]
            for lag in era5_lags:
                valid_t = [t for t in train_ids if t - lag >= 0 and not np.isnan(data.cases[t, dist_idx])]
                if len(valid_t) < 30:
                    continue
                y_sub = log_cases[valid_t, dist_idx]
                x_sub = data.climate[[t - lag for t in valid_t], dist_idx, c_idx]
                # Filter out any NaNs if present
                mask = ~(np.isnan(y_sub) | np.isnan(x_sub))
                if mask.sum() < 30:
                    continue
                r = float(np.corrcoef(y_sub[mask], x_sub[mask])[0, 1])
                records.append({
                    "district": dist_name,
                    "covariate": f"era5_{channel_name}",
                    "lag": lag,
                    "r": r
                })

    # 2. NDVI: lags 0..20
    # data.ndvi shape: (T, N)
    ndvi_lags = list(range(0, 21))
    for dist_idx, dist_name in enumerate(data.names):
        for lag in ndvi_lags:
            valid_t = [t for t in train_ids if t - lag >= 0 and not np.isnan(data.cases[t, dist_idx])]
            if len(valid_t) < 30:
                continue
            y_sub = log_cases[valid_t, dist_idx]
            x_sub = data.ndvi[[t - lag for t in valid_t], dist_idx]
            mask = ~(np.isnan(y_sub) | np.isnan(x_sub))
            if mask.sum() < 30:
                continue
            r = float(np.corrcoef(y_sub[mask], x_sub[mask])[0, 1])
            records.append({
                "district": dist_name,
                "covariate": "ndvi",
                "lag": lag,
                "r": r
            })

    df = pd.DataFrame(records)
    csv_path = OUT_DIR / "causal_ccf_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} CCF records to {csv_path}")

    # Summary table per covariate: max positive & negative correlation across districts and lags
    summary = df.groupby(["covariate", "lag"])["r"].mean().reset_index()
    print("\n=== Mean Cross-Correlation across 25 Districts ===")
    for cov in df["covariate"].unique():
        sub = summary[summary.covariate == cov]
        best_row = sub.loc[sub["r"].abs().idxmax()]
        print(f"{cov:30s} peak lag={int(best_row.lag):2d}  r={best_row.r:+.4f}")

    # Create figure plot
    plt.figure(figsize=(10, 6))
    for cov in df["covariate"].unique():
        sub = summary[summary.covariate == cov]
        plt.plot(sub["lag"], sub["r"], label=cov, marker="o", ms=4)
    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Lag (weeks)")
    plt.ylabel("Mean Pearson r with log1p(cases)")
    plt.title("Stage S2: Causal Covariate Cross-Correlation (Origin 0.55 Train)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig_path = OUT_DIR / "causal_ccf_plot.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved plot to {fig_path}")

    return 0


if __name__ == "__main__":
    sys.exit(run_s2())
