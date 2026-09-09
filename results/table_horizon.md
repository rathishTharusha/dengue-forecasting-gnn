# Per-horizon breakdown

Peak week error is the mean absolute displacement of the predicted outbreak
peak, over districts reaching PEAK_FLOOR cases. Persistence is shifted h weeks
by construction, which is the sanity check that the metric works.

| Model | h | RMSE | MAE | Peak err (wks) |
|---|---|---|---|---|
| Persistence floor | 1 | 43.27 | 17.15 | 1.70 |
| Persistence floor | 2 | 53.31 | 21.55 | 2.73 |
| Persistence floor | 3 | 65.62 | 26.40 | 4.02 |
| Seasonal naive (52 wk) | 1 | 152.06 | 68.15 | 12.75 |
| Seasonal naive (52 wk) | 2 | 152.97 | 68.18 | 12.57 |
| Seasonal naive (52 wk) | 3 | 153.85 | 68.20 | 12.14 |
| Ridge regression | 1 | 387.68 | 42.78 | 3.38 |
| Ridge regression | 2 | 255.28 | 38.83 | 3.89 |
| Ridge regression | 3 | 168.48 | 36.96 | 5.05 |
| Random forest | 1 | 44.37 | 17.26 | 2.64 |
| Random forest | 2 | 55.45 | 21.85 | 3.68 |
| Random forest | 3 | 66.28 | 26.14 | 4.85 |
| XGBoost | 1 | 44.68 | 17.63 | 3.50 |
| XGBoost | 2 | 55.81 | 22.16 | 3.96 |
| XGBoost | 3 | 68.37 | 27.09 | 4.94 |
| LSTM (no graph) | 1 | 45.92 | 17.97 | 2.32 |
| LSTM (no graph) | 2 | 56.29 | 22.16 | 3.40 |
| LSTM (no graph) | 3 | 68.46 | 26.65 | 4.80 |
| Dense GCN, fixed graph | 1 | 52.78 | 19.77 | 2.55 |
| Dense GCN, fixed graph | 2 | 64.43 | 24.73 | 3.89 |
| Dense GCN, fixed graph | 3 | 73.97 | 28.85 | 4.83 |
| + adaptive graph | 1 | 48.34 | 18.57 | 2.41 |
| + adaptive graph | 2 | 58.37 | 22.95 | 3.54 |
| + adaptive graph | 3 | 70.26 | 27.24 | 4.84 |
| + spatial reg. ($\lambda$=0.01) | 1 | 49.51 | 18.80 | 2.46 |
| + spatial reg. ($\lambda$=0.01) | 2 | 61.11 | 23.43 | 3.55 |
| + spatial reg. ($\lambda$=0.01) | 3 | 68.88 | 27.20 | 4.91 |
| + spatial reg. ($\lambda$=0.1) | 1 | 61.99 | 22.23 | 2.99 |
| + spatial reg. ($\lambda$=0.1) | 2 | 70.52 | 26.09 | 4.25 |
| + spatial reg. ($\lambda$=0.1) | 3 | 76.42 | 28.84 | 5.05 |
| + spatial reg. ($\lambda$=1.0) | 1 | 91.66 | 37.50 | 5.04 |
| + spatial reg. ($\lambda$=1.0) | 2 | 94.77 | 38.80 | 5.74 |
| + spatial reg. ($\lambda$=1.0) | 3 | 97.80 | 39.93 | 6.77 |
| adaptive, $d$=4 | 1 | 52.47 | 19.49 | 2.41 |
| adaptive, $d$=4 | 2 | 61.24 | 23.51 | 3.63 |
| adaptive, $d$=4 | 3 | 94.20 | 31.62 | 4.69 |
| adaptive, $d$=20 | 1 | 53.33 | 19.55 | 2.45 |
| adaptive, $d$=20 | 2 | 62.34 | 24.08 | 3.80 |
| adaptive, $d$=20 | 3 | 72.01 | 27.63 | 5.10 |
| adaptive, hidden=32 | 1 | 56.53 | 20.15 | 2.37 |
| adaptive, hidden=32 | 2 | 56.98 | 22.51 | 3.79 |
| adaptive, hidden=32 | 3 | 83.42 | 30.45 | 4.91 |
| adaptive, hidden=128 | 1 | 49.22 | 18.86 | 2.44 |
| adaptive, hidden=128 | 2 | 57.24 | 22.57 | 3.66 |
| adaptive, hidden=128 | 3 | 85.32 | 30.35 | 4.68 |
