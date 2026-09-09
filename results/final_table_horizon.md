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
| Adaptive graph (no temporal) | 1 | 51.91 | 19.24 | 2.54 |
| Adaptive graph (no temporal) | 2 | 60.82 | 23.61 | 3.74 |
| Adaptive graph (no temporal) | 3 | 70.15 | 27.38 | 4.85 |
| Adaptive graph + shrinkage | 1 | 45.02 | 17.73 | 2.20 |
| Adaptive graph + shrinkage | 2 | 56.31 | 22.56 | 3.45 |
| Adaptive graph + shrinkage | 3 | 68.83 | 27.24 | 4.60 |
| + growth constraint + shrinkage | 1 | 44.68 | 17.56 | 2.08 |
| + growth constraint + shrinkage | 2 | 56.33 | 22.46 | 3.26 |
| + growth constraint + shrinkage | 3 | 69.46 | 27.40 | 4.63 |
| + gated temporal conv. | 1 | 44.65 | 17.56 | 2.20 |
| + gated temporal conv. | 2 | 55.76 | 22.35 | 3.37 |
| + gated temporal conv. | 3 | 66.86 | 26.85 | 4.64 |
| + gated TCN + shrinkage | 1 | 44.66 | 17.61 | 2.16 |
| + gated TCN + shrinkage | 2 | 56.42 | 22.62 | 3.38 |
| + gated TCN + shrinkage | 3 | 68.66 | 27.41 | 4.62 |
| aug: jitter | 1 | 49.71 | 18.89 | 2.56 |
| aug: jitter | 2 | 65.02 | 24.13 | 3.84 |
| aug: jitter | 3 | 83.44 | 30.05 | 4.90 |
| aug: window warp | 1 | 49.74 | 18.98 | 2.66 |
| aug: window warp | 2 | 59.99 | 23.14 | 3.74 |
| aug: window warp | 3 | 82.48 | 29.24 | 4.70 |
| aug: GAN (WGAN-GP) | 1 | 105.83 | 33.07 | 2.26 |
| aug: GAN (WGAN-GP) | 2 | 73.44 | 28.56 | 3.69 |
| aug: GAN (WGAN-GP) | 3 | 81.38 | 31.31 | 5.05 |
