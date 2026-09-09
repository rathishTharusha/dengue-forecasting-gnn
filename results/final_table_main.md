# Main results — rolling-origin CV, pooled across horizons

Mean ± SD over matched runs. `vs floor` is a two-sided paired sign test
against persistence on matched (fold, seed) runs.

| Model | n | RMSE | MAE | SMAPE | vs floor |
|---|---|---|---|---|---|
| Persistence floor | 8 | 55.12 ± 41.03 | 21.70 | 59.7% | — |
| Seasonal naive (52 wk) | 8 | 153.17 ± 79.72 | 68.18 | 113.3% | 0/8, p=0.008 |
| Ridge regression | 8 | 290.81 ± 675.61 | 39.52 | 67.1% | 2/8, p=0.289 |
| Random forest | 24 | 56.31 ± 39.38 | 21.75 | 64.5% | 9/24, p=0.307 |
| XGBoost | 24 | 57.41 ± 40.36 | 22.29 | 65.9% | 7/24, p=0.064 |
| LSTM (no graph) | 24 | 57.95 ± 42.17 | 22.26 | 66.3% | 10/24, p=0.541 |
| Adaptive graph (no temporal) | 64 | 61.81 ± 46.33 | 23.41 | 65.8% | 21/64, p=0.008 |
| Adaptive graph + shrinkage | 64 | 57.84 ± 40.27 | 22.51 | 64.1% | 22/64, p=0.017 |
| + growth constraint + shrinkage | 64 | 58.03 ± 40.71 | 22.47 | 63.8% | 24/64, p=0.060 |
| + gated temporal conv. | 64 | 56.78 ± 38.74 | 22.25 | 65.2% | 26/64, p=0.169 |
| + gated TCN + shrinkage | 64 | 57.74 ± 40.43 | 22.55 | 65.1% | 18/64, p=0.001 |
| aug: jitter | 24 | 68.57 ± 65.28 | 24.36 | 64.9% | 6/24, p=0.023 |
| aug: window warp | 24 | 66.60 ± 60.05 | 23.79 | 65.1% | 6/24, p=0.023 |
| aug: GAN (WGAN-GP) | 24 | 93.81 ± 108.00 | 30.98 | 71.2% | 3/24, p=0.000 |
