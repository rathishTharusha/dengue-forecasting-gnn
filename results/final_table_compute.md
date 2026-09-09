# Computational analysis

Training wall-clock is per fold, single-threaded. Inference is per
25-district forward pass. Measured during the run, not estimated.

| Model | Params | Train (s/fold) | Inference (ms/window) |
|---|---|---|---|
| Persistence floor | 0 | 0.0 | 0.000 |
| Seasonal naive (52 wk) | 0 | 0.0 | 0.000 |
| Ridge regression | 102 | 4.5 | 0.018 |
| Random forest | 619,692 | 83.3 | 3.851 |
| XGBoost | 66,926 | 10.6 | 0.914 |
| LSTM (no graph) | 19,907 | 60.2 | 0.787 |
| Adaptive graph (no temporal) | 7,032 | 59.0 | 0.995 |
| Adaptive graph + shrinkage | 7,032 | 56.6 | 0.954 |
| + growth constraint + shrinkage | 7,032 | 67.8 | 0.965 |
| + gated temporal conv. | 11,960 | 153.8 | 3.101 |
| + gated TCN + shrinkage | 11,960 | 154.3 | 2.749 |
| aug: jitter | 7,032 | 85.6 | 0.903 |
| aug: window warp | 7,032 | 90.0 | 0.907 |
| aug: GAN (WGAN-GP) | 7,032 | 81.5 | 0.892 |
