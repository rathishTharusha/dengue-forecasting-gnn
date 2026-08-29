# Per-horizon breakdown

Peak week error is the mean absolute displacement of the predicted outbreak
peak, over districts reaching PEAK_FLOOR cases. Persistence is shifted h weeks
by construction, which is the sanity check that the metric works.

| Model | h | RMSE | MAE | Peak err (wks) |
|---|---|---|---|---|
| Persistence floor | 1 | 39.34 | 13.07 | 0.99 |
| Persistence floor | 2 | 44.11 | 15.57 | 1.90 |
| Persistence floor | 3 | 49.79 | 18.53 | 3.44 |
| Dense GCN, fixed graph | 1 | 40.21 | 13.37 | 1.98 |
| Dense GCN, fixed graph | 2 | 44.70 | 15.76 | 4.59 |
| Dense GCN, fixed graph | 3 | 50.29 | 18.49 | 5.47 |
| + adaptive graph | 1 | 39.44 | 13.18 | 2.92 |
| + adaptive graph | 2 | 44.60 | 15.79 | 4.70 |
| + adaptive graph | 3 | 49.73 | 18.42 | 6.56 |
| + spatial reg. ($\lambda$=0.01) | 1 | 40.12 | 13.33 | 2.73 |
| + spatial reg. ($\lambda$=0.01) | 2 | 45.70 | 16.03 | 5.11 |
| + spatial reg. ($\lambda$=0.01) | 3 | 50.70 | 18.59 | 6.76 |
| + spatial reg. ($\lambda$=0.1) | 1 | 40.68 | 13.72 | 4.69 |
| + spatial reg. ($\lambda$=0.1) | 2 | 45.42 | 16.04 | 5.36 |
| + spatial reg. ($\lambda$=0.1) | 3 | 50.31 | 18.47 | 6.55 |
| + spatial reg. ($\lambda$=1.0) | 1 | 66.47 | 24.28 | 6.77 |
| + spatial reg. ($\lambda$=1.0) | 2 | 67.59 | 25.08 | 7.73 |
| + spatial reg. ($\lambda$=1.0) | 3 | 69.03 | 25.96 | 8.56 |
