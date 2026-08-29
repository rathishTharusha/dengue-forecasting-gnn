# Main results — rolling-origin CV, pooled across horizons

Mean ± SD over 3 origins × 3 seeds (persistence is deterministic: 3 folds).
`vs floor` is a paired sign test against persistence on matched (fold, seed) runs.

| Model | RMSE | MAE | SMAPE | vs floor |
|---|---|---|---|---|
| Persistence floor | 44.80 ± 21.49 | 15.72 | 62.8% | — |
| Dense GCN, fixed graph | 45.47 ± 16.91 | 15.87 | 69.6% | 4/9 — not significant |
| + adaptive graph | 45.00 ± 16.18 | 15.79 | 69.0% | 3/9 — not significant |
| + spatial reg. ($\lambda$=0.01) | 45.92 ± 15.98 | 15.98 | 68.9% | 3/9 — not significant |
| + spatial reg. ($\lambda$=0.1) | 45.79 ± 14.08 | 16.08 | 71.9% | 3/9 — not significant |
| + spatial reg. ($\lambda$=1.0) | 67.72 ± 15.44 | 25.11 | 85.8% | 2/9 — not significant |
