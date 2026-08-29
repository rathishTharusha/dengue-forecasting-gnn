# Spatial regularisation sweep

λ=0 is the adaptive graph with no regulariser. Applied to log1p(counts);
see docs/PHASE2_REVIEW.md F5 for why not raw counts.

| λ | RMSE | SD | learned gate σ(g) |
|---|---|---|---|
| 0.00 | 45.00 | ± 16.18 | 0.623 |
| 0.01 | 45.92 | ± 15.98 | 0.871 |
| 0.10 | 45.79 | ± 14.08 | 0.945 |
| 1.00 | 67.72 | ± 15.44 | 0.860 |
