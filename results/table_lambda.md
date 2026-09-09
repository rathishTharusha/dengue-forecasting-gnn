# Spatial regularisation sweep

λ=0 is the adaptive graph with no regulariser. Applied to log1p(counts);
see docs/PHASE2_REVIEW.md F5 for why not raw counts.

| λ | RMSE | SD | learned gate σ(g) |
|---|---|---|---|
| 0.00 | 60.08 | ± 44.25 | 0.635 |
| 0.01 | 60.72 | ± 44.12 | 0.865 |
| 0.10 | 70.27 | ± 60.95 | 0.926 |
| 1.00 | 94.93 | ± 77.97 | 0.854 |
