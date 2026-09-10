> ## ⚠️ SUPERSEDED — do not cite
>
> This table is from the **n=24** run. It was superseded by
> [`final_table_main.md`](final_table_main.md), generated from the **n=64** replication
> (EXP-009). Kept for provenance only.
>
> The difference matters. At n=24 the adaptive graph appeared to beat its
> matched fixed-graph control (17/24, p=0.064). At n=64 that effect **did not
> survive** — see EXP-009 in `docs/EXPERIMENT_LOG.md`. An independent
> reimplementation under the frozen protocol reached the same conclusion
> (EXP-018, `analysis/notebooks/E2_adaptive_graph.ipynb`): paired against a fixed
> graph, `adaptive` was +0.46 RMSE *worse* (p=0.09), and removing the graph
> entirely was statistically indistinguishable from keeping it (p=0.49).
>
> `paper/sections/05_results.tex` already cites the n=64 figures and is correct.

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
| Dense GCN, fixed graph | 24 | 64.67 ± 50.99 | 24.45 | 66.8% | 7/24, p=0.064 |
| + adaptive graph | 24 | 60.08 ± 44.25 | 22.92 | 65.5% | 9/24, p=0.307 |
| + spatial reg. ($\lambda$=0.01) | 24 | 60.72 ± 44.12 | 23.14 | 66.1% | 10/24, p=0.541 |
| + spatial reg. ($\lambda$=0.1) | 24 | 70.27 ± 60.95 | 25.72 | 68.9% | 7/24, p=0.064 |
| + spatial reg. ($\lambda$=1.0) | 24 | 94.93 ± 77.97 | 38.74 | 88.3% | 5/24, p=0.007 |
| adaptive, $d$=4 | 24 | 73.64 ± 83.21 | 24.87 | 66.2% | 7/24, p=0.064 |
| adaptive, $d$=20 | 24 | 63.68 ± 51.33 | 23.75 | 66.3% | 7/24, p=0.064 |
| adaptive, hidden=32 | 24 | 68.82 ± 69.05 | 24.37 | 65.7% | 6/24, p=0.023 |
| adaptive, hidden=128 | 24 | 67.20 ± 60.30 | 23.93 | 65.0% | 8/24, p=0.152 |

## The contribution, against its own control

Adaptive graph vs the matched fixed-graph control: **17/24 paired wins, p=0.064**. This is the comparison that isolates the learned adjacency —
same propagation path, same head, same optimiser, same target handling.
