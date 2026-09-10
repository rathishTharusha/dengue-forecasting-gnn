> ## ⚠️ SUPERSEDED — do not cite
>
> This table is from the **n=24** run. It was superseded by
> [`final_table_compute.md`](final_table_compute.md), generated from the **n=64** replication
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

# Computational analysis

Training wall-clock is per fold, single-threaded. Inference is per
25-district forward pass. Measured during the run, not estimated.

| Model | Params | Train (s/fold) | Inference (ms/window) |
|---|---|---|---|
| Persistence floor | 0 | 0.0 | 0.000 |
| Seasonal naive (52 wk) | 0 | 0.0 | 0.000 |
| Ridge regression | 102 | 5.9 | 0.029 |
| Random forest | 619,692 | 86.3 | 3.978 |
| XGBoost | 66,926 | 11.0 | 0.958 |
| LSTM (no graph) | 19,907 | 65.5 | 0.848 |
| Dense GCN, fixed graph | 6,531 | 74.9 | 1.020 |
| + adaptive graph | 7,032 | 72.7 | 1.166 |
| + spatial reg. ($\lambda$=0.01) | 7,032 | 101.1 | 1.270 |
| + spatial reg. ($\lambda$=0.1) | 7,032 | 91.6 | 1.111 |
| + spatial reg. ($\lambda$=1.0) | 7,032 | 64.5 | 0.807 |
| adaptive, $d$=4 | 6,732 | 60.4 | 1.169 |
| adaptive, $d$=20 | 7,532 | 56.2 | 0.938 |
| adaptive, hidden=32 | 2,744 | 45.9 | 0.774 |
| adaptive, hidden=128 | 21,752 | 74.3 | 0.815 |
