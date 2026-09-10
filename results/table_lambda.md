> ## ⚠️ SUPERSEDED — do not cite
>
> This table is from the **n=24** run. It was superseded by
> [`final_table_lambda.md`](final_table_lambda.md), generated from the **n=64** replication
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

# Spatial regularisation sweep

λ=0 is the adaptive graph with no regulariser. Applied to log1p(counts);
see docs/PHASE2_REVIEW.md F5 for why not raw counts.

| λ | RMSE | SD | learned gate σ(g) |
|---|---|---|---|
| 0.00 | 60.08 | ± 44.25 | 0.635 |
| 0.01 | 60.72 | ± 44.12 | 0.865 |
| 0.10 | 70.27 | ± 60.95 | 0.926 |
| 1.00 | 94.93 | ± 77.97 | 0.854 |
