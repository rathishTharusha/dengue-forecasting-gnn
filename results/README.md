# Results

Committed metric tables and figures. **These are the artifacts the report cites** — a number
that exists only inside a notebook's output cell cannot be checked, diffed, or re-plotted.

## Conventions

Filename: `<phase>_<model>_<protocol>.csv`, e.g.

```
p1_gcn_rolling-origin.csv
p1_gat_rolling-origin.csv
p1_gcn_ablation.csv
p2_adaptive-graph_rolling-origin.csv
```

Every CSV carries one row per (fold, seed, horizon) — **not** pre-aggregated. Aggregate at
plot/report time so anyone can recompute mean ± std, or drop a bad seed, without a re-run.

Suggested columns:

```
phase,model,fold,seed,horizon,rmse,mae,smape,mape_masked,n_obs,commit
```

Figures go in `results/figures/` as PNG (300 dpi for the report) plus, where practical, the
script or notebook cell that produced them.

## Rule

Anything committed here must have a matching entry in [`../docs/EXPERIMENT_LOG.md`](../docs/EXPERIMENT_LOG.md)
with the config and commit SHA that produced it. No orphan numbers.

## Current contents

Empty — Phase-1 numbers currently live in the notebook outputs and in `EXPERIMENT_LOG.md`
(EXP-002) and still need exporting here. That export is the first task of Phase 2.
