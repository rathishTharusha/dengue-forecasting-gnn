# 0001 — Residual-over-persistence, log1p target, and rolling-origin CV

- **Status:** Accepted
- **Date:** 2026-08-04
- **Deciders:** Group 05

## Context

The v1 baseline (plain GCN, absolute case scale, single chronological 70/10/20 split) lost to
naive persistence on every metric: GCN test RMSE ≈ 67–72 / MAE ≈ 24 against persistence
RMSE ≈ 59.5 / MAE ≈ 13.8.

Two properties of the data explain it, and neither is a bug:

1. **Heavy tail.** Median 13 weekly cases, max 2631, 9.7% zeros. MSE on the raw scale is
   dominated by a handful of outbreak weeks, which pushes the network toward a bland
   near-mean prediction.
2. **Strong autocorrelation.** Lag-1 ≈ 0.68. Persistence rides that for free; a model
   predicting absolute counts has to re-learn it from scratch before it can add anything.

Separately, the single split proved unusable for **model selection**: across the tuning grid,
validation and test RMSE were **anti-correlated (r = −0.74)**. The config chosen on validation
(best val 90.9) scored test 67.2 — near-worst — while three configs that would have beaten
persistence on test (RMSE 53–56) were discarded for poor validation RMSE. The 45-week
validation slice and the test slice fall in different epidemic regimes, so "best on
validation" carried no information about test.

## Decision

Three changes, adopted together as the v2 baseline:

1. **Residual-over-persistence.** The network predicts the *correction* to last week's value,
   not the absolute count.
2. **log1p target space.** Train on `log1p(cases)`, invert before scoring, so MSE is not
   hijacked by peaks.
3. **Rolling-origin (expanding-window) cross-validation** — 3 chronological origins, averaged
   over 3 seeds — replacing the single 70/10/20 split for both selection and reporting.

Plus gradient clipping (`grad_clip = 5.0`) to suppress occasional hallucinated spikes, and
z-normalization computed from **training-fold statistics only**.

## Alternatives considered

| Option | Why not |
|---|---|
| log1p alone, without residual | Measured: RMSE 66.2 → 67.1, *worse*. Compressing the tail without removing the autocorrelation baseline does not help. |
| Keep the single chronological split | Demonstrably anti-predictive here (r = −0.74). Cheaper, but the selection it produces is noise. |
| Weighted loss emphasizing outbreak weeks | Not tried; adds a hyperparameter that would need its own sweep, and residual learning already recovered most of the gap. Worth revisiting if peak-timing metrics stay weak. |
| Shuffled splits / k-fold | Temporal leakage. Non-negotiable on a forecasting task. |

## Consequences

- The baseline moves from RMSE ≈ 66 to ≈ 45, **matching** the persistence floor (44.8) rather
  than beating it. That is the honest framing carried into the proposal.
- A stronger baseline is a higher bar for our own contributions — deliberately. An improvement
  over a floor-matching baseline is a real improvement, not the rescue of a weak one.
- Reported numbers are now mean ± std across folds, which costs ~3× the compute per
  configuration. Acceptable: a fold trains in minutes on a 25-node graph.
- **Every subsequent phase must use this identical protocol**, or the ablation table cannot be
  compared row to row.
- Ablation evidence lives in `notebooks/baseline/dengue_baseline_GNN_v2.ipynb` §9 and
  `docs/EXPERIMENT_LOG.md` EXP-001/EXP-002.

## Revisit if

Peak-onset detection stays poor even after the physics loss lands — that would suggest the
residual formulation is capping how much outbreak-shape signal the model can express, and a
direct-count head with a tail-aware loss deserves another look.
