# Baseline results: what v1 showed, and what v2 fixes

## What your v1 run told us (and it's important)

**1. The plain GCN loses to naive persistence.**
- GCN (tuned) overall test RMSE ≈ 67–72, MAE ≈ 24.
- Persistence ("next week = this week") overall RMSE ≈ 59.5, MAE ≈ 13.8.
- Persistence wins on every metric. This is real, not a bug: dengue here is heavy-tailed
  (median 13 cases, max 2631) with lag-1 autocorrelation ≈ 0.68, so MSE pushes the GNN
  toward a bland near-mean prediction while persistence rides the autocorrelation for free.

**2. The single 70/10/20 split makes model selection unreliable.**
- In the tuning table, validation RMSE and test RMSE are **anti-correlated (r = −0.74)**.
- The config selected on validation (best val 90.9) gave test RMSE 67.2 — one of the *worst*.
- Three configs that would have **beaten persistence on test** (RMSE 53–56) were discarded
  because their validation RMSE looked bad.
- Cause: the 45-week validation slice and the test slice are in different epidemic regimes,
  so "best on validation" ≠ "best on test." A single split cannot be trusted here.

## What v2 changes (all tested locally on the real data)

| Fix | Why | Effect (rolling-origin CV) |
|---|---|---|
| **Rolling-origin cross-validation** | one split is regime-dependent and anti-predictive | robust, leakage-free model selection; report mean ± std across folds |
| **Residual-over-persistence** | GNN predicts the *correction* to last week, not absolute cases | RMSE ≈ 73 → ≈ 46, now **matches the persistence floor** |
| **log1p target space** | compresses the heavy tail so MSE isn't hijacked by peaks | stabilizes training (note: log *alone*, without residual, makes it worse) |
| **Gradient clipping** | kills the occasional hallucinated spike | steadier convergence |

Tested rolling-origin summary (GCN, 3 folds):
- plain GCN: RMSE 73.5 ± 19.4, MAE 29.1
- **residual + log GCN: RMSE 45.8 ± 15.0, MAE 16.6** (persistence ≈ 44.8)

## The honest framing for the paper

The tuned spatio-temporal GNN baseline **matches the persistence floor** — it does not beat it
by a wide margin. That is the correct, honest starting point on a 25-node, heavy-tailed,
highly-autocorrelated dataset, and it is *good* for the project: it gives your three
contributions (physics-informed loss, GAN augmentation, adaptive graph) clear, measurable
room to push **below** the persistence floor. A baseline that already matched or beat a strong
naive method is far more credible to reviewers than one that loses embarrassingly.

Trade-off to be aware of: a stronger baseline is a higher bar for your own contributions.
That's the right kind of hard — improvements over it will be real.

## How to run v2

Same setup as before (Drive or GitHub download). Then Run all. New sections:
- **§8 Rolling-origin CV** — the main baseline numbers for the paper.
- **§9 Ablation** — plain → +log → +residual → residual+log (the before/after table).
- **§10 GAT** — same protocol for the GAT baseline.
- **§11** — prediction plot at the last origin.

Toggle `CFG.residual` / `CFG.log_transform` to reproduce any row of the ablation.

## Still to do for Phase 1
- [ ] Run v2 end-to-end on Colab GPU (fast — the CV is the heavy part).
- [ ] Capture the §8 GCN + §10 GAT rolling-origin tables.
- [ ] Capture the §9 ablation table.
- [ ] One §11 plot.
Then we drop these into the proposal's Baseline Model section and move to Proposed Contribution + Experimental Plan.
