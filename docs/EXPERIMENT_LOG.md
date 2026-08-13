# Experiment log

Append-only. Newest entries at the top. Every run whose numbers might reach the report goes
here — a number without a reproducible config does not go in the report.

Copy this block for a new entry:

```markdown
## EXP-NNN — <short title>
- **Date:** YYYY-MM-DD
- **Who:** name
- **Commit:** <git sha>
- **Notebook / script:** path
- **Hardware:** Colab T4 / CPU / ...
- **Config:** model=, hidden=, lr=, dropout=, residual=, log_transform=, epochs=, seeds=, folds=
- **Question:** what this run was supposed to settle
- **Result:** table (unrounded), CSV committed to `results/`
- **Verdict:** answered / inconclusive / superseded by EXP-NNN
- **Notes:** anything surprising
```

---

## EXP-002 — Baseline v2: rolling-origin CV, residual + log1p
- **Date:** 2026-08-04
- **Who:** Group 05
- **Commit:** _(pre-git — recorded retroactively at repo initialization)_
- **Notebook:** `notebooks/baseline/dengue_baseline_GNN_v2.ipynb` §8–§10
- **Hardware:** Colab
- **Config:** `window=3, horizon=3, cases_idx=5, self_loops=True, residual=True,`
  `log_transform=True, grad_clip=5.0, model∈{GCN,GAT}, hidden=64, gat_heads=8,`
  `dropout=0.1, lr=1e-3, weight_decay=5e-4, epochs=120, patience=25`, 3 folds × 3 seeds
- **Question:** does a spatio-temporal GNN beat naive persistence on this dataset?
- **Result:**

  | Model | RMSE | MAE |
  |---|---|---|
  | Persistence | 44.8 | 15.7 |
  | GCN (residual + log) | 45.3 | 15.9 |
  | GAT (residual + log) | 45.5 | 15.9 |

  Ablation of the v2 refinements (GCN):

  | Setting | RMSE | MAE |
  |---|---|---|
  | Plain GCN (absolute scale) | 66.2 | 31.6 |
  | + log1p only | 67.1 | 27.0 |
  | + residual only | 45.0 | 16.8 |
  | residual + log | 45.3 | 15.8 |

- **Verdict:** answered — the GNN **matches** the persistence floor, does not beat it.
  Residual-over-persistence is what does the work (66 → 45); log1p alone *hurts*, and only
  helps MAE once combined with residual. On the most recent fold the GCN does surpass
  persistence (66.4 vs 68.6), suggesting the gap is regime-dependent.
- **Notes:** This is the honest Phase-1 starting point and it is deliberately reported as such
  in the proposal. It leaves clear headroom for Phases 2–4. Results tables still need to be
  exported to `results/` as CSV.

## EXP-001 — Baseline v1: single 70/10/20 chronological split
- **Date:** 2026-08-04
- **Who:** Group 05
- **Commit:** _(pre-git)_
- **Notebook:** `notebooks/baseline/dengue_baseline_GNN.ipynb`
- **Config:** plain GCN, absolute case scale, single chronological 70/10/20 split
- **Question:** baseline GCN vs persistence under a standard split.
- **Result:** GCN (tuned) test RMSE ≈ 67–72, MAE ≈ 24. Persistence RMSE ≈ 59.5, MAE ≈ 13.8.
  Persistence wins on every metric. In the tuning table, **validation and test RMSE are
  anti-correlated (r = −0.74)**: the config selected on validation (best val 90.9) scored
  test 67.2 — one of the worst — while three configs that would have beaten persistence on
  test (RMSE 53–56) were discarded for poor validation RMSE.
- **Verdict:** superseded by EXP-002.
- **Notes:** The anti-correlation is the important finding, not the loss itself. The 45-week
  validation slice and the test slice sit in different epidemic regimes, so a single split
  cannot support model selection here. This is what forced rolling-origin CV.
