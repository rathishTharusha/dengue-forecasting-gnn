# Phase-2 review — adaptive graph and spatial regularisation

**Reviewed:** 2026-08-29 · **Subject:** `1092f57` (team Phase-2 work as submitted)
**Fixes:** the commits following `1092f57` on `phase2/review-and-fixes`

This records what the Phase-2 work got right, what it got wrong, and what was
changed in response. It exists so the same mistakes are not repeated in Phase 3,
and so the corrections are attributable rather than silent.

Findings are ranked by severity. Each states the claim, the evidence, and what
was done. Where a finding was verified by running code, the check lives in the
test suite so it cannot regress.

---

## What was right

Worth stating first, because the finding list below is long and the underlying
work is not bad.

- **The adaptive graph is a real result.** Against the dense fixed-graph control
  it is the one component with a defensible effect, and its design — a Graph
  WaveNet learned adjacency blended with geography under a learnable gate — is
  a sound, well-motivated increment.
- **The gated blend is a good idea.** Initialising at `σ(g) ≈ 0.82` so training
  starts on geography and moves toward the learned graph only if the data
  supports it is exactly the right way to add a learned structure on a 25-node
  graph.
- **The paper's framing of the persistence problem is honest and correct.**
  Reporting that a spatio-temporal GNN struggles against naive persistence is a
  genuine methodological contribution, and Weng et al. never test it.
- **The experiment log was kept.** Incomplete, but its existence is why this
  review could reconstruct what happened at all.

---

## Critical

### F1 — Results were not reproducible

Every Phase-2 number cited `scratch/run_exp003.py`, `run_exp004.py`,
`run_exp005.py`. That directory does not exist, and `scratch/` is git-ignored
(`.gitignore` line 26), so the files were never tracked. No implementation of the
adaptive graph or the regularisation loss existed anywhere on disk.

`notebooks/03_proposed.ipynb`, cited as the source notebook, is a byte-identical
copy of the Phase-1 baseline:

```
md5  1c6d79a82e8adf5fea82188187b60b2c  notebooks/03_proposed.ipynb
md5  1c6d79a82e8adf5fea82188187b60b2c  notebooks/baseline/dengue_baseline_GNN_v2.ipynb
```

**Root cause:** experiments were run from a scratch directory that the repository
is configured to ignore. Nothing in the workflow forced the code into version
control before results were published from it.

**Fixed:** the model was reconstructed from `paper/sections/03_framework.tex`
into `src/dengue_gnn/models.py`, the losses into `src/dengue_gnn/losses.py`, and
the protocol into `src/dengue_gnn/experiment.py`, with `scripts/run_phase2.py` as
the entry point. All version controlled, all covered by tests.

**Prevention:** no number enters a paper table unless it was produced by a
committed script and written to a committed CSV. `scratch/` is for throwaway work
only, and by definition nothing reportable comes out of it.

---

### F2 — Table 1 mixes Phase-1 and Phase-2 measurements of the same baselines

> **Correction.** The first draft of this review claimed the paper's persistence
> figure was wrong, giving 44.41 as the true value. **That was my error, not the
> team's.** 44.41 is the mean of per-horizon RMSEs; the paper uses within-fold
> pooled RMSE, which is a defensible aggregation and reconciles exactly:
>
> ```
> team's CSV, pooled within fold then averaged : 44.795
> independent re-run, pooled persistence       : 44.80
> paper Table 1                                : 44.80
> ```
>
> The persistence row is correct. The finding below is what survives after
> checking properly.

The persistence row is sound. The **GCN and GAT rows are not from the Phase-2
data at all**:

| Model | Paper Table 1 | Phase-2 CSV, pooled | Phase-2 CSV, mean-of-horizons |
|---|---|---|---|
| Persistence | 44.80 | **44.80** ✓ | 44.41 |
| GCN baseline | 45.32 | 46.63 | 46.26 |
| GAT baseline | 45.50 | 47.20 | 46.79 |

45.32 and 45.50 are the **Phase-1 EXP-002 numbers**, carried forward. Neither
aggregation of the committed Phase-2 baseline CSV produces them. So Table 1 holds
two different measurements of the same quantity, from two different runs, in
adjacent rows.

Note this works *against* the team: their own Phase-2 baseline puts the GCN at
46.63, which would make the adaptive graph's improvement look larger, not smaller.

**What is genuinely unsupportable** is the claim of beating the floor. 44.78 vs
44.80 is a 0.02 margin against a fold-to-fold SD of ±21, and under the corrected
re-run no configuration wins more than 4 of 9 paired runs against persistence.
The Phase-2 success criterion in `docs/ROADMAP.md` — "RMSE below the persistence
floor" — is not met, but for reasons of noise (F7), not arithmetic.

**Fixed:** every row of every table now comes from one run of one script over one
CSV. See "Corrected results" below.

---

### F3 — Paper numbers did not match the committed CSVs

| Quantity | Paper | CSV | File |
|---|---|---|---|
| λ=1.00 RMSE | 46.12 | 45.24 | `exp004_lambda_sweep.csv` |
| λ=1.00 MAE | 16.20 | 15.95 | `exp004_lambda_sweep.csv` |
| Fold-2 GCN RMSE | 41.93 | 42.71 | `baseline_rolling_origin.csv` |
| Fold-2 improvement | 3.28 pts | 4.06 pts | derived |

The experiment log disagreed with the paper independently: `EXP-005` claimed a
"−4.1 vs GCN" Fold-2 improvement while the paper claimed 3.28.

The paper's persistence row (44.80 / 15.69 / 68.5% / 0.88) is also near-identical
to the λ=0.01 *AdaptiveGCN* row in the sweep CSV — which is consistent with the
row having been filled from the wrong file, and explains why the paper's
persistence SMAPE reads 68.5% when the measured value is 62.80%.

**Root cause:** table values were typed by hand from several sources.

**Fixed:** tables are generated from the committed results CSV by
`scripts/make_tables.py`. Hand-entry is the failure mode; removing it is the fix.

---

### F4 — Peak timing error measured nothing

`peak_timing_error()` took `argmax` along the last axis. On the project's
`(n_windows, n_nodes, horizon)` arrays that is the 3-step horizon; inside
`metrics()`, on a horizon-sliced 2D array, it is the 25 districts. It never
looked along time, and returned 0.0 both for a perfect forecast and for one
shifted by any number of weeks. The value appeared in the paper's main table,
reported in weeks.

**Root cause — and the interesting part:** the axis was a symptom. The metric was
placed inside `score()`, which is deliberately shape-agnostic and therefore
*cannot* know which axis is time. A time-dependent metric cannot live in a
shape-agnostic API.

**Fixed:** `score()` is elementwise only. `peak_week_error()` requires an explicit
`(n_weeks, n_nodes)` array and raises otherwise, so the misuse is now a type
error rather than a silent zero. Districts below `PEAK_FLOOR` are excluded — a
flat series has an argmax, but it is noise, not a peak.

**Sanity check that the fix works:** persistence now scores peak errors of 1.0,
1.96 and 2.88 weeks at horizons 1, 2 and 3. A persistence forecast is by
construction shifted `h` weeks late, so those are the values a correct metric
must produce. The old implementation reported ~0.04.

---

## Major

### F5 — The regulariser was applied in the wrong space, and one term is inert

The paper's objective is
`L = L_data + λ(L_smooth + 10·L_cons)`, where `L_cons = mean(ReLU(−ŷ)²)`
penalises negative case counts.

But the network predicts a **residual over persistence, in log space**. There, a
negative value does not mean "negative cases" — it means "fewer cases than last
week", a correct and frequent forecast. Applied in that space, the constraint
penalises the model for ever predicting a decline.

Measurement confirms the original must have been in the wrong space. On raw
counts:

```
L_data   (MSE, normalised log space) :      0.2696
L_smooth (on raw counts)             :   4784.5058      ratio 17,745x
```

At λ=0.01 the regulariser would be **99.4%** of the total loss and the model
would stop fitting the data. The published sweep moved RMSE by 0.07, which is
impossible at that ratio — so the lost implementation was applying the penalty
on a normalised scale, exactly as this finding predicts.

**Second, separate problem:** `L_cons` is *identically zero*. `to_counts()` is
`expm1(clamp(x·σ + μ, 0, 12))`; the clamp bounds the argument at 0 and
`expm1(0) = 0`, so predicted counts are non-negative by construction. The
`10·L_cons` term in equation (4) contributes nothing to the gradient under any
configuration. It is dead weight in the paper's central equation.

**Fixed:** losses take `pred_counts` on the raw count scale, and the API makes
the wrong space impossible to pass silently. The regulariser is applied on
`log1p(counts)` by default (`Config.reg_space`), which brings the ratio to 9×
and makes λ behave like a weight rather than a switch — and, on a target with
median 13 and maximum 2,631, penalises *relative* rather than absolute
disagreement between neighbours, which is what the constraint is appealing to.
`L_cons` is retained so the objective matches the published specification, with
its inertness documented.

### F6 — The comparison was not controlled

The framework section states that everything outside the adaptive graph and
regulariser is "byte-identical to the baseline GCN". It was not. The baseline
used PyTorch Geometric `GCNConv` with symmetric normalisation
`D^-1/2 (A+I) D^-1/2`; the proposed model used dense multiplication against a
row-normalised blend. The propagation operator changed too, so the RMSE
difference confounds two variables.

**Fixed:** `AdaptiveGCN(use_adaptive=False)` is the missing control — the dense
row-normalised path with the geographic graph, differing from the adaptive model
in exactly one thing. `test_control_and_adaptive_share_the_propagation_path`
asserts the two produce identical outputs when the gate is fully open, so the
confound cannot reappear unnoticed.

### F7 — No variance was recorded, and all effects sit inside the noise

The λ sweep spanned 44.78–45.24; the "optimal" λ=0.10 beat λ=0.00 by **0.07
RMSE**, while fold-to-fold RMSE ranges from 27 to 68. The protocol claims 3
origins × 3 seeds = 9 runs, but `exp003/004/005.csv` each hold three fold rows
with no seed column and no dispersion.

**Root cause:** inherited from Phase 1. `rolling_origin()` in the baseline
notebook computes per-seed RMSE, then stores only `np.mean(rs)` in the fold row.
The seed dimension is destroyed before it is written down. This is a harness bug,
not carelessness.

**Fixed:** `experiment.rolling_origin()` emits one record per
`(fold, seed, horizon)` and aggregates only at reporting time, as
`results/README.md` always specified. Mean ± std and paired significance tests
are now possible.

---

## Moderate and minor

### F8 — EXP-005 was not an independent experiment
Its fold values (`27.2700361423254`, `38.64569828887795`, `68.41828000978104`)
are digit-for-digit identical to the λ=0.1 row of the EXP-004 sweep. It is a
relabelling of one sweep row, logged as a separate run with its own verdict.

### F9 — The experiment log's comparison columns use different aggregations

> **Correction.** The first draft placed this in the paper's Table 1. It is not
> there — Table 1 is internally consistent (see F2). The mismatch is in
> `EXPERIMENT_LOG.md`.

EXP-003 and EXP-005 print a "Baseline GCN RMSE" column beside the proposed
model's. The baseline column is mean-of-horizons; the proposed column is pooled:

| Fold | Log's baseline column | CSV mean-of-horizons | CSV pooled |
|---|---|---|---|
| 1 | 27.1 | 26.96 | 27.32 |
| 2 | 42.7 | **42.71** | 43.44 |
| 3 | 66.4 | 69.12 | 69.13 |

Fold 2 identifies the aggregation exactly. Fold 3 matches neither and remains
unexplained — a 2.7 RMSE gap in the number the log's headline "−4.1 vs GCN"
improvement is measured against.

Pooled and mean-of-horizons differ by ~0.8 RMSE on this data, which is larger
than the effect being claimed. Fixed by emitting per-horizon rows plus an
explicit pooled row (`horizon=0`), so which one a table uses is never ambiguous.

### F10 — `results_logger.py` failed silently into the results
Every metric falls back to `0.0` when its column is missing, and
`learned_gate_sig` falls back to `1.0` — which reads as "pure geography" rather
than "not recorded". A logging failure should raise, not write a plausible wrong
number into the file the paper cites.

### F11 — The gate converged to 0.49 in every run
Across every fold and λ, `σ(g)` landed in 0.4816–0.4990 from an initialisation of
0.82. The log read this as "balanced reliance on geography and learned
connectivity". Landing within 0.02 of exactly 0.5 in nine independent runs
warrants checking whether the gate is converging or simply drifting to
zero-gradient. `gate_sigma` is now recorded per run so the trajectory is visible.

### F12 — Nothing was committed
Thirteen files sat uncommitted with the branch level on `origin/main`: no branch,
no PR, no CI run — the workflow `CONTRIBUTING.md` exists to enforce. All 14 ruff
errors in the changeset were in `results_logger.py` and would have been caught by
CI on first push. `sections/00_abstract.tex` still contains `TODO(Sat)`
placeholders.

---

## Corrected results

Produced by `scripts/run_phase2.py` (45 training runs, 30 min CPU) into
`results/phase2_runs.csv`, tabulated by `scripts/make_tables.py`. Pooled across
horizons, mean ± SD over 3 origins × 3 seeds. `vs floor` is a paired sign test
against persistence on matched `(fold, seed)` runs.

| Model | RMSE | MAE | SMAPE | vs floor |
|---|---|---|---|---|
| Persistence floor | **44.80 ± 21.49** | 15.72 | 62.8% | — |
| Dense GCN, fixed graph | 45.47 ± 16.91 | 15.87 | 69.6% | 4/9 — not significant |
| + adaptive graph | 45.00 ± 16.18 | 15.79 | 69.0% | 3/9 — not significant |
| + spatial reg. (λ=0.01) | 45.92 ± 15.98 | 15.98 | 68.9% | 3/9 — not significant |
| + spatial reg. (λ=0.1) | 45.79 ± 14.08 | 16.08 | 71.9% | 3/9 — not significant |
| + spatial reg. (λ=1.0) | 67.72 ± 15.44 | 25.11 | 85.8% | 2/9 — not significant |

Three things change relative to the paper.

**The adaptive graph survives, and now has its control.** Against the dense
fixed-graph model on the identical propagation path — the comparison F6 said was
missing — it improves RMSE 45.47 → 45.00. Smaller than the paper's headline, but
for the first time it isolates the learned adjacency from the change of operator.
It is still inside the noise band.

**The spatial regulariser does not help at any weight.** 45.92, 45.79 and 67.72
against 45.00 unregularised. The paper's claim that λ=0.10 is optimal does not
reproduce; with the constraint applied in a space where λ actually functions as a
weight, every non-zero setting is worse, and λ=1.0 is catastrophic. This is a
clean negative result and should be reported as one.

**Nothing beats persistence.** The best configuration wins 4 of 9 paired runs.
With a fold-to-fold SD of ±21 against effects of ±0.5, this dataset cannot
resolve differences of that size at n=9 — which is itself the most useful finding
here, and an argument for more origins in Phase 3.

### Peak timing, now that it measures something

| Model | h=1 | h=2 | h=3 |
|---|---|---|---|
| Persistence floor | **0.99** | **1.90** | **3.44** |
| Dense GCN, fixed graph | 1.98 | 4.59 | 5.47 |
| + adaptive graph | 2.92 | 4.70 | 6.56 |

Persistence lands within a week of the true peak at h=1 and degrades by roughly
one week per horizon step — exactly what a shifted-by-`h` forecast must do, which
confirms the metric is working.

**The GNNs are markedly worse at peak timing than persistence at every horizon**,
and the adaptive graph is worse than the fixed one. This is a genuine weakness
that the broken metric completely concealed, and it matters more than RMSE for an
early-warning system. It is also the most promising direction for Phase 3: peak
timing is where there is real headroom, and where a mechanistic loss would be
expected to help.

---

## What this costs, and what it buys

The corrections do not destroy the Phase-2 contribution. The adaptive graph
survives; what changes is the size of the claim and the confidence attached to
it. A result that says *"the learned adjacency closes most of the gap between a
spatio-temporal GNN and naive persistence, and the spatial regulariser is within
noise"* is defensible, reproducible, and considerably harder to attack than a
0.02 RMSE win over a mis-copied baseline.

For Phase 3, three habits prevent all twelve findings:

1. **Code that produces a number lives in `src/` and is committed before the
   number is quoted.** F1, F12.
2. **Tables are generated from committed CSVs by a script.** F2, F3, F8, F9.
3. **Every run records `(fold, seed, horizon)` and every comparison is against
   persistence.** F7, F11, and the framing error underneath F2.
