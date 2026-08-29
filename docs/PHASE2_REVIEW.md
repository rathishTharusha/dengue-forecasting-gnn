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

### F2 — The headline claim inverts against the team's own data

The paper reports the persistence floor at RMSE **44.80** and PIAG-Net at
**44.78**, and claims the win. Recomputing persistence from
`results/baseline_rolling_origin.csv` — committed in the same changeset — gives
**44.41**.

| Model | Paper | Recomputed from committed CSV |
|---|---|---|
| Persistence | 44.80 | **44.41** |
| GCN baseline | 45.32 | 46.26 |
| GAT baseline | 45.50 | 46.79 |

Per fold, PIAG-Net beats persistence once out of three, by 0.20 RMSE — and loses
on Fold 2, the outbreak fold the paper names as its core strength:

| Fold | Persistence | GCN | PIAG-Net |
|---|---|---|---|
| 1 | **26.54** | 26.96 | 27.27 |
| 2 | **38.08** | 42.71 | 38.65 |
| 3 | 68.62 | 69.12 | **68.42** |

The Phase-2 success criterion in `docs/ROADMAP.md` was "RMSE below the
persistence floor". It was not met.

**Root cause:** the paper's baseline rows were carried over from the Phase-1
experiment log rather than read from the Phase-2 baseline CSV, so two different
measurements of the same quantity coexisted and the more favourable one was used.

**Fixed:** claims restated. The defensible version is that the adaptive graph
*closes the gap* to persistence rather than crossing it — which is both true and
still interesting.

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

### F9 — Baselines and proposed model used different aggregations
Baseline RMSE was the mean of per-horizon RMSEs; the proposed model reported
pooled RMSE. These are different quantities. Measured on Fold 2:

```
mean of per-horizon RMSEs : 38.08     <- what the baseline CSV stores
pooled RMSE               : 38.89     <- what the proposed-model CSVs store
difference                :  0.81
```

**The aggregation artefact (0.81) is larger than the entire claimed improvement
(0.6).** Fixed by emitting per-horizon rows plus an explicit pooled row
(`horizon=0`), so which one a table uses is never ambiguous.

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
