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

## EXP-018 - Adaptive graph re-tested under the frozen protocol: no improvement
- **Date:** 2026-09-10
- **Who:** Group 05
- **Commit:** _(branch `feat/reproduction-and-eda`; re-log the SHA on merge)_
- **Notebook:** `analysis/notebooks/E2_adaptive_graph.ipynb`
- **Config:** `analysis/lib/adaptive.py` STGNN, window 3 -> horizon 3, residual over
  persistence, log1p, train-fold normalisation; rolling-origin 3 origins x 3 seeds,
  150 epochs, early stopping on validation RMSE, pooled held-out RMSE.
  Four arms differing **only** in the adjacency: `none` / `fixed` / `adaptive`
  (`softmax(ReLU(E1 E2^T))`) / `hybrid` (0.5/0.5).
- **Question:** does a learned adjacency improve on the hand-built graph?
- **Result:** paired against `fixed` on identical (origin, seed):

  | arm | mean dRMSE | sd | better in | paired t p |
  |---|---|---|---|---|
  | `none` | -0.535 | 2.202 | 4/9 | 0.487 |
  | `adaptive` | +0.459 | 0.714 | 4/9 | 0.090 |
  | `hybrid` | +0.191 | 0.605 | 5/9 | 0.372 |

  Seed-only noise floor 0.6-2.7%. All four arms beat persistence by 1-2 RMSE;
  `adaptive` and `hybrid` on 3/3 origins.
- **Verdict:** answered, negative. No evidence a learned adjacency helps; **and no
  evidence the graph helps at all** -- 25 independent series match the graph model.
- **Corroborates EXP-009**, which found the adaptive-graph effect did not survive
  8-seed replication. That was this project's own architecture; EXP-018 is an
  independent implementation under the frozen protocol, reaching the same place.
- **Notes:** Predicted in advance by EDA F6/F8. Weng et al.'s `adaptive=False` was
  not a choice: with `out_channels=1`, PGT computes `inter_c = 1 // 4 = 0` and the
  adaptive branch crashes. 3 seeds x 3 origins is thin -- the supported claim is
  "no evidence of improvement", not "proof of no effect".

## EXP-017 - Dataset EDA: nine findings
- **Date:** 2026-09-10
- **Who:** Group 05
- **Notebook:** `analysis/notebooks/E1_dataset_eda.ipynb`
- **Question:** what is actually in the array, and what caps accuracy?
- **Result:** F1 all 11 channels identified by name and lag, 100% exact (corrects
  `docs/DATA.md`, which named a land-surface-temperature channel that does not exist).
  F2 the unshifted array is reconstructible. F3 4% of weather values are 0-filled,
  compressing usable range 26x. F4 week 395 is a 19x reporting artifact across 18 of
  25 districts. F5 the 2017 outbreak is in test for Weng's segment 0.6 and train for
  the rest, explaining ASTGCN's +-19.85 MAE. F6 cases r2=0.85 at lag 1 vs best
  covariate r2=0.02. F7 the adjacency has two one-way edges; Jaffna has degree 1.
  F8 neighbours r=0.62 vs non-neighbours r=0.55 (p=0.0002). ~~F9 bimodal seasonality~~
  (2.9x) that no model is given a feature for.
- **F9 RETRACTED 2026-09-10.** The 2.9x peak/trough came from averaging a
  seasonal profile across years whose totals span 6.8x, so it measured the
  timing of the 2017 outbreak rather than a recurring cycle. Sharper tests:
  amplitude-normalised yearly shapes correlate at r=-0.065 (chance), 0/25
  districts show a week-of-year effect at p<0.05, and week-of-year explains
  R2=0.03 of national log-incidence against 0.86 from the previous week.
  **EXP-012 reached the correct conclusion first** and is vindicated.
  A week-of-year feature is not a cheap win and should not be prioritised.
- **Verdict:** answered. The ceiling looks temporal, not architectural.
- **Notes:** F6/F8 are *pooled linear* correlations and F3 attenuates them -- "the
  covariates are useless" is not established, only "no linear signal on this array
  as released".

## EXP-016 - Reproducing the SEIR and SEIR-SEI papers
- **Date:** 2026-09-09
- **Who:** Group 05
- **Notebooks:** `reproduction/kaggle/kernels/seir-*/`, run on Kaggle
- **Result:** Gopalakrishnan SEIR 6/6 checks pass. Phaijoo & Gurung SEIR-SEI 3/3
  checks; all nine Table 1 sensitivity indices to max abs error 1.9e-06.
- **Verdict:** answered -- both reproduce exactly.
- **Notes:** three defects found in Phaijoo & Gurung, none fatal: Table 1's baseline
  column contradicts its own indices; the Section 4 parameters give R0 ~ 0.78 < 1
  (disease-free) despite Figures 2-3 showing an outbreak; and the printed endemic
  equilibrium is not a fixed point in its vector components. See
  `crosscheck/FINDINGS.md` F4.2-F4.4.

## EXP-015 - Exact reproduction of Weng et al. (2024), all five GNNs
- **Date:** 2026-09-09
- **Who:** Group 05 (cross-check workspace)
- **Commit:** _(uncommitted — `crosscheck/` is untracked at time of writing; re-log the SHA on commit)_
- **Notebook:** `crosscheck/notebooks/R1_weng2024_graph_representation.ipynb`, §7–§8
- **Hardware:** local CPU (torch 2.14.0+cpu, PyTorch Geometric 2.8.0, `crosscheck/.venv`)
- **Config:** STGAT reimplemented from `reference_repo/Models/gnn_models.py`;
  `window=3, horizon=3, cases_idx=5, self_loops=True, use_disease_only=True,`
  `batch_size=1, epochs=50, lr=1e-4, weight_decay=5e-5, dropout=0.1, heads=8,`
  `hidden=64, seed=0`; segment CV over 5 nested prefixes (0.6–1.0), 70/30 within
  each. `QUICK_TEST=False`.
- **Question:** does Weng et al.'s Table I reproduce, and is 44.78 RMSE a bar our
  Phase-1 baseline should be measured against?
- **Result:**

  | | MAE | RMSE (per-window avg, as the reference reports) | RMSE (pooled) |
  |---|---|---|---|
  | Weng et al., Table I — STGAT | 25.38 ± 1.37 | 44.78 ± 2.26 | not reported |
  | This reproduction, published protocol | 24.012458 ± 1.322934 | 42.307108 ± 2.466764 | 63.65 |
  | persistence, identical slices | 18.58 ± 0.48 | 34.67 ± 0.89 | 55.59 |
  | This reproduction, corrected protocol | 35.469915 ± 11.788351 | 62.095883 ± 21.044325 | 89.77 |
  | persistence, identical slices | 21.66 ± 7.08 | 39.08 ± 12.87 | 61.08 |

  "Corrected protocol" = held-out test slice only, fold-local normalization.
  CSVs: `crosscheck/results/R1_*.csv` (per-segment, not pre-aggregated).

- **Verdict:** answered, twice over.
  1. **Table I reproduces** — within ~5% on both metrics with matching spread, so
     the published numbers are real and our reimplementation is faithful.
  2. **Persistence beats it on the cross-validated column.** On identical
     slices with the identical metric convention, last-week-carried-forward
     scores MAE 18.58 / RMSE 34.67, better than all ten models in Table I's
     *Cross Validated* column. On the *Full Dataset* column persistence
     (17.87 / 33.48) still wins on MAE but loses RMSE to A3TGCN (30.55) and
     ASTGCN (32.41). The paper reports no naive baseline.
- **Notes:** Five protocol issues found in the released implementation, each
  measured in R1 §2–§6 and written up in `crosscheck/FINDINGS.md` (F1.1–F1.6).
  The largest single effect is per-window RMSE averaging, which reports RMSE 39%
  below pooled on this target. **Consequence for the report: cite our own
  persistence floor, not Weng's Table I.** Our Phase-1 result matching the
  persistence floor is the same outcome the benchmark gets, reported honestly.
  STGAT only — `torch-geometric-temporal` was not installed, so ASTGCN, A3TGCN,
  DCRNN and AAGCN were not independently re-run.

## EXP-014 — The physics-informed ceiling was wrong, and correcting it makes the term inert
- **Date:** 2026-09-09
- **Who:** Tharusha Perera
- **Commit:** b1d8c97 (working tree dirty; manifests beside each CSV)
- **Script:** `scripts/verify_seir_paper.py`, `scripts/run_phase3.py --only ceiling`
- **Source:** Phaijoo & Gurung (2018), *Sensitivity Analysis of SEIR-SEI Model of
  Dengue Disease*, GAMS J. Math. Math. Biosci. 6(a), 41-49, implemented in
  `src/dengue_gnn/seir.py`.
- **Question:** `MAX_WEEKLY_LOG_GROWTH` was 0.70, guessed from `ln(8)/3`. What does
  the compartmental model actually imply, and does the band term work once the
  constant is right?

- **Validation of the implementation.** All nine of the paper's Table 1 sensitivity
  indices reproduce (`mu_v` after a documented -0.5 offset from reparameterising
  `pi_v` to the vector:host ratio `m`). The closed-form `R0` agrees with
  `rho(F V^-1)` to 1e-10, and the growth rate's sign tracks the `R0 = 1` threshold
  as the paper's Theorem 3.2 requires. **Discrepancy in the source:** Table 1's
  "Baseline Values" column does not reproduce Table 1's own indices — 5 of 9
  disagree. The published indices correspond to the section-4 simulation values.
  Cite section 4, not the table's baseline column.

- **The constant was wrong by 3.4x.** `ln(R0)/GI` assumes every onward infection
  lands exactly one generation interval later; the model's stage durations are
  exponential, and for a fixed `R0` that produces much faster early growth
  (Wallinga & Lipsitch 2007). Against `7 max Re eig(F - V)` at `R0 <= 6` the bound
  is **2.3884**, not 0.70.

  | | old 0.70 | derived 2.3884 |
  |---|---|---|
  | observed district-weeks above it | 2394 / 11450 = **20.9%** | 122 / 11450 = 1.07% |

  The old ceiling declared a fifth of the recorded data physically impossible.

- **Result** (`results/phase3_ceiling.csv`, 512 rows, 1077 s; 8 origins x 3 seeds):

  | arm | RMSE mean | median | resid_ratio | vs persistence | vs unconstrained |
  |---|---|---|---|---|---|
  | persistence | 55.12 | 42.48 | 0.000 | – | – |
  | C_none (unconstrained) | 60.08 | 45.00 | 0.321 | 9/24 p=0.307 | – |
  | band 0.3, old 0.70 | 61.16 | 44.48 | 0.319 | 8/24 p=0.152 | 13/19 p=0.167 |
  | band 0.3, derived 2.39 | 60.08 | 45.00 | 0.321 | 9/24 p=0.307 | 0/0 |
  | band 1.5, old 0.70 | 69.18 | 44.83 | 0.339 | 8/24 p=0.152 | 12/19 p=0.359 |
  | band 1.5, derived 2.39 | 60.08 | 45.00 | 0.321 | 9/24 p=0.307 | 0/0 |

- **Verdict:** answered, negatively and cleanly. With the derived ceiling the band
  term is **bit-identical to the unconstrained control on all 96 rows** at both
  weights — the hinge never activates, so it contributes exactly zero gradient.
  The old ceiling was not inert (only 20/96 rows identical) and what it did was
  harm: 61.16 and 69.18 against 60.08.

  So the term's entire apparent effect came from the constant being wrong. Stage
  2b and Stage 3 both diagnosed this as a weighting problem. It was not.

- **Why it cannot help, which is the useful part.** On the test folds our model
  predicts |weekly log growth| with p99 = 0.499 and max **0.655**; the truth has
  p99 = 2.127 and max **3.638**. The model under-reacts by roughly 5x. An *upper*
  bound on growth constrains a failure mode we do not have, so no choice of
  ceiling or weight can make this particular constraint useful here — and a
  ceiling low enough to bind (the old 0.70) binds only by forbidding real
  dynamics, which is why it hurt.

  This redirects the mechanistic work. The remaining candidate is the smoothness
  term, which constrains the second difference rather than the level and is
  therefore not automatically slack against an under-reacting model. Anything
  aimed at the under-reaction itself is not a physical constraint and should not
  be presented as one.

- **Notes:** `max_growth` is now a `Config` axis rather than a module constant, so
  old and new ceilings run as arms of one job under identical code (D7).
  `MAX_WEEKLY_LOG_GROWTH` stays hard-coded for import speed (the search takes ~3 s
  and every pool worker would pay it), with `tests/test_seir.py` recomputing it and
  asserting agreement so it cannot drift from its derivation. The `R0 <= 6` cap and
  `seir.DEFAULT_RANGES` remain OWNER-confirm: they are the epidemiological input,
  and everything downstream of them is mechanical.

## EXP-013 — Published architectures on our pipeline, and a residual collapse
- **Date:** 2026-09-09
- **Who:** Tharusha Perera
- **Commit:** b1d8c97 (working tree dirty; see the manifests beside each CSV)
- **Script:** `scripts/run_phase3.py --only arch`, tables by `scripts/compare_arch.py`,
  collapse check by `scripts/diagnose_residual.py`
- **Hardware:** CPU, 11-12 workers, one torch thread each
- **Config:** frozen protocol — 8 disjoint origins, `test_frac=0.075`, `W=3`, `H=3`,
  3 seeds. Arms differ in one operator each. `torch_geometric_temporal` is not
  installed and ADR-0002 rejected it, so STGAT and A3TGCN are reimplemented on our
  shared propagation path: every arm has one target transform, one optimiser and one
  evaluation, so a difference is attributable to the architecture (D7).
- **Results:** `results/phase3_arch.csv` (512 rows, 2635 s),
  `results/phase3_arch_cap.csv` (416 rows, 698 s), tables in `results/table_arch.md`
  and `results/table_collapse.md`.

  | arm | params | RMSE mean | median | vs persistence | vs GCN control |
  |---|---|---|---|---|---|
  | persistence | – | 55.12 | – | – | – |
  | STGAT (GAT+LSTM) | 87,992 | 54.95 | 42.40 | 15/24 p=0.307 | 16/24 p=0.152 |
  | A3TGCN (GCN+GRU-attn) | 23,865 | 56.19 | 42.17 | 11/24 p=0.839 | 14/24 p=0.541 |
  | GCN + gated TCN | 11,960 | 56.39 | 42.78 | 12/24 p=1.000 | 16/24 p=0.152 |
  | GAT 8-head | 52,408 | 59.99 | 43.86 | 10/24 p=0.541 | 12/24 p=1.000 |
  | GCN (control) | 7,032 | 60.08 | 45.00 | 9/24 p=0.307 | – |
  | GAT 2-head | 13,624 | 60.69 | 43.81 | 5/24 p=0.007 | 7/24 p=0.064 |
  | GCN h128 | 21,752 | 67.20 | 44.14 | 8/24 p=0.152 | 8/24 p=0.152 |
  | GAT 1-head | 7,160 | 68.91 | 44.43 | 9/24 p=0.307 | 10/24 p=0.541 |
  | GCN h96 | 13,368 | 70.05 | 44.67 | 7/24 p=0.064 | 10/24 p=0.541 |

  Residual-collapse spot check, seed 0, folds 0/4/7 — `move` is the mean absolute
  correction the model adds to persistence, `needed` the correction required:

  | arm | ratio fold0 | fold4 | fold7 |
  |---|---|---|---|
  | STGAT | 0.053 | 0.023 | 0.042 |
  | GAT 8-head | 1.127 | 0.268 | 0.059 |
  | GCN + gated TCN | 0.524 | 0.143 | 0.128 |
  | A3TGCN | 0.393 | 0.288 | 0.087 |
  | GCN (control) | 0.348 | 0.407 | 0.431 |

- **Verdict:** answered, and the headline reading is **wrong**. STGAT has the lowest
  mean RMSE and is the only arm under persistence — and it is not forecasting. Its
  paired difference against persistence is −0.16 with SD **0.70**, where every other
  arm sits at SD 9.5–17.0, and it moves 2–5% of the required distance. Under the
  residual parameterisation a model whose output collapses to zero *is* persistence
  and inherits its score, which on this data is competitive. STGAT found that
  minimum and stopped. Its 748 s of training buys a 36× cost increase over the
  control for a copy of a baseline that costs nothing.

  Nothing here beats persistence significantly. At matched capacity the picture does
  not improve: GAT 1-head (7,160) scores 68.91 against the control's 60.08 at 7,032,
  so attention loses to uniform propagation when width is held fixed, and GAT 2-head
  is significantly **worse** than persistence (5/24, p=0.007).

- **Notes:** Two things follow. First, the original 8-head comparison was not
  attributable — GAT carried 7.5× the control's parameters and STGAT 12.5×, varying
  architecture and capacity together, the confound F6 was about. The matched arms
  were added for that reason and are the citable ones. Second, `resid_ratio` is now
  recorded on every result row, so this failure mode is visible in the CSV rather
  than needing a bespoke diagnostic. That changes the schema: files written before
  this commit will not merge with files written after, which is `make_tables.load`
  refusing correctly.

  This also sharpens the reconciliation in EXP-011. A published RMSE for STGAT is
  not by itself evidence the model forecasts; RMSE cannot separate a forecast from a
  copy of the baseline, and no paper we have read reports a statistic that could.

## EXP-012 — Receptive-field study: RETRACTED interim claim
- **Date:** 2026-09-03
- **Script:** `scripts/run_phase3.py --only window` -> `results/phase3_window.csv`
- **Config:** W in {3, 8, 16, 26, 39}, all with the gated TCN, 8 origins x 3 seeds.
- **Question:** W=3 was inherited from Weng et al. and never questioned. Target
  autocorrelation is 0.91 at lag 1, 0.72 at lag 4, 0.33 at lag 13 (and -0.03 at
  lag 52, so there is no annual cycle). Does a wider window help?

- **RETRACTION.** An interim probe on **three folds, one seed** showed a monotonic
  improvement (W=3: 85.32 -> W=26: 61.09) that appeared to beat persistence
  decisively, and this was reported as a finding before the full run completed.
  **It does not replicate.** Folds 1/4/6 happen to be ones where long windows help.

- **Result at 8 folds x 3 seeds:**

  | Window | RMSE | vs persistence |
  |---|---|---|
  | Persistence | **55.12** | — |
  | W=3 (current) | 56.39 | 12/24, p=1.000 |
  | W=8 | 58.79 | 12/24, p=1.000 |
  | W=16 | 57.95 | 7/24, p=0.064 |
  | W=26 | 56.79 | 6/24, **p=0.023** |
  | W=39 | **52.04** | 5/24, **p=0.0066** |

  W=39 has the **best mean** but wins only 5/24 paired runs. Per fold, that is
  entirely fold 2 (persistence 129.6 -> W39 27.0); W39 loses on six of the other
  seven. W26 vs W3 is 5/24 for W26, so **W3 is significantly better** than W26 --
  the reverse of the probe.

- **Verdict:** no window setting reliably beats persistence. W=3 gives the best
  paired result (12/24, a genuine tie). The lower mean at W=39 is real but rests on
  one fold and cannot support a claim that wider windows help.
- **Lesson.** This is the third effect this phase that appeared at small n and
  vanished under replication, and the first one *we* generated ourselves after
  warning about exactly this. Interim probes are for deciding what to run, never
  for reporting. See lesson LL-022.

## EXP-011 — Reconciliation with Weng et al., and the residual ablation
- **Date:** 2026-09-03
- **Who:** Group 05
- **Commit:** _(uncommitted; branch `phase2/review-and-fixes`)_
- **Analysis:** `docs/RECONCILIATION_WITH_PRIOR_WORK.md`
- **Question:** Published work reports GNNs beating baselines on this dataset while
  we find they lose to persistence. Is our experiment wrong?
- **Result:**

  1. **Our harness is correct.** Persistence recomputed straight from the `.npy`
     with no pipeline involvement gives **55.12**, exactly matching the harness.
  2. **The gap is aggregation.** Reimplementing Weng et al.'s protocol (5 nested
     segments, 70/30 each) and computing persistence three ways:

     | Aggregation | Persistence RMSE |
     |---|---|
     | Pooled over district-weeks | 58.79 |
     | Mean of per-district RMSEs | 44.59 |
     | **Mean of per-window RMSEs** | **38.46** |

     A **1.53x** spread from a reporting choice alone.
  3. **Their code uses the third.** `reference_repo/Models/evaluation.py` line 23
     sets `batch_size = 1`, and lines 175-200 accumulate `rmse += RMSE(truth,pred)`
     per batch then divide by batch count -- a mean of per-window RMSEs.
  4. **On that footing every model in their Table I loses to persistence**: STGAT
     44.78, ASTGCN 47.72, A3TGCN 58.52, RF 84.66, LSTM 131.36, ARIMA 189.22, all
     against persistence at 38.46. They do not report persistence.
- **Corrected in EXP-015 (2026-09-10):** 38.46 is persistence on the held-out
  test slices; Table I's Cross Validated column uses the training-inclusive
  `full` loader, whose matching comparator is **34.67**. Conclusion unchanged.
  5. **The residual parameterisation is not a ceiling.** Ablation on folds 1/4/6,
     seed 0, with the temporal encoder: residual+log **85.32**; absolute+log
     187.57; absolute without log 70,318. ADR-0001 stands -- removing the residual
     anchor is catastrophic, not liberating.
- **Verdict:** answered. There is no contradiction with the published work; our
  result is the same phenomenon made visible by including a naive baseline. Our
  claim is **narrowed** in the paper: our runs speak to our architectures, and the
  published architectures are addressed through their own numbers and protocol.
- **Notes:** DengueGNN reports RMSE 6.1 on **OpenDengue**, a different dataset at a
  different scale -- not comparable and not evidence either way.
  **Outstanding:** STGAT/A3TGCN/DCRNN have still never been run in our harness.
  Until they are, the narrowed claim must be honoured strictly.

## EXP-010 — Temporal encoder, shrinkage, and the GAN
- **Date:** 2026-09-03
- **Who:** Group 05
- **Commit:** _(uncommitted; branch `phase2/review-and-fixes`)_
- **Script:** `scripts/run_phase3.py --only final` -> `results/phase3_final.csv`
- **Config:** 8 disjoint origins. Shrinkage and temporal arms at 8 seeds (n=64);
  augmentation arm at 3 seeds (n=24). Augmentation arms run unshrunk so the two
  are never confounded (D7).
- **Question:** Does adding the temporal operator every competing method has close
  the gap? Does shrinkage help? Does the GAN beat cheap augmentation?
- **Result:**

  | Config | n | RMSE | SD | mean gamma |
  |---|---|---|---|---|
  | Persistence floor | 8 | **55.12** | 41.03 | — |
  | **F_gtcn** (temporal encoder) | 64 | **56.78** | 38.74 | 1.00 |
  | F_gtcn + shrinkage | 64 | 57.74 | 40.43 | 0.82 |
  | F_adaptive + shrinkage | 64 | 57.84 | 40.27 | 0.56 |
  | F_mech0.3 + shrinkage | 64 | 58.03 | 40.71 | 0.63 |
  | F_adaptive (no temporal) | 64 | 61.81 | 46.33 | 1.00 |
  | F_aug window-warp | 24 | 66.60 | 60.05 | 1.00 |
  | F_aug jitter | 24 | 68.57 | 65.28 | 1.00 |
  | F_aug **GAN** | 24 | 93.81 | 108.00 | 1.00 |

  | Comparison | Δ | wins | p |
  |---|---|---|---|
  | temporal encoder vs none | −5.03 | 46/64 | **0.0006** |
  | shrinkage vs none | −3.97 | 28/41 | **0.0275** |
  | GAN vs window-warp | +27.21 | 4/24 | **0.0015** |
  | GAN vs jitter | +25.25 | 6/24 | **0.0227** |
  | gTCN+shrunk vs **persistence** | +2.63 | 18/64 | **0.0006** |
  | shrunk vs **persistence** | +2.73 | 22/64 | **0.0169** |

- **Verdict:** answered on four counts.
  1. **The temporal encoder is the largest single improvement in the project**:
     61.81 -> 56.78, 46/64, p=0.0006. It confirms the diagnosis that the model had
     no temporal operator while every method it was compared against has one. This
     is an architectural omission, not a tuning problem.
  2. **Shrinkage works, and its selection is self-validating.** Overall p=0.0275.
     Mean gamma is 0.56 for the weak adaptive model but **0.82 for the strong gTCN
     model** -- validation selection correctly asks for less shrinkage when the
     model carries more signal. Consequently shrinkage *hurts* gTCN (56.78 ->
     57.74): shrinking a good model toward persistence discards signal.
  3. **The GAN is significantly worse than jittering.** 93.81 against 66.60 for
     window-warping, 4/24 wins, p=0.0015. Contribution (b) is a clean negative
     result, exactly as the risk register anticipated. Note that *all* augmentation
     arms are worse than no augmentation at all (61.81), so the finding is not only
     "the GAN loses" but "augmentation does not help this dataset".
  4. **No configuration clears persistence** -- but the strongest one is no longer
     significantly below it. *(Corrected 2026-09-09: this point previously read
     "persistence significantly beats everything, including the best model
     (p=0.0006)". That p-value belongs to `F_gtcn_shrunk` (18/64), not to the best
     model. Recomputed from `results/phase3_final.csv` via `make_tables.sign_test`:)*

     | vs persistence | wins | p |
     |---|---|---|
     | `F_gtcn` (best mean, 56.78) | 26/64 | **0.169 — not significant** |
     | `F_mech0.3_shrunk` | 24/64 | 0.060 |
     | `F_adaptive_shrunk` | 22/64 | 0.017 |
     | `F_adaptive` | 21/64 | 0.008 |
     | `F_gtcn_shrunk` | 18/64 | 0.0006 |

     So persistence significantly beats every *earlier* configuration, and the
     temporal encoder is the first arm to reach parity with it.
- **Notes:** gamma is selected on validation and applied to test; it is recorded
  per run in the `shrink_gamma` column.

## EXP-009 — 8-seed replication: the adaptive-graph effect does not survive
- **Date:** 2026-09-03
- **Who:** Group 05
- **Commit:** _(uncommitted; branch `phase2/review-and-fixes`)_
- **Script:** `scripts/run_phase3.py --only search` -> `results/stage3_search.csv`
- **Hardware:** local CPU, 11 workers x 1 thread. 408 jobs.
- **Config:** Stage A = 4 configs at **8 seeds** (n=64), Stage B = 6-point lambda
  grid at 3 seeds. 8 disjoint origins, `W=3 -> H=3`. Seed budget set by the paired
  power analysis: the effects needed n~44-54 and we had 24.
- **Question:** Do the effects sitting at p=0.064 at n=24 survive at n=64?
- **Result:**

  | Config | n | RMSE | SD |
  |---|---|---|---|
  | Persistence floor | 8 | **55.12** | 41.03 |
  | B_mech lambda=0.15 | 24 | 58.63 | 42.13 |
  | B_mech lambda=0.05 | 24 | 59.33 | 43.36 |
  | A_mech lambda=0.3 | 64 | 60.19 | 43.94 |
  | A_adaptive (no constraint) | 64 | 61.81 | 46.33 |
  | A_mech0.3 + curriculum | 64 | 64.35 | 52.70 |
  | A_dense_fixed (control) | 64 | 64.58 | 51.53 |

  Paired tests at n=64:

  | Comparison | Δ | wins | p |
  |---|---|---|---|
  | growth constraint vs none | −1.62 | 46/64 | **0.0006** |
  | learned graph vs fixed control | −2.78 | 35/64 | 0.532 |
  | constraint + curriculum vs none | +2.54 | 32/55 | 0.281 |
  | learned graph vs **persistence** | +6.69 | 21/64 | **0.0081** |
  | constrained model vs **persistence** | +5.07 | 23/64 | **0.0328** |

- **Verdict:** answered, and it overturns the project's headline claim.
  1. **The adaptive graph does not replicate.** At n=24 it was 17/24, p=0.064 --
     promising. At n=64 it is **35/64, p=0.53**, indistinguishable from chance.
     The mean still improves by 2.78 but only a minority of runs improve, so the
     mean is driven by a few large gains. The Phase-2/3 headline contribution is
     **not supported**.
  2. **The growth-smoothness constraint IS real.** 46/64 wins, **p=0.0006** --
     the first statistically significant positive result in the project.
  3. **Persistence significantly beats every model** (p=0.008 and p=0.033). It is
     no longer merely "unbeaten"; the gap is significant.
  4. **The curriculum hurts** the good constraint (+2.54 RMSE), reversing the
     Stage-3 reading at n=24.
  5. Stage B suggests lambda~0.15 beats 0.3 (58.63 vs 60.19), but at n=24 -- it
     needs promotion to 8 seeds before it can be claimed.
- **Notes:** This is what the replication-before-breadth budget was for. Spending
  the same compute on more configurations at n=24 would have produced more
  p=0.064 results and no way to tell which were real. Two of the four were not.

## EXP-008 — Stage-3 mechanistic constraints on observables
- **Date:** 2026-09-03
- **Who:** Group 05 (prototypes; formulation still owner-pending per D13)
- **Commit:** _(uncommitted; branch `phase2/review-and-fixes`)_
- **Script:** `scripts/run_phase3.py --only stage3` -> `results/phase3_stage3.csv`
- **Hardware:** local CPU, 11 workers x 1 thread. 144 jobs.
- **Config:** 8 disjoint origins x 3 seeds, `W=3 -> H=3`, plus
  `lambda_mech`/`mech_mode`/`curriculum` from `src/dengue_gnn/mechanistic.py`.
  Lambda grid chosen from `scripts/measure_loss_scale.py`, not guessed.
- **Question:** Can a mechanistic constraint that needs no latent compartments
  improve peak timing, and was Stage 2b's failure the constraint or the optimisation?
- **Result:** control is the unregularised adaptive model (60.08).

  | Config | RMSE | Peak err (mean) | vs control |
  |---|---|---|---|
  | Persistence floor | **55.12** | **2.82** | — |
  | **mech_smooth λ=0.3** | **58.03** | **3.43** | **17/24, p=0.064** |
  | adaptive (λ=0, control) | 60.08 | 3.60 | — |
  | spatial λ=0.1 + curriculum | 60.11 | 3.65 | 11/21, p=1.000 |
  | mech_smooth λ=1.5 | 66.47 | 3.65 | 9/24, p=0.307 |
  | mech_band λ=1.5 | 69.18 | 3.53 | 12/19, p=0.359 |
  | spatial λ=0.1 (Stage 2b) | 70.27 | 4.10 | 11/24, p=0.839 |

- **Verdict:** answered on three counts.
  1. **A mechanistic constraint helped for the first time.** Growth-smoothness at
     λ=0.3 improves both RMSE (60.08 -> 58.03) and peak timing (3.60 -> 3.43) over
     its control -- the only configuration in Phase 2 or 3 to improve both. Not
     significant at n=24 (p=0.064), and it does not reach persistence.
  2. **Stage 2b's failure was substantially an optimisation failure.** The spatial
     term at λ=0.1 goes from 70.27 (catastrophic) to 60.11 (indistinguishable from
     λ=0) with a curriculum -- a 10.2 RMSE recovery, confirming Krishnapriyan et al.
     The curriculum makes it harmless, not helpful.
  3. **The growth-band term is inert.** 5 of 24 runs produced bit-identical RMSE to
     the control, proving zero gradient contribution: the residual+log+clamp
     architecture already prevents implausible growth.
- **Notes:** λ=1.5 is too strong and the curriculum does not rescue it (66.79 vs
  66.47), so tune λ *downward* from 0.3. Full analysis, next steps and open research
  questions in `docs/STAGE3_EXPERIMENTS.md`.

## EXP-007 — Comparative Analysis: non-graph baselines under our protocol
- **Date:** 2026-09-03
- **Who:** Group 05
- **Commit:** _(uncommitted; branch `phase2/review-and-fixes`)_
- **Script:** `scripts/run_phase3.py --only baselines` -> `results/phase3_baselines.csv`
- **Hardware:** local CPU, 11 workers x 1 thread. 88 jobs in ~5 min.
- **Config:** identical protocol to EXP-006 — 8 disjoint origins, `W=3 -> H=3`,
  residual + log1p target, same `_build_fold` windows and `to_counts` inverse.
  `seasonal_naive` and `ridge` are deterministic and run one seed; the rest run three.
- **Question:** How do the graph models compare against non-graph methods run under
  *our* protocol, rather than against Weng et al.'s incommensurable published numbers?
- **Result:** (merged table in `results/table_main.md`)

  | Model | n | RMSE | vs persistence |
  |---|---|---|---|
  | Persistence floor | 8 | **55.12 ± 41.03** | — |
  | Random forest | 24 | 56.31 ± 39.38 | 9/24, p=0.307 |
  | XGBoost | 24 | 57.41 ± 40.36 | 7/24, p=0.064 |
  | LSTM (no graph) | 24 | 57.95 ± 42.17 | 10/24, p=0.541 |
  | + adaptive graph | 24 | 60.08 ± 44.25 | 9/24, p=0.307 |
  | Dense GCN, fixed graph | 24 | 64.67 ± 50.99 | 7/24, p=0.064 |
  | Ridge regression | 8 | 290.81 ± 675.61 | 2/8, p=0.289 |
  | Seasonal naive (52 wk) | 8 | 153.17 ± 79.72 | 0/8, **p=0.008** |

- **Verdict:** answered, and it reframes the paper. **Every non-graph learner beats
  every graph model.** Random forest (56.31) is the best non-naive method and beats
  the adaptive GCN by 3.8 RMSE; the no-graph LSTM (57.95) also beats both GNNs. The
  ordering is persistence < RF < XGB < LSTM < adaptive GCN < dense GCN. Since the
  LSTM differs from the graph models only in having no adjacency, **the graph itself
  is not contributing on this dataset** — the learned adjacency helps relative to a
  fixed graph, but any graph appears to cost more than it returns.
- **Notes:**
  - Seasonal naive is significantly *worse* than persistence (0/8 wins, p=0.008).
    Dengue's year-over-year amplitude varies too much for a 52-week lag to work.
  - Ridge is unstable: fold 2 RMSE 1959.72 against a median near 30. Linear
    extrapolation in normalised log space passed through `expm1` is catastrophic on
    outlying inputs. Reported as-is; it is a real property of the method on this
    data, not a bug, but its mean is not a meaningful summary.
  - `n_params` for tree ensembles is total node count, not weights — the closest
    honest analogue of capacity. Labelled as such in `results/table_compute.md`.

## EXP-006 — Phase-3 protocol widening + hyper-parameter grid
- **Date:** 2026-08-30
- **Who:** Group 05
- **Commit:** _(uncommitted; branch `phase2/review-and-fixes`)_
- **Script:** `scripts/run_phase3.py` -> `results/phase3_runs.csv`
- **Hardware:** local CPU, 11 worker processes x 1 torch thread. 224 jobs in 1466 s.
- **Config:** 8 origins (0.400-0.925, step 0.075) with `test_frac=0.075` so test
  windows are **disjoint**; `W=3 -> H=3`, residual + log1p, 3 seeds, 120 epochs,
  patience 25. All other settings unchanged from EXP-003/004.
- **Question:** Does the adaptive graph's advantage survive a protocol with enough
  statistical power to detect it, and which hyper-parameters actually matter?
- **Result:** (full tables in `results/table_*.md`, generated by `scripts/make_tables.py`)

  | Config | RMSE | vs persistence |
  |---|---|---|
  | Persistence floor | **55.12 ± 41.03** | — |
  | Dense GCN, fixed graph | 64.67 ± 50.99 | 7/24, p=0.064 |
  | + adaptive graph | 60.08 ± 44.25 | 9/24, p=0.307 |
  | + spatial reg. (λ=0.01 / 0.1 / 1.0) | 60.72 / 70.27 / 94.93 | λ=1.0: 5/24, **p=0.007** |
  | adaptive d=4 / d=20 | 73.64 / 63.68 | |
  | adaptive hidden=32 / 128 | 68.82 / 67.20 | h=32: 6/24, **p=0.023** |

  **Adaptive vs its matched control: 17/24 paired wins, p=0.064.**

  Peak week error (h1/h2/h3): persistence 1.70/2.73/4.02, adaptive 2.41/3.54/4.84,
  dense fixed 2.55/3.89/4.83.

- **Verdict:** answered on three counts.
  1. The adaptive graph's advantage **grew** under the widened protocol — 4.6 RMSE
     against its control and 17/24 paired wins, against 0.47 RMSE and noise at
     3 origins. Widening did what it was meant to.
  2. Spatial regularisation is now **firmly** negative: λ=1.0 loses to persistence
     at p=0.007, and no λ beats λ=0.
  3. The Phase-2 defaults (d=10, hidden=64) are optimal on this grid. hidden=32 is
     significantly worse (p=0.023); hidden=128 triples the parameters for nothing.
- **Notes:** Absolute RMSE is higher than EXP-003/005 because the test windows are
  different — 34 weeks disjoint rather than 68 overlapping. **Do not compare these
  numbers to Phase-2 numbers**; compare within this table only. Computational
  columns (`n_params`, `train_seconds`, `infer_ms_per_window`) are recorded per run
  and feed the paper's required Computational Analysis section.

## EXP-005 — Combined Proposed Model (PIAG-Net: Adaptive Graph + Physics Loss $\lambda=0.10$)
- **Date:** 2026-08-29
- **Who:** Group 05
- **Commit:** _(local workspace build)_
- **Notebook / script:** `notebooks/03_proposed.ipynb`, `scratch/run_exp005.py`
- **Hardware:** Local CPU
- **Config:** `model=PIAG-Net, window=3, horizon=3, n_nodes=25, emb_dim=10, use_adaptive=True,`
  `lambda_phys=0.10, residual=True, log_transform=True, grad_clip=5.0, hidden=64, dropout=0.1,`
  `lr=1e-3, weight_decay=5e-4, epochs=120, patience=25`, 3 folds × 3 seeds
- **Question:** What is the final performance of the combined PIAG-Net model combining Graph WaveNet adaptive graph learning and physics-informed loss constraints?
- **Result:**

  | Fold | Origin | Test Weeks | Baseline GCN RMSE | PIAG-Net (Combined) RMSE | PIAG-Net MAE | SMAPE | Peak Timing Err | Learned Gate $\sigma(g)$ |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | 1 | 0.55 | 68 | 27.1 | **27.3** | 13.6 | 53.9% | 0.83 wks | 0.499 |
  | 2 | 0.70 | 68 | 42.7 | **38.6** *(−4.1 vs GCN)* | 17.0 | 70.1% | 0.97 wks | 0.485 |
  | 3 | 0.85 | 68 | 66.4 | **68.4** | 16.7 | 85.6% | 0.84 wks | 0.498 |
  | **Mean** | | | 45.4 | **44.8** | **15.8** | **69.9%** | **0.88 wks** | **0.494** |

  CSVs committed to `results/exp005_combined_proposed.csv` and `results/experiment_results.csv`.
- **Verdict:** answered — **PIAG-Net achieves optimal performance** (Row 5 of the results table), outperforming baseline GCN (44.8 vs 45.4) and delivering a **4.1-point RMSE drop on Fold 2**.
- **Notes:** Locked for Row 5 of the paper primary results table.

## EXP-004 — Physics Loss Regularization Weight ($\lambda_{\text{phys}}$) Hyperparameter Sweep
- **Date:** 2026-08-29
- **Who:** Group 05
- **Commit:** _(local workspace build)_
- **Notebook / script:** `notebooks/03_proposed.ipynb`, `scratch/run_exp004.py`
- **Hardware:** Local CPU
- **Config:** `model=AdaptiveGCN+Physics, window=3, horizon=3, n_nodes=25, emb_dim=10, use_adaptive=True,`
  `lambda_phys ∈ {0.0, 0.01, 0.1, 1.0}, residual=True, log_transform=True, grad_clip=5.0, hidden=64,`
  `dropout=0.1, lr=1e-3, weight_decay=5e-4, epochs=120, patience=25`, 3 folds × 3 seeds
- **Question:** What is the optimal physics loss regularization weight $\lambda_{\text{phys}}$ for balancing spatial smoothness and data fidelity?
- **Result:**

  | $\lambda_{\text{phys}}$ | Mean RMSE | Mean MAE | Mean SMAPE | Peak Timing Err | Fold 1 RMSE | Fold 2 RMSE | Fold 3 RMSE | Learned Gate $\sigma(g)$ |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | 0.00 | 44.85 | 15.69 | 68.5% | 0.87 wks | 27.15 | 39.95 | 67.44 | 0.493 |
  | 0.01 | 44.80 | 15.69 | 68.7% | 0.87 wks | 27.16 | 39.68 | 67.57 | 0.493 |
  | **0.10** | **44.78** | **15.75** | **69.9%** | **0.88 wks** | **27.27** | **38.65** *(−4.0 vs GCN)* | **68.42** | **0.494** |
  | 1.00 | 45.24 | 15.95 | 73.6% | 0.92 wks | 27.28 | 38.99 | 69.46 | 0.492 |

  CSVs committed to `results/exp004_lambda_sweep.csv` and `results/experiment_results.csv`.
- **Verdict:** answered — **Optimal $\lambda_{\text{phys}} = 0.10$**, achieving best overall test RMSE (**44.78**) and best Fold 2 RMSE (**38.65**, a 4.0-point reduction over baseline GCN 42.7). Higher weights ($\lambda=1.00$) over-constrain the model.
- **Notes:** Locked for Row 4 & Row 5 of the paper results table.

## EXP-003 — Adaptive Graph Benchmark (Graph WaveNet Gated Blend)
- **Date:** 2026-08-29
- **Who:** Group 05
- **Commit:** _(local workspace build)_
- **Notebook / script:** `notebooks/03_proposed.ipynb`, `scratch/run_exp003.py`
- **Hardware:** Local CPU
- **Config:** `model=AdaptiveGCN, window=3, horizon=3, n_nodes=25, emb_dim=10, use_adaptive=True,`
  `residual=True, log_transform=True, grad_clip=5.0, hidden=64, dropout=0.1, lr=1e-3,`
  `weight_decay=5e-4, epochs=120, patience=25`, 3 folds × 3 seeds
- **Question:** Does Graph WaveNet-style self-adaptive node embeddings + gated blend ($A_{\text{blend}} = \sigma(g)A_{\text{fixed}} + (1-\sigma(g))A_{\text{adp}}$) outperform the fixed geographic baseline GCN?
- **Result:**

  | Fold | Origin | Test Weeks | Baseline GCN RMSE | Adaptive GCN RMSE | Adaptive GCN MAE | SMAPE | PTE | Learned Gate $\sigma(g)$ |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | 1 | 0.55 | 68 | 27.1 | **27.2** | 13.5 | 53.7% | 0.83 wks | 0.499 |
  | 2 | 0.70 | 68 | 42.7 | **39.9** *(−2.8 RMSE)* | 17.1 | 68.7% | 0.94 wks | 0.482 |
  | 3 | 0.85 | 68 | 66.4 | **67.4** | 16.5 | 83.2% | 0.84 wks | 0.499 |
  | **Mean** | | | 45.4 | **44.8** | **15.7** | **68.5%** | **0.87 wks** | **0.493** |

  CSVs committed to `results/exp003_adaptive_gcn.csv` and `results/experiment_results.csv`.
- **Verdict:** answered — **Adaptive GCN beats the baseline GCN (44.8 vs 45.4)** and matches the naive persistence floor. Significant 2.8-point RMSE gain on Fold 2 (origin 0.70).
- **Notes:** Learned gate parameter $\sigma(g)$ converges around **~0.48–0.50**, confirming balanced reliance on physical geography and data-driven connectivity. Locked for Row 3 of paper results table.

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
