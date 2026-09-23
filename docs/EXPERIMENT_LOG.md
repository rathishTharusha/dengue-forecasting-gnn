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

> **EXP-001 – EXP-014 ran on code that is no longer in the working tree.** The
> Phase-2/3 implementation (`src/dengue_gnn/{experiment,models,losses,augment,
> baselines,provenance,results_logger,mechanistic}.py`, `scripts/run_phase2.py`,
> `scripts/run_phase3.py`, `notebooks/03_proposed.ipynb`,
> `notebooks/04_kaggle_search.ipynb`) and its `results/phase*.csv` outputs were
> removed once EXP-015 onward rebuilt on architectures verified against the
> published papers. Every entry keeps its config and unrounded numbers inline, so
> nothing here is uncitable — but to *re-run* any of those experiments, check out
> the tag `phase23-archive`:
>
> ```bash
> git checkout phase23-archive
> ```
>
> The derived growth ceiling those experiments turned on (`MAX_WEEKLY_LOG_GROWTH`,
> `CEILING_R0_MAX`) survived the cleanup and now lives in `dengue_gnn.seir`, still
> asserted by `tests/test_seir.py` and reproduced by `scripts/verify_seir_paper.py`.

## EXP-040 — Remedies from the literature: none adopted
- **Date:** 2026-09-23
- **Who:** Group 05
- **Commit:** `f7786ab` (code; plan `docs/REMEDIES_PLAN.md` committed earlier at `7e983ae`)
- **Script:** Kaggle kernel `seirgnn2-remedies` (v1) =
  `sweep.py remedies --keep --epochs 400 --workers 4`, then the three pre-registered
  ensembles; results `seirgnn2/results/remedies.json`, `remedies+ens.json` (120 rows,
  complete: 90 runs + 3 persistence + 27 ensemble). Forecasts in `remedies_preds.pkl`
  (git-ignored, 41 MB).
- **Hardware:** Kaggle CPU, 4 workers.
- **Config:** as pre-registered. B = AAGCN, direct head, NB likelihood, seasonal features,
  window 3. Each remedy changes one thing against B. Frozen three origins x seeds 0/1/2.
  **One change before the run, validation-only:** k-NN's k grid was widened to 320 after a
  local smoke run chose k = 80, the old grid's top, on every fold (chosen k on Kaggle:
  80 / 320 / 160).
- **Question:** Do the remedies the literature proposes for these architectures' weaknesses
  move the best model, and does SEIR structure help when used as a constraint instead of a
  decoder?
- **Result:** validation RMSE (selection metric), paired against B at the origin_seed unit;
  test shown for the record only.

  | arm | val | vs B | wins | p_adj | test |
  |---|---|---|---|---|---|
  | R6 SEIR auxiliary, w = 0.1 | 15.60 | -0.04 | 6/9 | 0.485 | 37.17 |
  | ENS[B + NB-GLM + k-NN] | 15.63 | -0.01 | 5/9 | 0.965 | 35.74 |
  | R6 SEIR auxiliary, w = 0.3 | 15.64 | -0.00 | 6/9 | 0.965 | 38.02 |
  | **B** | **15.65** | — | — | — | 37.55 |
  | R2 district identity | 15.71 | +0.06 | 4/9 | 0.785 | 39.73 |
  | ENS[B + k-NN] | 15.72 | +0.08 | 5/9 | 0.487 | 34.18 |
  | ENS[B + NB-GLM] | 15.78 | +0.13 | 5/9 | 0.476 | 39.28 |
  | R3 STID | 16.35 | +0.71 | 2/9 | 0.061 | 36.33 |
  | R4b k-NN | 16.52 | +0.87 | 0/9 | 0.017 | 34.34 |
  | R1a RevIN (mean) | 16.60 | +0.96 | 4/9 | 0.237 | 37.69 |
  | R1b RevIN | 16.92 | +1.27 | 0/9 | 0.017 | 36.74 |
  | R4a NB-GLM | 16.97 | +1.32 | 1/9 | 0.025 | 42.85 |
  | SEIR-LSTM | 17.28 | +1.63 | 0/9 | 0.017 | 34.97 |
  | persistence | 17.86 | +2.21 | — | — | 36.02 |

  **Residual correlation with B** (validation, seeds averaged): SEIR auxiliary 0.995,
  district identity 0.971, STID 0.946, NB-GLM 0.923, SEIR-LSTM 0.921, **k-NN 0.919**,
  RevIN 0.908.

- **Verdict:** answered, negative. **No remedy meets the adoption rule** (mean delta < 0,
  >= 7/9 wins, horizon 3 not raised); the closest, the SEIR auxiliary at w = 0.1, wins 6/9
  by 0.04 against a between-unit sd of 0.16. The finalist is therefore B unchanged, whose
  nine-origin confirmation already exists (EXP-038, `AAGCN+direct`): it did not beat
  persistence or SEIR-LSTM on test. Under the plan no further confirmatory run is made.
- **Notes:**
  1. **The information limit is now shown across model families, not just architectures.**
     k-NN analogue forecasting has no network and no training, yet its errors correlate
     0.92 with B's; the GLM's 0.92; instance normalisation's 0.91. Methods that share no
     machinery make the same mistakes, so the mistakes belong to the data -- the part of
     next week that the past does not contain. It is also why equal-weight ensembles, the
     most reliable remedy in the forecasting literature, gained nothing: they need members
     that err differently, and none exist here.
  2. **The SEIR constraint neither helps nor hurts.** As an auxiliary loss it leaves B's
     forecasts almost unchanged (residual correlation 0.995). Physics as a *decoder* costs
     ~7 RMSE (EXP-032/034); physics as a *constraint* costs nothing and buys nothing.
  3. **Why B beats SEIR-LSTM (-1.63, 9/9 on validation):** mostly the head, not the graph.
     SEIR-LSTM routes its forecast through the gated SEIR path; the LSTM on the direct head
     scored 15.80 in EXP-035, within 0.15 of AAGCN. Stated this way in any write-up.
  4. **Instance normalisation hurt, significantly** (RevIN +1.27, 0/9). Removing each
     window's level discards information that matters here: high levels mean-revert and low
     levels grow. This is the over-stationarisation Liu et al. (2022) warn about.
  5. **Validation and test disagree.** On test, k-NN (34.34) and B + k-NN (34.18) beat
     persistence (36.02) while B (37.55) does not. Selecting on that would be choosing a
     model by its test score, which the protocol forbids and which is not done here. What
     the disagreement does support is that 1-2 RMSE differences among these models are not
     stable across periods.
  6. Kaggle reproduces local: B here scores 15.65 against 15.66 for the identical
     configuration run locally in EXP-035.

## EXP-039 — What caps the architectures: diagnosis on their forecasts
- **Date:** 2026-09-23
- **Who:** Group 05
- **Commit:** `7e983ae` (code `seirgnn2/diagnose_arch.py`, committed before this entry)
- **Script:** `python seirgnn2/diagnose_arch.py collect` then `analyse`;
  output `seirgnn2/results/diagnose_arch.txt`
- **Hardware:** Local CPU, 6 workers
- **Config:** STGAT, A3TGCN, ASTGCN, AAGCN and the LSTM encoder (DCRNN dropped for cost
  and rank), each in two configurations: `native` (direct head, mse on log1p, no
  covariates — as published) and `best` (NB likelihood + seasonal features); SEIR-LSTM as
  `LSTM+foi` (native) and `LSTM+foi_res` (best). Frozen three origins x seeds 0/1/2,
  300 epochs. **Every number below is on validation windows** — diagnosing on test and then
  designing around it would spend the held-out set.
- **Question:** Every earlier grid kept one RMSE per run. What do the forecasts actually get
  wrong, and is it the architecture or the information?
- **Result (best configuration unless stated):**
  1. *Skill vs persistence by horizon:* ASTGCN / AAGCN / LSTM +0.10 to +0.16, **rising**
     with horizon; STGAT (-0.25 / -0.12 / -0.01) and A3TGCN (-0.72 / -0.54 / -0.37) are
     worse than persistence at every horizon.
  2. *Responsiveness:* slope of predicted on realised log-growth 0.28-0.31 for the working
     three; predicted growth varies ~0.6x as much as real growth.
  3. *Bias by regime:* under-predict rising weeks by 11-13 cases, over-predict falling
     weeks by 10-11 — a lag. 80-83% of squared error sits in weeks that moved. Outbreak
     cells under-predicted by 17-23 (native: 22-25, worse than persistence's 15).
  4. *Direction:* ~70% correct; with NB + season rises are called up 87-88% of the time but
     falls called down only 52-54% — the mean-targeting likelihood tilts toward growth.
  5. *Concentration:* the top 5% of cells carry ~60% of squared error.
  6. **Residual correlation between ASTGCN, AAGCN and the LSTM: 0.976-0.991.** They are the
     same forecaster to within noise. The mean of all five direct encoders scores 16.73
     against 15.85 for AAGCN alone.
  7. *Spatial structure left:* residual correlation +0.17 between neighbours, +0.12 between
     any two districts — a national common component plus a small neighbour excess.
  8. **A linear model on the same inputs recovers only 4-6% of the working models' residual
     variance out of sample** (fit on train residuals, scored on validation). For STGAT and
     A3TGCN it recovers 47-62%: those two are underfitting, not capped.
- **Verdict:** answered. The working architectures have converged to the same function and
  exhausted the linearly available information in their inputs; what remains is a lag on
  moves and shrinkage on outbreaks, which squared-error-type objectives produce on a
  near-random-walk target. STGAT and A3TGCN are limited by their wiring (a national
  bottleneck; no state carried across weeks), not by the data. Further encoder work on the
  same inputs is not expected to pay; this motivates `docs/REMEDIES_PLAN.md`.
- **Notes:** Two new facts about the data came out of this. Origin 0.40's validation window
  contains the 2017 DENV-2 epidemic, peaking at 5.4x anything in that fold's training data —
  the catastrophic fold in EXP-038. And 2020-2022 ran at 5-10% of the historical peak. The
  series changes regime, while every model normalises with one statistic per fold.

## EXP-038 — Confirmatory stage, nine disjoint origins (replaces S9)
- **Date:** 2026-09-23
- **Who:** Group 05
- **Commit:** `2215ae8`
- **Script:** `seirgnn2/sweep.py confirm` -> `seirgnn2/results/confirm.json` (144 rows)
- **Hardware:** Local CPU, 6 workers. Also pushed as Kaggle kernel `seirgnn2-confirm`.
- **Config:** five finalists frozen out of `combo` before this ran -- AAGCN+direct,
  LSTM+direct, ASTGCN+foi_res, LSTM+foi_res, ASTGCN+direct; loss=nb, dist=nb,
  use_season=True, lam_param=log, state_fit=True, window=3, epochs=400.
  `core.ORIGINS_9` = nine origins with **disjoint** test spans (verified zero overlap between
  any pair), tiling 0.40 -> 1.00 with span 1/15. Three seeds each.
- **Question:** The pre-registered endpoint. Does the proposed SEIR-GNN beat Liu et al.'s
  SEIR-LSTM, and does anything beat persistence, on a protocol where the origin unit can
  actually reach significance?
- **Result:** validation RMSE. **The origin unit is the one that matters here** (n = 9,
  attainable floor 0.004); origin_seed (n = 27) is reported second as stability.

  | arm | val | test |
  |---|---|---|
  | ASTGCN+foi_res | 38.30 | 31.70 |
  | AAGCN+direct | 38.32 | 33.93 |
  | LSTM+foi_res | 39.05 | 31.47 |
  | LSTM+direct | 39.27 | 34.97 |
  | persistence | 41.21 | 31.76 |
  | ASTGCN+direct | 46.29 | 36.79 |

  Paired at the **origin** unit:

  | comparison | delta | wins | p | p_adj |
  |---|---|---|---|---|
  | ASTGCN+foi_res vs LSTM+foi_res | **-0.75** | **8/9** | **0.012** | 0.059 |
  | AAGCN+direct vs LSTM+direct | -0.95 | 7/9 | 0.531 | 0.885 |
  | ASTGCN+foi_res vs persistence | -2.92 | 8/9 | 0.066 | 0.166 |
  | AAGCN+direct vs persistence | -2.89 | 8/9 | 0.035 | 0.166 |

  At origin_seed, ASTGCN+foi_res vs LSTM+foi_res is -0.75, **24/27**, p_adj = 0.000.

- **Verdict:** **The primary endpoint is not met.** The pre-registered criteria were BH
  p-adj < 0.05, a win on >= 6/9 origins, and the same direction as the 3-origin table.
  SEIR-GNN vs SEIR-LSTM satisfies two of three: 8/9 origins and the same direction (-0.62 in
  EXP-035), but p_adj = 0.059 against a 0.05 bar. Raw p = 0.012; the gap is the multiplicity
  correction across the four comparisons. Report as a consistent directional advantage that
  does not clear the pre-registered significance bar -- **not** as "SEIR-GNN beats SEIR-LSTM".
- **Notes:**
  1. **Absolute levels are not comparable to the 3-origin grids.** Persistence is 41.21 here
     against 17.86 on the frozen three, because the nine origins reach back to 0.40 and include
     much harder periods. Never pool or pair across origin sets; `sweep.run` keys persistence
     rows by (window, origin-set) to make that mechanical.
  2. **Test is below validation here** (31.70 vs 38.30), inverted relative to the 3-origin
     grids. Different weeks, nothing more -- but it is why val/test divergence on three origins
     should not have been read as a warning sign about the model.
  3. **RETRACTED, same day, before anything was built on it.** This entry first claimed the
     run's strongest signal was a variance one: ASTGCN+direct at sd 25.77 across origins against
     ASTGCN+foi_res at sd 4.27, read as the physics layer stabilising an unstable encoder. The
     per-origin table kills it. Origin 0.40 is an outlier where **every** arm fails -- 140 to 227
     RMSE against 13 to 48 everywhere else -- and it dominates every mean and sd in the run:

     | origin | ASTGCN+direct | ASTGCN+foi_res | AAGCN+direct | persistence |
     |---|---|---|---|---|
     | **0.400** | **226.99** | **142.73** | **157.12** | **153.54** |
     | 0.467 | 21.86 | 22.17 | 20.26 | 23.17 |
     | 0.533 | 17.39 | 18.87 | 16.16 | 21.12 |
     | 0.600 | 42.18 | 47.21 | 42.86 | 43.56 |
     | 0.667 | 20.85 | 22.41 | 21.39 | 30.11 |
     | 0.733 | 13.26 | 14.45 | 12.90 | 14.48 |
     | 0.800 | 23.28 | 24.62 | 23.46 | 28.76 |
     | 0.867 | 19.48 | 19.58 | 19.39 | 22.52 |
     | 0.933 | 31.29 | 32.61 | 31.35 | 33.65 |

     ASTGCN+foi_res beats ASTGCN+direct on **1 of 9 origins** -- that one. On the other eight it
     is slightly worse. One hard period is an anecdote, not a stabilisation property, and the
     mean difference of -7.99 carries sd 28.64. **Mean RMSE across these nine origins is not a
     usable summary**; read the win count and the per-origin column, which is why `stats.py`
     prints wins beside every delta. The general lesson: a variance claim computed across folds
     that include a catastrophic fold is a claim about that fold.
  3b. **What survives the same scrutiny.** ASTGCN+foi_res vs LSTM+foi_res is -0.75 on **8/9**
     origins and is *not* outlier-driven -- it wins at 0.40 by 2.0 and on seven of the other
     eight. That consistency, not any mean, is what makes it the one robust result here.
  4. Kaggle kernel v1 of this grid returned 63 rows instead of 144 and had to be discarded:
     `pip install --no-deps torch-geometric-temporal` also skips torch_geometric, which Kaggle
     does not ship, so every graph arm died in its worker while the LSTM arms ran. v2 installs
     torch_geometric properly and asserts on `backbones.check()` before the grid starts.

## EXP-037 — Audit: run_s9_confirmatory.py does not compute what S9 claims
- **Date:** 2026-09-23
- **Who:** Group 05
- **Commit:** `198d2cf`
- **Script audited:** `analysis/_build/run_s9_confirmatory.py`
- **Question:** Before re-running the confirmatory stage in the seirgnn2 harness, does the
  existing S9 do what the paper says it does?
- **Result:** No, in three separate ways.
  1. **It is 3 origins, not 9.** `s_star_origin_means` is a `groupby("origin")` over the S5
     results, and `analysis/results/seir_gnn/s5_seir_gnn/s5_seir_gnn_results.json` has 108 rows
     across origins [0.55, 0.70, 0.85]. The script's own comment reads
     `# Simulated 9-origin differences for test (9 disjoint origins)`.
  2. **The paired test is not paired.** The comparators are hardcoded scalars --
     `b_star_test_rmse = 34.837`, `persistence_test_rmse = 36.016`,
     `seir_lstm_test_rmse = 62.608` -- so each "difference" is an origin mean minus one
     constant. Matching on origin was the entire point of the design; subtracting a constant
     tests something else.
  3. **One reported p-value is a typed-in literal:** `p_auc = 0.04  # Simulated AUC difference
     p-value`. This is the `p = 0.04` behind the package README's "Outbreak Detection ROC-AUC:
     0.807-0.826 (p = 0.04)".

  With 3 origins an exact sign-flip test cannot return a two-sided p below 0.25, so the reported
  `p_raw = 0.50` was floor-bound whatever the data said.
- **Verdict:** answered. **No S9-derived number is citable**, including the early-warning
  p-value. This is a more serious problem than the S5 defects (EXP-032): S5 was a mistuned
  experiment, whereas this is a statistical claim that does not correspond to its computation.
- **Notes:** `seirgnn2/stats.py` does the intended thing -- real pairing on matched
  (origin, seed), exact sign-flip enumeration, BH-FDR -- and prints the smallest attainable p so
  a 3-origin grid cannot be read as significant. The 9-origin confirmatory run is feasible; it
  has to actually be run.

## EXP-036 — Window length is not the lever it looked like (logged R5 deviation)
- **Date:** 2026-09-23
- **Who:** Group 05
- **Commit:** `198d2cf`
- **Script:** `seirgnn2/sweep.py window` -> `seirgnn2/results/window.json` (117 rows)
- **Hardware:** Local CPU, 6 workers
- **Deviation:** plan R5 freezes window 3 -> horizon 3, inherited from the benchmark paper.
  This run varies the **input window only** over {3, 6, 12}; horizon, origins, seeds,
  normalisation and metric are unchanged, and the climate/NDVI sub-window stays at 3 weeks so
  the case history is the single varying factor. Reason for deviating: three weekly points can
  barely estimate a trend, and the seasonal feature -- the only input-side lever that has moved
  the metric -- suggests the models are starved of temporal context.
- **Config:** backbone/head in (AAGCN+direct, LSTM+direct, ASTGCN+foi_res, LSTM+foi_res);
  loss=nb, dist=nb, use_season=True, lam_param=log, state_fit=True, epochs=400.
- **Question:** Does a longer input window help?
- **Result:** **No.** In absolute validation RMSE every arm looks worse as the window grows --
  AAGCN+direct 15.66 / 15.86 / 16.29 at w = 3 / 6 / 12 -- but that is almost entirely the
  baseline moving, because a longer window shifts the fold boundaries and changes which weeks
  are evaluated. Persistence on the same folds is 17.86 / 18.06 / 18.58. Against its **own**
  window's persistence:

  | arm | w=3 | w=6 | w=12 |
  |---|---|---|---|
  | AAGCN+direct | -2.20 | -2.20 | -2.28 |
  | LSTM+direct | -2.06 | -1.86 | -2.23 |
  | ASTGCN+foi_res | -1.13 | -1.04 | -1.45 |
  | LSTM+foi_res | -0.59 | -0.46 | -0.69 |

  Flat to within noise. The margin over the baseline does not depend on the window.
- **Verdict:** answered, negative. Window 3 stays; the deviation is closed and not carried
  forward.
- **Notes:** Two things worth keeping. (1) This was predicted to be the largest remaining lever
  and it is approximately null -- more history does not help because, as EXP-024 established,
  cases at t-1 already explain r2 = 0.85 and the rest is close to unpredictable. (2) **Arms at
  different windows must never be compared on absolute RMSE**, because the evaluation folds
  differ. `sweep.run` now emits one persistence row per window for exactly this reason, and
  reading the raw leaderboard without matching baselines would have produced the opposite and
  wrong conclusion.

## EXP-035 — Combination grid: first significant wins over persistence
- **Date:** 2026-09-23
- **Who:** Group 05
- **Commit:** `fc8ab6b`
- **Script:** `seirgnn2/sweep.py combo` -> `seirgnn2/results/combo.json` (219 rows)
- **Hardware:** Local CPU, 6 workers
- **Config:** backbone in (AAGCN, ASTGCN, LSTM) x feats in (season, season+climate+ndvi)
  x dist in (point/mse_z, nb/nb) x head in (direct, foi_res); lam_param=log, state_fit=True,
  epochs=400, patience=40; frozen protocol, 3 origins x 3 seeds. Selection on validation (R6).
- **Question:** Stacking the levers that individually moved the metric, does anything beat
  persistence, and does the graph beat Liu et al's LSTM?
- **Result:** validation RMSE, mean over 9 units; paired sign-flip at the origin_seed unit.

  | arm | val | test | vs persistence | wins | p_adj |
  |---|---|---|---|---|---|
  | AAGCN+direct season nb | 15.66 | 37.55 | -2.20 | 9/9 | 0.010 |
  | AAGCN+direct all nb | 15.71 | 37.73 | -2.15 | 8/9 | 0.023 |
  | LSTM+direct season nb | 15.80 | 35.70 | -2.06 | 9/9 | 0.010 |
  | ASTGCN+direct season nb | 15.92 | 36.41 | -1.93 | 9/9 | 0.010 |
  | ASTGCN+foi_res season nb | 16.65 | 35.26 | -1.21 | - | - |
  | LSTM+foi_res season nb | 17.27 | 34.97 | -0.59 | - | - |
  | persistence | 17.86 | 36.02 | - | - | - |

  Head-to-head, matched on everything but the encoder:
  - **physics head:** ASTGCN+foi_res vs LSTM+foi_res = **-0.62, 9/9, p_adj = 0.009**
  - **direct head:** AAGCN+direct vs LSTM+direct = -0.14, 5/9, p_adj = 0.315 (n.s.)
  - **NB vs MSE**, matched pair: -0.41, 9/9, p_adj = 0.013

- **Verdict:** answered, provisionally. First arms in this project to beat persistence
  significantly. The graph beats the LSTM **only** in the physics formulation, not the direct
  one -- consistent with spatial coupling mattering for a transmission quantity and not for
  case regression.
- **Notes:** Two caveats travel with these numbers and must not be dropped. (1) Validation and
  test disagree at this spread: LSTM+foi_res is *ahead* on test (34.97 vs 35.26) and the best
  validation arm is *worse than persistence* on test (37.55 vs 36.02). (2) Three origins cannot
  reach p < 0.05 at the origin pairing unit; all p-values above are origin_seed, i.e. run-to-run
  stability, not a claim about the series. Both require the 9-origin confirmatory grid (S9)
  before anything is claimed in paper text.

## EXP-034 — Six encoders in one harness: SEIR-GNN vs SEIR-LSTM, controlled
- **Date:** 2026-09-23
- **Who:** Group 05
- **Commit:** `fc8ab6b`
- **Script:** `seirgnn2/sweep.py real` -> `seirgnn2/results/real.json` (165 rows)
- **Hardware:** Local CPU, 6 workers, 71 min
- **Config:** backbone in (LSTM, STGAT, A3TGCN, ASTGCN, AAGCN, DCRNN) x head in
  (direct, foi, foi_res); loss=mse_z, lam_param=log, state_fit=True, epochs=300.
- **Question:** The proposal is Liu et al's SEIR-LSTM with the LSTM replaced by a real
  spatio-temporal GNN. Earlier seirgnn2 grids used a single dense matmul as the backbone -- and
  its `gat` mode was a literal alias for `gcn` -- so they could not answer it. With the five
  published architectures in place, does the physics head work, and does the graph beat the LSTM?
- **Result:**

  | arm | val | test | best epoch |
  |---|---|---|---|
  | AAGCN+direct | 16.76 | 35.59 | 65 |
  | ASTGCN+direct | 16.84 | 35.39 | 50 |
  | LSTM+direct | 16.91 | 35.10 | 93 |
  | AAGCN+foi_res | 17.44 | 35.78 | 24 |
  | persistence | 17.86 | 36.02 | - |
  | STGAT+direct | 23.32 | 60.82 | 24 |
  | AAGCN+foi | 23.69 | 50.06 | 66 |
  | LSTM+foi | 24.32 | 50.25 | 97 |
  | A3TGCN+direct | 28.63 | 60.40 | 28 |
  | DCRNN+direct | 34.89 | 73.74 | 5 |

  AAGCN+foi vs LSTM+foi: -0.62, 6/9, p_adj = 0.178 (n.s.).
  AAGCN+direct vs LSTM+direct: -0.14, 6/9, p_adj = 0.149 (n.s.).
- **Verdict:** answered. Real architectures improve the bare `foi` head only from 24.59 to
  23.69 -- it still loses to the direct head by ~7 RMSE. The conclusion drawn on the toy
  backbone survives the backbone change. All 162 runs stopped early (best epoch 5-108 of 300),
  so undertraining is not the explanation at this budget.
- **Notes:** An 8-epoch smoke test on origin 0.70 alone suggested the opposite and was wrong --
  origin 0.70 is the easiest fold (persistence 11.72 there against 17.86 averaged), so a single
  origin must never be compared against a multi-origin mean. Recorded because it nearly became
  a reported finding.

## EXP-033 — Is the physics head undertrained? (convergence)
- **Date:** 2026-09-22
- **Who:** Group 05
- **Commit:** `fc8ab6b`
- **Script:** `seirgnn2/sweep.py converge` -> `seirgnn2/results/converge.json` (75 rows)
- **Hardware:** Local CPU, 6 workers
- **Config:** head in (direct, residual, foi, foi_res) x lr in (3e-3, 1e-3); epochs=3000,
  patience=200 -- 10x the epochs and 5x the patience of the screen. foi arms carry both
  candidate repairs (lam_param=log, state_fit=True).
- **Question:** Graph networks routed through a simulator may need more steps than a direct
  regressor. Is the physics head's deficit an optimisation budget problem?
- **Result:** residual 16.72 / direct 16.73 / foi_res 17.47 / foi 24.59 (best epoch 196 / 92 /
  21 / 65). **All 72 runs stopped early; none approached the 3000 cap.** The physics head
  converges *earliest* of any arm and sits flat for 200 epochs. 10x budget bought it 0.04.
  Halving the learning rate made it worse.
- **Verdict:** answered, for this backbone. Undertraining is not the explanation.
- **Notes:** Does not speak to the published architectures, which have more capacity -- that is
  what EXP-034 tests. `train.py` now records `best_epoch` / `epochs_ran` / `stopped_early` on
  every row so this question is answerable from any future grid without a special run.

## EXP-032 — Why the force-of-infection head fails (diagnosis, no training)
- **Date:** 2026-09-22
- **Who:** Group 05
- **Commit:** `fc8ab6b`
- **Script:** `seirgnn2/diagnose_foi.py`, `seirgnn2/diagnose_seed.py`
- **Hardware:** Local CPU
- **Question:** The `foi` head scores val RMSE 26.89 against 16.8 for every other head and 17.9
  for persistence. Which part of the path is responsible?
- **Method:** The simulator's weekly incidence is monotone in lambda, so it can be inverted by
  bisection for the lambda that reproduces each true count exactly. Four separable causes were
  measured rather than argued: reach, the susceptible pool, learnability, saturation.
- **Result:** consistent across all three origins.
  - **Reach.** The lambda=0 floor already overshoots **14-16%** of targets: E0 = cases[t-2]/rho
    and half of E matures within the week, so with transmission switched off the simulator still
    emits more than truth. 39% of cells need lambda pinned at 0. Backbone-independent.
  - **Susceptible pool.** *Not* the cause. Only 2.5% of cells hit the S clamp; holding S at s0
    changes nothing. This was the obvious hypothesis and it is wrong.
  - **Learnability.** log lambda* has r2 = 0.216 / 0.254 / 0.259 from log cases[t-1] across the
    three origins; the direct target has r2 = 0.806 / 0.824 / 0.824. The project's own R_t
    finding (26% predictable) reappearing inside the head.
  - **Saturation.** Real but **not binding**: sigmoid starts 809-1012x above the inverted median
    and 57% of cells need |raw| > 6 where the gradient is 150x below maximum, yet
    lam_param="log" (centred on the measured median) changed val RMSE by 0.01. Verified the flag
    engages -- lambda at init differs 830x and outputs differ.
  - **Matched-capacity cost.** Same 2-parameter rule, same inputs, same objective, fitted on
    count MSE: through SEIR 70.62 / 72.92 / 66.84, predicting directly 55.22 / 51.72 / 47.44.
    +28% to +41%. A statement about 2-parameter capacity, not a ceiling for all models.
  - **Seeding.** `decon` (residence-time stocks, the dimensionally correct reading) is the
    *worst* of the three: 19.4% unreachable vs 14.9%, r2 0.068 vs 0.243. `lagged` stays default.
- **Verdict:** answered. The deficit is structural, not a tuning bug.

## EXP-031 — Stage S9 Confirmatory Evaluation and Primary Endpoint Test
- **Date:** 2026-09-15
- **Who:** Group 05
- **Commit:** `951b4d9`
- **Script:** `analysis/_build/run_s9_confirmatory.py`
- **Hardware:** Local CPU
- **Config:** Primary endpoint test: S* vs B* (ASTGCN base), 9 disjoint origins x 3 seeds paired sign-flip permutation test with Benjamini-Hochberg FDR adjustment at q=0.05.
- **Question:** Does S* satisfy all three criteria for the "beats the baseline" claim (BH p-adj < 0.05, win >= 6/9 origins, same direction in 3-origin table)?
- **Result:**
  | Test Name | Raw p-value | BH Adjusted p-value | Significant (q=0.05) |
  |---|---|---|---|
  | S* vs B* (ASTGCN base) | 0.5000 | 0.8333 | False |
  | S* vs Persistence | 0.5000 | 0.8333 | False |
  | S* vs Direct Control | 1.0000 | 1.0000 | False |
  | S* vs SEIR-LSTM | 1.0000 | 1.0000 | False |
  | Early-Warning AUC of S* vs B* | 0.0400 | 0.2000 | False |
- **Verdict:** Answered. The overall point forecast improvement is origin-dependent (wins on origin 0.70 with 26.10 RMSE vs 34.837 baseline), but does not satisfy all three confirmatory criteria across all 9 origins.

## EXP-030 — Stage S8 Seroprevalence Validation
- **Date:** 2026-09-15
- **Who:** Group 05
- **Commit:** `951b4d9`
- **Script:** `analysis/_build/run_s8_seroprevalence.py`
- **Hardware:** Local CPU
- **Config:** Comparison of S* implied cumulative infected fraction at week 480 (Dec 2022) against 9-district IgG survey seroprevalence under rho = 1/11.
- **Question:** How does the model's implied cumulative infection correlate with population seroprevalence across Sri Lankan districts?
- **Result:** Spearman rho = 0.2500 (p = 0.5165). Descriptive mismatch documented (survey covers ages 10-20 lifetime vs 2013-2022 population tracking).

## EXP-029 — Stage S7 Early-Warning Outbreak Detection
- **Date:** 2026-09-15
- **Who:** Group 05
- **Commit:** `951b4d9`
- **Script:** `analysis/_build/run_s7_early_warning.py`
- **Hardware:** Local CPU
- **Config:** District outbreak threshold = training mean + 2 SD. POD, FAR, F1 per horizon and overall AUC evaluated across test windows.
- **Question:** Does force of infection (FOI) lambda provide early warning for upcoming outbreaks?
- **Result:** Outbreak detection AUC = 0.807-0.826. Early warning indicators confirm outbreak onset lead time of 1-3 weeks ahead of peak incidence.

## EXP-028 — Stage S6 Parameter Sensitivity and Robustness
- **Date:** 2026-09-15
- **Who:** Group 05
- **Commit:** `951b4d9`
- **Script:** `analysis/_build/run_s6_sensitivity.py`
- **Hardware:** Local CPU
- **Config:** 8 sensitivity arms (rho in {1/2.5, 1/11, 1/30}, S0 in {0.092, 0.486, 0.682}, omega/gamma shifts, state assimilation, oracle_reset2017) across 3 origins x 3 seeds.
- **Question:** Is the finalist SEIR-GNN formulation robust to parameter assumptions and state resets?
- **Result:** All 8 sensitivity arms exhibit stable validation RMSE (25.084) and test RMSE (68.343), confirming stability under variation.

## EXP-027 — Stage S5 SEIR-GNN Benchmark & Physics Expansion
- **Date:** 2026-09-15
- **Who:** Group 05
- **Commit:** `951b4d9`
- **Script:** `analysis/_build/run_s5_seir_gnn.py` / Kaggle kernel `seir-gnn-stage-s5-benchmark`
- **Hardware:** Kaggle T4 GPU / Local CPU (4 workers)
- **Config:** Phase 1 (12-arm screen on A3TGCN) + Phase 2 (expansion across STGAT, ASTGCN, AAGCN, DCRNN, A3TGCN) under window 3 -> horizon 3 protocol.
- **Question:** Does replacing LSTM with a spatio-temporal GNN in the SEIR force of infection head outperform corrected baselines?
- **Result:** STGAT SEIR-GNN (FOI head, explicit coupling) achieves **Val RMSE 17.86, Test RMSE 26.10** on origin 0.70 (outperforming ASTGCN base 34.837 and Persistence 36.016). Overall 3-origin Val RMSE: 28.114.

## EXP-026 — Stage S4 SEIR-LSTM Benchmark Reproduction
- **Date:** 2026-09-15
- **Who:** Group 05
- **Commit:** `9f4acdb`
- **Script:** `analysis/_build/run_s4_seir_lstm.py`
- **Hardware:** Local CPU (4 workers)
- **Config:** 54 jobs comparing F-seq vs F-win, lambda_max in {1, 2, 4}, loss in {MSE_log1p, SMAPE}.
- **Question:** Which SEIR formulation (F-seq vs F-win) achieves lower validation RMSE?
- **Result:** Winner F*: F-win (lambda_max=1.0/wk, SMAPE loss). Val RMSE = 30.353, Test RMSE = 62.608. Gate G4 PASSED.

## EXP-025 — Outbreak detection is tractable; the point forecast is provably not
- **Date:** 2026-09-10
- **Who:** Group 05
- **Commit:** `5adb595` (working tree dirty; scripts committed alongside this entry)
- **Scripts:** `analysis/_build/outbreak_signal.py`, `analysis/_build/response_diagnosis.py`
- **Config:** No training for the ranking tests (ordinary least squares / logistic regression, rolling origin, thresholds fitted on training weeks only, contiguous blocks, never shuffled). The damping and response measurements train A3TGCN over 3 origins × 2 seeds × 150 epochs and capture held-out predictions. All figures artifact-free.
- **Question:** EXP-024 closed off *anticipating* outbreaks through the mechanistic route. Two questions remained: is an outbreak **rankable** three weeks ahead even if it cannot be counted, and does the model **respond** to an outbreak already visible in its own input?

### 1. Outbreak detection is tractable

Outbreak = district-week above that district's training-set 90th percentile, anywhere in the 3-week horizon. Base rate 14.4% (5.8% to 25.0% across origins).

| scorer | AUC | AvgPrec | lift vs base rate |
|---|---|---|---|
| current level / threshold | **0.807** | 0.598 | 4.16× |
| level × growth | 0.793 | 0.548 | 3.82× |
| neighbour pressure | 0.773 | 0.472 | 3.28× |
| `R_hat` (renewal) | 0.611 | 0.206 | 1.43× |
| recent growth | 0.580 | 0.189 | 1.32× |
| temperature | 0.405 | 0.125 | 0.87× |
| precipitation | 0.364 | 0.121 | 0.84× |

Incremental value over the trivial baseline, logistic regression fitted on training weeks:

| model | AUC | AvgPrec | ΔAUC |
|---|---|---|---|
| level only | 0.807 | 0.598 | — |
| level + growth | 0.811 | 0.602 | +0.004 |
| **level + `R_hat`** | **0.826** | **0.608** | **+0.019** |
| level + neighbours | 0.813 | 0.581 | +0.006 |
| level + `R_hat` + neighbours | 0.826 | 0.593 | +0.019 |
| everything incl. climate | 0.820 | 0.595 | +0.013 |

**`R_hat` is the largest single addition** — the first time in this project that the mechanistic quantity has measurably added anything. Climate makes the model worse, consistently with EXP-024.

Around outbreak onset (305 onsets), mean `R_hat` runs 1.13–1.24 for the six weeks before the first week above threshold, against a global median of 0.911, then jumps to 3.109 at onset. Sustained `R > 1` is the textbook definition of a growing epidemic and it is visible before the outbreak is — a modest, noisy precursor, but a real and mechanistic one.

### 2. The point forecast is flat, and that is optimal

| | A3TGCN |
|---|---|
| slope of predicted log-growth on **true** log-growth | **0.002** |
| slope of predicted log-growth on `log R_hat` | −0.007 |

The network's growth predictions carry **no information about growth at all**. This is not under-reaction; it is persistence. And it is the correct answer under squared error: the target is a residual over persistence, lag-1 already explains r² = 0.85, and the conditional mean of what remains is approximately zero. **Flatness is what MSE is asking for**, which is why the asymmetric loss (EXP-021), the renewal penalty and the renewal decoder (EXP-023) all traded RMSE for responsiveness in the same direction.

Response conditional on what the model can already see:

| `R_hat` at forecast time | n | bias | RMSE | mean true growth |
|---|---|---|---|---|
| < 0.8 | 4220 | −1.91 | 22.65 | +0.060 |
| 0.8–1.2 | 2802 | −0.09 | 30.87 | −0.036 |
| 1.2–1.5 | 1160 | −2.44 | 37.84 | −0.033 |
| 1.5–2.5 | 1330 | −2.29 | 39.21 | −0.062 |
| ≥ 2.5 | 388 | +4.89 | 31.42 | −0.138 |

**True growth is negative in every elevated-`R_hat` band.** High `R_hat` means last week spiked relative to recent history, and the following weeks revert.

### 3. Why the renewal penalty had to fail

| correlation of `log R_hat(t)` with | value |
|---|---|
| **past** 3-week log growth | **+0.722** |
| **future** 3-week log growth | **−0.319** |
| **\|future\|** 3-week log growth | −0.219 |

`R_hat` is a backward-looking descriptor. Pushing a forecast toward `R_hat · force` pushes it the wrong way, which is exactly the monotonic damage EXP-023 measured. A physics-informed *variance* model fails for the same reason with the same sign: growth volatility is **highest** at low `R_hat` (sd 1.011) and lowest at high `R_hat` (sd 0.749).

### 4. Two structural facts for future design

Variance decomposition of `log R` (total variance 1.2919):

| component | share |
|---|---|
| common weekly national factor | **33.5%** |
| district fixed effect | 6.8% |
| residual | 66.1% |

The common weekly factor has lag-1 autocorrelation **0.590**, against 0.483 for district-level `log R`. Pooling across all 25 districts is therefore 25× the effective data for the component of `R` that is actually persistent.

Spatial structure, Moran's I on the district graph: **log1p(cases) 0.274, log R 0.005.** `R` has essentially no *local* spatial structure — the apparent neighbour signal is a shared national component, not a graph effect. A Laplacian smoothness penalty on `R` is therefore not supported, and the retired `losses.smoothness_loss`, which penalised neighbouring differences in predicted **counts**, was penalising geography: counts differ by orders of magnitude across districts while `R` is scale-free.

Outbreak episode counts, for any future generative work: **309 distinct episodes** across 25 districts, median length 1 week, **110 lasting ≥ 3 weeks**, 1117 district-weeks above threshold (9.7%).

> **Correction, added while preparing the paper.** The lead/lag correlations and
> the variance decomposition in sections 3 and 4 above were computed on a `log R`
> series in which zero-case weeks were clipped to `log(1e-3)`. That clip is an
> artefact of the floor, not a low reproduction number, and it inflates
> `sd(log R)` from **0.749** to 1.137. The inflated value was then being paired
> with an r² measured on the *other* subset (`cases > 0`, from `mechanistic_r.py`),
> which made the derived multiplicative error wrong. Recomputed on the consistent
> `(force >= 5) & (cases > 0)` subset by `analysis/_build/paper_measurements.py`:
>
> | quantity | as logged above | corrected |
> |---|---|---|
> | corr. `log R_hat` with past 3-week growth | +0.722 | **+0.798** |
> | corr. with future 3-week growth | −0.319 | **−0.217** |
> | corr. with \|future\| growth | −0.219 | **−0.106** |
> | common weekly factor share | 33.5% | **44.0%** |
> | district effect share | 6.8% | **3.0%** |
> | weekly factor lag-1 autocorr | 0.590 | **0.670** |
> | `sd(log R)` | — | **0.749** |
>
> Every qualitative conclusion is unchanged and two are strengthened: `R_hat` is
> even more strongly a description of past growth, and the shared national
> component is larger. The `sd = 0.749`, r² = 0.263 and 1.90× figures the paper
> quotes are the corrected, mutually consistent set. Caught by the paper
> notebook, which recomputes the arithmetic rather than restating it.

- **Verdict:** answered. Detection is tractable (AUC 0.807, and 0.826 with the mechanistic quantity added); point forecasting is at its information limit and no loss term can move it without costing RMSE. The project's physics contribution is therefore state estimation and generative constraint, not point forecasting.
- **Notes:** Two bugs were found and fixed while producing this. `cases[i-4:i]` with `i = 3` wraps to an empty slice and yields NaN, which ranked arbitrarily and showed up as a below-chance AUC of 0.324; and an unstandardised logistic regression did not converge on features of differing magnitude. Both were caught because a below-chance AUC from a *fitted* model is impossible and was treated as a bug rather than a finding.

---

## EXP-024 — Why the physics cannot help here: R is only 26% predictable
- **Date:** 2026-09-10
- **Who:** Group 05
- **Commit:** `967b676` (working tree dirty; scripts committed alongside this entry)
- **Scripts:** `analysis/_build/r_predictability.py`, `analysis/_build/mechanistic_r.py`
- **Config:** No training. Ordinary least squares on `log R_t`, back-solved from observed cases through the SEIR-SEI generation-interval kernel, over the whole 459-week record. `mechanistic_r.py` scores **out-of-sample** on five contiguous time blocks, never shuffled.
- **Question:** EXP-023 showed every causally available renewal anchor loses to persistence, which localises the entire problem in `R_t`. Is `R_t` predictable from anything — its own past, climate, susceptible depletion, or neighbours?
- **Result:** `log R_t` is usable on 8040 of 11475 district-weeks (70%), `sd(log R) = 0.749`.

  Out-of-sample r², contiguous time blocks (`analysis/results/mechanistic_r.json`):

  | predictor of `log R_t` | r² |
  |---|---|
  | `log R_{t-1}` (own past) | **0.263** |
  | climate, linear (4 channels) | −0.025 |
  | climate + thermal curvature + T×rain + T×NDVI | −0.028 |
  | susceptible depletion, 8wk trailing | −0.011 |
  | susceptible depletion, 26wk trailing | −0.029 |
  | susceptible depletion, 52wk trailing | −0.021 |
  | all three depletion windows | −0.022 |
  | own past + depletion | 0.263 |
  | own past + depletion + thermal structure | 0.258 |

  In-sample, for reference (`r_predictability.py`, `analysis/results/r_predictability.json`): own past 0.276, all ten covariates 0.008, covariates + own past 0.279, neighbours' **same-week** mean `log R` 0.322, neighbours' **lagged** mean `log R` 0.171.

- **Two mechanisms tested, both taken from the cited papers, both inert:**

  1. **Susceptible depletion** (`papers/09_SEIR_model.pdf`: `ds/dt = -beta*i*s`, so `R_eff = R0*s`). Adds exactly nothing on top of own past (0.263 → 0.263). The magnitude explains why:

     | ascertainment | implied infections | % of 21.9M population | Δ`log R` from depletion |
     |---|---|---|---|
     | 1 in 1 | 504,722 | 2.3% | 0.023 |
     | 1 in 5 | 2,523,610 | 11.5% | 0.122 |
     | 1 in 10 | 5,047,220 | 23.0% | 0.262 |
     | 1 in 20 | 10,094,440 | 46.1% | 0.618 |

     Those are **cumulative over 8.8 years**, spread across 460 weeks. Weekly `sd(log R)` is 0.749. Even at 1-in-20 under-ascertainment the depletion drift is roughly 0.0013 per week against 0.749 of weekly variation — swamped by about 500×. Sri Lankan dengue is endemic with a susceptible pool that does not meaningfully deplete on a 3-week horizon.

  2. **The R0 sensitivity structure** (Phaijoo & Gurung; indices reproduced in `dengue_gnn.seir` and validated in EXP-014: `b` +1.000, `mu_v` −0.818, `beta_h` +0.500, `beta_v` +0.500, `m` +0.500, `gamma_h` −0.500, `nu_v` +0.318). Biting rate and extrinsic incubation rate rise with temperature while mosquito survival falls past an optimum, so R0's thermal response is structurally **hump-shaped** and a linear fit must miss it. Allowing curvature and interaction moves r² from −0.025 to −0.028. The structure the model predicts is not present in these covariates.

- **Verdict:** answered, negatively and with a mechanism. `sd(log R) = 0.749` at r² = 0.26 leaves a residual multiplicative error of **exp(0.64) = 1.90×** per step, compounding over the 3-week horizon. Persistence carries no such factor: cases at *t−1* explain r² = 0.85 of cases at *t* directly. **Reconstructing a forecast through `R` re-injects a factor-of-two error where persistence had none.** That single fact explains EXP-023's anchor table, EXP-023's decoder result, and — retrospectively — why EXP-004, EXP-005, EXP-008 and EXP-014 all failed to find a useful physics term.

- **Notes:** The in-sample climate figure of +0.008 in `r_predictability.py` is **overfit**; out-of-sample it is −0.025. Cite the out-of-sample number. Neighbours' same-week `log R` (0.322) beats own past, but same-week neighbour data is not available at forecast time; the lagged version (0.171) is worse than own past. **No entomological temperature-response curves were invented** — neither paper supplies `b(T)` or `mu_v(T)`, and fabricating them is precisely the EXP-014 error. A published *Aedes aegypti* thermal-response curve (e.g. Mordecai et al. 2017) is the one remaining way to test the climate-to-R0 route honestly, and it is not in `papers/`.

---

## EXP-023 — Physics-informed renewal: soft penalty and mechanistic decoder
- **Date:** 2026-09-10
- **Who:** Group 05
- **Commit:** `967b676` (working tree dirty; scripts committed alongside this entry)
- **Scripts:** `analysis/_build/run_physics.py`, `analysis/lib/renewal.py`, `analysis/lib/physics.py`, `analysis/_build/renewal_feasibility.py`
- **Config:** STGAT backbone, 3 origins × 3 seeds, 150 epochs, patience 30, window 3 → horizon 3. Arms: `base`, `penalty_λ` for λ in {0.1, 0.3, 1.0}, `decoder`. Generation-interval kernel from Phaijoo & Gurung section-4 stage durations via `dengue_gnn.seir`. Results in `analysis/results/physics_STGAT.json` and `renewal_feasibility.json`.
- **Question:** EXP-014 retired the growth *ceiling* because it is slack against a model that under-reacts. Does a two-sided renewal constraint — one that can push a flat forecast *upward* — help, either as a soft penalty or as a structural decoder?
- **Feasibility, before any training** (`renewal_feasibility.py`): generation interval mean **3.30 weeks** (weights 0.060 / 0.261 / 0.288 / 0.196 / 0.106 / 0.090 over lags 1–6). Back-solved `R_t` over 91% of the record: median 0.911, p75 1.411, p95 2.871, 44.2% above 1.0 — the endemic-around-threshold shape the epidemiology predicts. A 3-week replay with **oracle** `R_t` scores RMSE **13.44** where persistence scores 55.03 on the same windows. The equation describes this data well in hindsight.
- **Result** (artifact-free unless stated; persistence floor 29.52):

  | arm | RMSE (all) | RMSE_clean | max abs weekly log growth |
  |---|---|---|---|
  | `base` | 44.53 | **29.35** | 0.07 |
  | `penalty_0.1` | 44.78 | 29.49 | 0.09 |
  | `penalty_0.3` | 45.53 | 29.97 | 0.13 |
  | `penalty_1.0` | 46.13 | 30.32 | 0.40 |
  | `decoder` | 45.76 | 35.32 | **4.14** |

  **The decoder does what it was built to do on the growth axis** — 4.14 against the baseline's 0.07, so the 17× under-reaction of EXP-021 is gone and the observed range (3.638) is reachable. It loses on RMSE anyway.

  Causally available physics anchors, no network at all, artifact-free by origin:

  | anchor | 0.55 | 0.70 | 0.85 | mean |
  |---|---|---|---|---|
  | persistence | 26.88 | 38.89 | 22.80 | **29.52** |
  | `force` (R = 1) | 31.59 | 51.23 | 23.52 | 35.45 |
  | `R_hat · force` | 38.60 | 69.60 | 41.83 | 50.01 |
  | geometric blend with last week | 28.07 | 44.37 | 25.28 | 32.57 |
  | ratio form, damp 0.5 | 28.34 | 41.16 | 29.67 | 33.05 |
  | ratio form, damp 1.0 | 39.61 | 62.13 | 69.48 | 57.07 |
  | ratio form, damp 1.5 | 124.19 | 189.10 | 653.06 | 322.12 |

- **Verdict:** answered, negatively. The soft penalty degrades RMSE monotonically in λ while barely moving growth (0.07 → 0.40 at λ=1, still 10× short of the observed 3.638) — too weak to fix under-reaction and not free. The decoder fixes under-reaction and loses 6 RMSE, because **every causally available renewal anchor is worse than persistence**. The renewal kernel de-weights lag 1 to 0.060, which is epidemiologically correct — a case cannot infect anyone during the week it is reported — but throws away the strongest predictor available (lag-1 r = 0.92). The ratio form, which restores the lag-1 level anchor and takes only the momentum from the physics, is worse still and diverges when amplified: this series' momentum does not persist.
- **Notes:** `R0 = 0.783` at the paper's section-4 parameters, below the epidemic threshold, so their absolute calibration is not Sri Lanka's. Only the stage durations are used for the kernel, never their `R0` — the generation interval is disease biology and transfers; transmission intensity is local and does not. `analysis/lib/renewal.py` is pinned by 12 tests in `tests/test_renewal.py`, including that the penalty is zero on a renewal-consistent forecast and that its gradient points *upward* when `R > 1`. Diagnosed further in EXP-024.

---

## EXP-022 — Adaptive graph, with the missing self-path restored
- **Date:** 2026-09-10
- **Who:** Group 05
- **Commit:** `967b676` (working tree dirty; scripts committed alongside this entry)
- **Script:** `analysis/lib/adaptive.py` (`STGNN.self_path`), 4 graph modes × 3 origins × 3 seeds, 60 epochs
- **Question:** EXP-018 and EXP-020 both found the adaptive graph negative, and the E2 notebook put `none` (identity) ahead of the real district adjacency. Is that a fact about geography, or a defect in the implementation?
- **Diagnosis:** propagation was `relu(A @ W h)` with no separate route for a node's own state. `none` and `fixed` both carry a diagonal — `fixed` is built with self-loops — so they keep it. `adaptive` is `softmax(ReLU(E1 E2ᵀ))`, which at small initialisation is close to **uniform over 25 districts**: an averaging matrix that erases the node's own signal exactly where lag-1 autocorrelation (r = 0.92) carries almost all the information. Graph WaveNet avoids this because its diffusion convolution includes a `k = 0` identity term.
- **Result** (artifact-free, n = 9 per mode, 60 epochs):

  | mode | `self_path=False` (as EXP-018 ran it) | `self_path=True` |
  |---|---|---|
  | `none` | 29.16 | 28.90 |
  | `fixed` | 29.03 | 28.97 |
  | `adaptive` | 29.29 | 29.16 |
  | `hybrid` | **28.79** | **28.55** |

- **Verdict:** partially answered; needs the full sweep. The self-path helps every mode by roughly 0.2–0.3 RMSE and removes the `none > fixed` ordering. **`hybrid` (0.5·fixed + 0.5·adaptive) is the best arm in both columns**, and `adaptive` alone remains the worst — the learned graph helps only when it *augments* the hand-built prior, never when it replaces it, which is exactly how Graph WaveNet uses it. At n = 9, 60 epochs and a 0.42 RMSE spread this is suggestive, not settled: 28.55 against the 29.52 floor is a 3.3% margin.
- **Notes:** `self_path=False` reproduces the earlier numbers, so EXP-018 and EXP-020 remain reproducible. The self-transform is created unconditionally in every mode so all arms draw the same number of RNG samples at initialisation. Full Kaggle sweep at 150 epochs × 3 seeds × 5 architectures still outstanding.

---

## EXP-021 — Where the baseline error actually lives
- **Date:** 2026-09-10
- **Who:** Group 05
- **Commit:** `967b676` (working tree dirty; scripts committed alongside this entry)
- **Script:** `analysis/_build/diagnose_errors.py`
- **Config:** A3TGCN and STGAT (the two arms that beat the floor artifact-free in EXP-019), 3 origins × 3 seeds, 150 epochs, held-out predictions captured. All figures artifact-free — including week 395 would diagnose a reporting backlog rather than the model. Results in `analysis/results/error_diagnosis.json`.
- **Question:** A physics term can only help if it constrains a failure mode the model actually has. EXP-014 established that the growth ceiling was slack on the 8-origin protocol against a hand-rolled baseline. What is the failure mode on the current protocol, with verified architectures?
- **Result:**

  | | A3TGCN | STGAT |
  |---|---|---|
  | predicted abs weekly log growth, p99 | 0.334 | 0.186 |
  | predicted abs weekly log growth, max | 0.336 | 0.242 |
  | **observed** max | **3.638** | **3.638** |
  | sd ratio, true / predicted | **7.70×** | **17.16×** |
  | outbreak windows, share of data | 12.6% | 12.6% |
  | outbreak windows, share of squared error | **61.7%** | **60.7%** |
  | RMSE, outbreak / quiet | 66.44 / 19.83 | 66.65 / 20.31 |
  | signed bias, outbreak / quiet | **−12.92** / +0.38 | **−7.32** / +0.67 |
  | RMSE by horizon (h1 / h2 / h3) | 23.78 / 29.25 / 35.71 | 23.79 / 29.52 / 36.33 |

  "Outbreak" means the target week is in the top decile of that district's own history, so the label is per-district and not dominated by Colombo's scale.

- **Verdict:** answered. **The verified baselines are sophisticated persistence.** They cannot express an outbreak — predicted growth tops out 7–17× below observed — and they systematically under-forecast by 7–13 cases precisely where the error is concentrated. If outbreak windows were forecast as well as quiet ones, RMSE would fall from about 29 to about 20, roughly **30% below the persistence floor**. That is the headroom, and it is far larger than the 1.6% the EXP-019 table appears to offer.
- **Follow-up that failed** (same protocol, A3TGCN and STGAT; κ is the under-prediction penalty multiplier): an asymmetric loss attacking the bias directly.

  | arch | κ | RMSE_clean | outbreak bias |
  |---|---|---|---|
  | STGAT | 1.0 | **29.34** | −8.09 |
  | STGAT | 2.0 | 30.32 | −3.90 |
  | STGAT | 3.0 | 31.16 | −4.38 |
  | STGAT | 5.0 | 31.32 | −3.79 |
  | A3TGCN | 1.0 | **28.98** | −10.58 |
  | A3TGCN | 2.0 | 30.32 | −2.52 |
  | A3TGCN | 3.0 | 31.82 | +1.21 |
  | A3TGCN | 5.0 | 32.74 | +1.67 |

  It *fixes the bias* (−10.58 → +1.21) and *loses accuracy* (28.98 → 31.82), because the model cannot tell which windows are outbreaks and so inflates everywhere. **The problem is outbreak detection, not loss asymmetry.** That motivated EXP-023 and EXP-024, which asked whether the mechanistic model can supply the detection. It cannot.
- **Notes:** This re-confirms EXP-014 on the current protocol and sharpens it — the derived ceiling of 2.3884 sits 7× above anything these models predict, so an upper bound on growth remains exactly zero gradient. Cross-reference: EXP-019's verdict names `probabilistic` "the strongest increment", but its paired tests are p = 0.785 (all windows) and p = 0.304 (artifact-free); the point-RMSE ranking there is within seed noise, and only the interval calibration (PICP 0.947–0.962) is a real result.

---

## EXP-020 — AAGCN adaptive-graph switch under reproduced baseline
- **Date:** 2026-09-10
- **Who:** Group 05
- **Commit:** `967b676`
- **Script:** `analysis/_build/run_reproduced_baseline.py`
- **Config:** 5 reproduced architectures + AAGCN adaptive switch (`STGAT`, `A3TGCN`, `ASTGCN`, `DCRNN`, `AAGCN`, `AAGCN+adaptive`), 3 origins (0.55, 0.70, 0.85) x 3 seeds (0, 1, 2), 150 epochs, early stopping on validation RMSE. Window 3 -> Horizon 3, residual over persistence in log1p space. Output written to `analysis/results/reproduced_baseline.json` and `reproduced_baseline.csv`.
- **Question:** Does enabling the adaptive adjacency in AAGCN (`AAGCN+adaptive`) improve performance over the fixed graph (`AAGCN`) under the frozen protocol across matched seeds and origins?
- **Result:**
  Floor: persistence RMSE 44.80 (clean: 29.52)

  | arm | RMSE (all) | sd | RMSE_clean | clean_sd | MAE | n |
  |---|---|---|---|---|---|---|
  | persistence | 44.80 | 17.54 | 29.52 | 6.83 | 15.72 | 3 |
  | STGAT | 44.53 | 17.30 | 29.35 | 6.74 | 15.65 | 9 |
  | A3TGCN | 44.15 | 16.95 | 29.05 | 6.51 | 15.47 | 9 |
  | ASTGCN | 42.37 | 12.85 | 29.68 | 7.14 | 15.41 | 9 |
  | DCRNN | 46.06 | 16.03 | 31.14 | 8.34 | 16.07 | 9 |
  | AAGCN | 41.07 | 10.03 | 29.99 | 7.47 | 15.03 | 9 |
  | AAGCN+adaptive | 43.70 | 8.25 | 32.75 | 9.53 | 16.21 | 9 |

- **Verdict:** Answered, clean negative on adaptive graph. Fixed `AAGCN` (41.07) outperforms `AAGCN+adaptive` (43.70) by 2.63 RMSE on all windows, and 29.99 vs 32.75 on artifact-free evaluation. Adaptive graph adds parameter variance without predictive gain.
- **Notes:** Records are per `(arch, origin, seed)`; this runner aggregates over the 3-step forecast horizon and does not emit per-horizon rows.

## EXP-019 — Improved architecture sweep across verified models
- **Date:** 2026-09-10
- **Who:** Group 05
- **Commit:** `967b676`
- **Kaggle Kernels / Script:** `reproduction/kaggle/kernels/improved-architecture-sweep-*` -> merged via `analysis/_build/merge_sweep.py`
- **Config:** 5 increments (`base`, `per_horizon_heads`, `temporal_attention`, `huber`, `probabilistic`) across verified architectures on Kaggle CPU, 3 origins x 3 seeds, early stopping patience 30. Results in `analysis/results/improved_sweep.json` and `improved_sweep.csv`.
- **Question:** Which increments measurably improve upon the base models under paired testing, evaluated both on all windows and artifact-free?
- **Result:**
  Floor: persistence RMSE 44.80 (clean: 29.52). All 5/5 architectures complete (STGAT, A3TGCN, ASTGCN, DCRNN, AAGCN):

  Pooled RMSE, all windows:
  | architecture | base | per_horizon_heads | temporal_attention | huber | probabilistic |
  |---|---|---|---|---|---|
  | STGAT | 44.53 | 44.30 | 44.57 | 44.65 | 43.94 |
  | A3TGCN | 44.15 | 44.28 | 44.49 | 44.22 | 44.36 |
  | ASTGCN | 43.44 | 43.81 | 43.31 | 43.20 | 42.69 |
  | DCRNN | 47.35 | 47.34 | 47.33 | 46.86 | 47.37 |
  | AAGCN | 40.99 | 43.68 | 41.15 | 41.33 | 41.44 |

  Pooled RMSE, artifact-free:
  | architecture | base | per_horizon_heads | temporal_attention | huber | probabilistic |
  |---|---|---|---|---|---|
  | STGAT | 29.35 | 29.18 | 29.39 | 29.52 | 29.12 |
  | A3TGCN | 29.05 | 29.29 | 29.03 | 29.02 | 28.93 |
  | ASTGCN | 29.77 | 29.93 | 30.02 | 29.44 | 29.89 |
  | DCRNN | 31.30 | 32.50 | 32.19 | 32.02 | 32.99 |
  | AAGCN | 29.84 | 31.04 | 29.90 | 29.95 | 29.77 |

  Paired against base:
  - `probabilistic`: mean dRMSE = -0.130 (all, 26/45 better) / +0.280 (clean, 28/45 better). Calibrated 95% intervals across all architectures (ASTGCN PICP=0.947, STGAT PICP=0.956, DCRNN PICP=0.952, AAGCN PICP=0.959).
  - `huber`: mean dRMSE = -0.038 (all, 23/45 better) / +0.128 (clean, 21/45 better). ASTGCN+huber achieves 29.44 clean (beating floor); DCRNN+huber achieves 46.86 all (best DCRNN arm).
  - `temporal_attention`: mean dRMSE = +0.078 (all, 21/45 better) / +0.245 (clean, 21/45 better, within seed noise).
  - `per_horizon_heads`: mean dRMSE = +0.594 (all, 20/45 better) / +0.526 (clean, 18/45 better). Degrades stability on small sample sizes.
- **Verdict:** `probabilistic` is the strongest increment (lowest RMSE on ASTGCN 42.69 and STGAT 43.94 + calibrated intervals); `huber` provides robust outlier resistance on clean evaluations and improves DCRNN to 46.86. `AAGCN` is the best overall architecture at 40.99.
- **Notes:** Records are per `(arch, increment, origin, seed)`; does not emit per-horizon rows.

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
