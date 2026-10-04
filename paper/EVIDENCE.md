# Evidence ledger

Audit base: `origin/main` @ 2800651 (2026-10-05). Branch-only sources are written `branch:path`.
No training was re-run. Aggregations marked "derived" were computed from saved result files by reading them (`seirgnn2/stats.py`, a short pandas script, and `seirgnn2.sweep.persistence_rows`, which only reads data).

**Status key:** VERIFIED = in a saved output file. LOG-ONLY = only in a markdown log. CONFLICT = sources disagree. MISSING = not found.

## Protocols (a number may only share a table with numbers from the same protocol)

| Code | Data | Origins and test | Persistence test RMSE | Where defined |
|---|---|---|---|---|
| **P3** | rebuilt (559 wk) | 3 origins 0.55 / 0.70 / 0.85, test fraction 0.15 of windows, seeds 0,1,2 | **36.0157** (val 17.8561) | `seirgnn2/core.py` ORIGINS, TEST_FRAC; EXP-050 |
| **P9** | rebuilt | 9 origins 0.50 to 0.90 step 0.05, test fraction 0.05, seeds 0,1,2. This is the "frozen protocol" of `docs/PROTOCOL.md` (fingerprint e9afbdfb0528) | **28.5410** (val 27.0363) | `docs/PROTOCOL.md`, `core.py` ORIGINS_F9 |
| **PK9** | rebuilt | 9 origins 0.40 + k/15, test fraction 1/15, seeds 0,1,2 (Kaggle confirmatory nine, 7 configs only) | **31.7619** (val 41.2133) | `core.py` ORIGINS_9, EXP-050 |
| **PL** | legacy benchmark array (459 wk, 2013-2022, with reporting backlogs) | 3 folds 0.55 / 0.70 / 0.85, 68 overlapping test windows per fold, seeds 0,1,2 | **44.7953** pooled (29.5210 without the week-395 spike) | EXP-001 to EXP-014 era, `docs/ARRAY_AUDIT.md` |
| PL-S5 | rebuilt, old defective S5 script | 3 origins | 33.87 (different build) | deleted `stage_s5_leaderboard.csv` |

Persistence = repeat the last observed week. It has no parameters, so any two people on the same protocol get the same number.
Mean ± sd convention below: **mean over runs, sd over runs.** The sd is dominated by differences between origins (a quiet year against an epidemic year), not by seed noise. For "does arm X beat persistence" the paired origin-level test is the right statistic and is given where available.

---

# Part A. The candidate story, claim by claim

## C1. Anchor head raises reachable FoI r² from 0.36 to 0.67: **WRONG as worded (mixes two things)**

- The source is EXP-060 (`docs/EXPERIMENT_LOG.md`, `docs/SEIR_ADAPTIVE_AUDIT_RESULTS.md` section 3). It is a **no-training regression diagnostic**, not a property of the anchor head. It regresses the oracle log force of infection on features, on "reachable" cells only (cells not pinned at lambda = 0; 39-41% of cells are pinned).
- Correct values, per origin 0.55 / 0.70 / 0.85, reachable cells: log cases[t-1] alone 0.363 / 0.384 / 0.353; plus log population 0.512 / 0.515 / 0.444; plus log population and log S 0.589 / 0.660 / 0.674. Across **all** cells the same three rows give 0.216-0.259, 0.218-0.260, 0.224-0.273, so almost no change.
- "0.36 to 0.67" takes the smallest value of row 1 and the largest of row 3 from different origins, and the 0.67 needs log S as well as population. The anchor head does not feed S to the network; it builds the population and S scale into lambda by construction.
- The statement that a free log/sigmoid head "must learn an unobserved offset of about -log(N*S)" is the authors' argument (log, models.py comments), supported by this regression but not by an ablation that isolates it.
- Safe wording: "Adding population and susceptible fraction as regressors raised the out-of-sample r² of the oracle log force of infection on reachable cells from 0.35-0.38 to 0.59-0.67 (three origins)."
- Source: `docs/EXPERIMENT_LOG.md` EXP-060; code `seirgnn2/diagnose_foi.py`. Commit 91d0778. Status LOG-ONLY (no JSON of the regression found on main).

## C2. Weight-decay exemption makes adaptive_gwn learn a non-uniform graph, entropy 1.000 to 0.469: **PARTLY CONFIRMED, attribution UNSUPPORTED**

- Confirmed: mean normalized adjacency entropy **0.4693**, range **0.3150 to 0.6182** over 135 runs (5 adaptive_gwn arms x 27 runs), read from `adj_entropy` in `seirgnn2/results/frozen9_plus_audit.json`.
- "1.000 before" is the pre-fix state stated in the log (uniform to four decimals at the old init). No pre-fix run with the same entropy logger exists on main. LOG-ONLY.
- Attribution is not supported. The fix bundled four changes at once: embeddings drawn at randn scale instead of 0.05 x randn; 10 dimensions instead of 16; one shared adjacency instead of one per layer; weight decay set to zero on the embeddings (`train.py::_optimizer`, active only when `adj_init="gwn"`). **No ablation separates the weight-decay exemption.** The initialisation scale alone already breaks the uniform collapse.
- The learned edges are not geographic. Examples in the audit sheet: Kalutara<-Kandy 0.918, Batticaloa<-Puttalam 0.812, Jaffna<-Kegalle 0.722. Do not call the learned graph interpretable.

## C3. ST-GNN baselines beat persistence on raw data only because of reporting backlogs: **UNSUPPORTED as written. No published paper makes the claim.**

- The benchmark paper is Weng et al. 2024, *Graph Representation Learning for Dengue Forecasting*, IEEE BigData (`reproduction/REPRODUCIBILITY_MATRIX.md`). **It reports no persistence or naive baseline** (`crosscheck/FINDINGS.md` F1.0). Its comparison set is ARIMA, random forest, XGBoost, ARNN, LSTM. So no published paper reports "beating persistence on this data with these models".
- What our crosscheck found: on the paper's own protocol, persistence beats all ten models on the cross-validated column. On the Full Dataset column persistence wins on MAE (17.87) but **A3TGCN (30.55) and ASTGCN (32.41) beat persistence on RMSE (33.48)**. So "persistence beats every model in Table I" would be an overstatement (the crosscheck warns against it).
- The backlog effect is real and measured: the legacy array has a 19x spike at week 395; persistence RMSE is 44.80 with it and 29.52 without (EXP-050 `results.json` array_floor, `docs/ARRAY_AUDIT.md`). On the legacy array, five re-implemented architectures (base variants) score 40.99 to 47.35 pooled against persistence 44.80 (`analysis/results/improved_sweep.md`). Artifact-free, the base variants score 29.05 to 31.30 against 29.52, i.e. they straddle it.
- The causal claim "only because of" is not tested. **Limit the claim to our own re-implementations and say "mixed".**

## C4. 559 weeks, 25 districts, 2013-2024; 9 rolling origins, test fraction 0.05, 810 runs: **PARTLY WRONG (three protocols merged)**

- 559 weeks (2013-W26 to 2024-W10), 25 districts, 7 weeks with no report left missing: **CONFIRMED** (`full_paper/outputs/kaggle_run/notebook_output.txt` lines 17-19). Note the series starts in 2013-W26 and ends 2024-W10, so "2013-2024" is correct but not complete years.
- "9 origins, test fraction 0.05" is protocol **P9**. "810 runs" matches **two different things**: (a) EXP-050 = 69 configs x 3 origins x 3 seeds (621) + 7 configs x 9 origins x 3 seeds (189) = 810, run on **P3 and PK9, not P9**; (b) `frozen9_plus_audit.json` has 30 arms x 27 runs = 810 training runs plus 9 persistence rows = 819 rows, on **P9**. The numbers in C5 and C6 come from (b). The paper must not say "810 runs" without saying which.

## C5. adaptive_gwn + residual: val 25.42, delta -1.62, BH-adjusted p = 0.023: **CONFIRMED for validation; test is not significant**

- Source: `seirgnn2/results/frozen9_plus_audit.json`, protocol P9, run through `seirgnn2/stats.py frozen9_plus_audit`.
- Validation: 25.42, persistence 27.04, delta **-1.62**, 9/9 origins better, raw p 0.004, BH p_adj **0.023**.
- **Test: 27.94 vs persistence 28.54, delta -0.60, 7/9 origins, raw p 0.301, p_adj 0.694. Not significant.**
- The EXP-061 log table prints persistence "28.54" in the **validation** column; the correct validation persistence is **27.04** (28.54 is the test value). The delta -1.62 is computed against 27.04.
- The audit sheet calls this "the best val arm in the whole table". **WRONG:** ASTGCN+residual 25.23 and AAGCN+residual 25.24 are better, and both are also significant (p_adj 0.023). Seven of 30 arms are significantly better than persistence on validation (and eight significantly worse); on test none is better (EV-149, EV-150). adaptive_gwn+residual is one of a tied group of seven. (An earlier version of this ledger said thirteen; that was a miscount, corrected here.)
- Same-protocol comparator: gcn+residual val 25.55, test 28.12. The graph effect over this control is 0.13 RMSE on validation, inside the paired sd (1.45).

## C6. adaptive_gwn + foi_res anchor (E0enc): val 25.53, test 27.45, p = 0.047: **CONFIRMED but the p-value is the validation p, and the "27.45" is shared**

- The 25.42 / 0.023 pair belongs to **adaptive_gwn+residual** (C5). The 25.53 / 0.047 pair belongs to **adaptive_gwn+foi_res anchor E0enc**. The draft attached 25.42/0.023 to both; they are different models.
- E0enc: val 25.53, delta -1.51, 8/9 origins, raw p 0.023, **p_adj (val) 0.047**. Test 27.45, delta -1.10, 7/9, raw p 0.199, **p_adj (test) 0.576**. So the headline p = 0.047 is a validation result and the test result is not significant.
- The plain `adaptive_gwn+foi_res anchor` (no E0enc) also has test **27.45** (delta -1.09, p_adj 0.576; E0enc is -1.10 on the same scale) with val 25.74, p_adj 0.090. The two arms are indistinguishable on test.
- `gcn+foi_res anchor` (same head, no adaptive graph) gets val 25.83, test 27.62, p_adj 0.158 / 0.576. Adding the anchored SEIR head on top of the adaptive graph does not beat adaptive_gwn+residual on validation.
- None of the six post-fix arms is significant on validation and test together (EXP-061 verdict).

## C5/C6 follow-ups requested

- **Was model selection on validation?** The protocol says selection on validation RMSE only (`docs/SEIR_GNN_EXPERIMENT_PLAN.md` R6, `CLAUDE.md`). For EXP-061 specifically, **the six arms were hand-picked by the person doing the audit, not pre-registered and not blind to likely RMSE** (log and audit sheet say so, "deviation from plan R6"). The 24 other arms are the full backbone x head grid of EXP-059. The pre-registered 46-config `audit()` grid has not been run.
- **Persistence test RMSE:** 28.5410 (P9). Test p-values for C5 and C6: **0.694 and 0.576** (BH adjusted); raw 0.301 and 0.199.
- **Statistical test:** exact paired sign-flip permutation test (two-sided) on 9 paired units, one unit per origin, seeds averaged within origin first. Source: `seirgnn2/stats.py`, `docs/PROTOCOL.md`.
- **BH correction family:** 30 arms compared with persistence in one call (`frozen9_plus_audit`). Not 6. Selecting six arms and correcting over 30 is conservative for those six; but the six were chosen after seeing the 24, so the family is not clean.
- `docs/PROTOCOL.md` says Holm across arms sharing a baseline, while `stats.py` uses Benjamini-Hochberg. **CONFLICT inside the repo.** The paper must say which one was used (BH, as run).
- The `origin_seed` unit (27 pairs) gives p_adj 0.000-0.001 for the same arms. It treats seeds as independent evidence and is explicitly never valid alone (`stats.py` docstring). Do not quote it.

## C7. Peak outbreak (Fold 2): RMSE 38.65 vs 42.10: **38.65 CONFIRMED; 42.10 NOT FOUND; framing WRONG**

- 38.65 = Fold 2 (origin 0.70, 68 test windows) of the Phase-2 adaptive-graph + physics-loss model PIAG-Net with lambda_phys = 0.10. Raw file `praveen:results/exp004_lambda_sweep.csv` (fold2_RMSE 38.64570) and `praveen:results/exp005_combined_proposed.csv`. Protocol **PL** (legacy array, backlog artifacts, 3 folds).
- The matching baseline in the log is **GCN 42.7** (EXP-005 table) and the raw file gives 42.708 (`praveen:results/baseline_rolling_origin.csv`, mean of 9 rows). **42.10 appears nowhere.**
- **One fold only.** On the other two folds PIAG-Net is worse than GCN: fold 1 27.27 vs 27.1, fold 3 68.42 vs 66.4 (log) or 69.12 (raw GCN mean). The overall mean is 44.78 vs persistence 44.80.
- **Persistence on fold 2 is 38.08** (mean of per-horizon RMSE, same file), so 38.65 does **not** beat persistence either.
- The same table shows lambda_phys = 0 (no physics) at 39.95 on fold 2. The physics term accounts for 1.3 of the 4.0-point drop; the rest is the adaptive graph versus the GCN baseline. The corrected-adjacency rerun (`origin/adaptive-graph:results/phase2_runs.csv`, 192 rows) says the adaptive graph "loses fold 1 and fold 2" and is not significant against the fixed-graph control (paired t-test p about 0.10-0.14, n = 9; commit 22e6404 message).
- "Peak outbreak" is a label the draft added. The fold is described in the log by period (origin 0.70), not as a peak. Do not use it.
- The physics term here is a **spatial regularisation loss**, not SEIR (`docs/PHASE2_REVIEW.md` finding F5, praveen branch).
- Verdict for the paper: exclude as a headline; at most a negative result.

## C8. SEIR-GNN FOI head test RMSE 71.80 vs STGAT Direct 83.99: **UNSUPPORTED and not citable (defective source)**

- The two numbers are the last two rows of `full_paper/outputs/csv/stage_s5_leaderboard.csv`, **deleted** in commit 65c437c (EXP-050). Row text: `STGAT (Direct),GNN,Direct,30.64,83.99` and `SEIR-GNN (Proposed),Physics-GNN,FOI Physics,27.21,71.8`. Same file: persistence floor test **33.87**, LSTM 65.91, A3T-GCN 63.15.
- They come from the old Stage-S5 script with four defects (one gradient step per epoch, SMAPE objective scored by RMSE, constant FoI across horizon, collapsing linear layer). `CLAUDE.md` says: do not propagate any S5-derived number.
- Both are worse than persistence 33.87 by more than a factor of two, so the table would read as "the physics model beats STGAT" only because STGAT was also broken. The corrected, same-protocol statement is: bare foi head 40.58-44.67 test (P9, pre-fix) and 27.99 to 29.80 with the anchor fix, against persistence 28.54, never better.
- 27.45 (C6, P9) and 71.80 (S5 leaderboard) are different data builds, protocols and code. They must never share a table.

## C9. Gaussian head 95% coverage 94.7%-96.2%: **CONFIRMED for coverage; width NOT in the draft and it is large**

- `analysis/results/improved_sweep.md` ("Eq. (14) intervals, nominal coverage 0.95"), generated by `analysis/_build/merge_sweep.py` from five Kaggle kernels. **Legacy array, 3 origins, protocol PL**, 5 architectures, 3 seeds each.

  | architecture | PICP (coverage) | MPIW (mean width, cases per district-week) |
  |---|---|---|
  | STGAT | 0.956 | 122.8 |
  | A3TGCN | 0.962 | 230.5 |
  | ASTGCN | 0.947 | 128.4 |
  | DCRNN | 0.952 | 249.9 |
  | AAGCN | 0.959 | 145.4 |

- Range 94.7% to 96.2% matches. **Mean widths are 122.8 to 249.9 cases** against a point RMSE of about 41 to 47 on the same protocol. Coverage near the nominal level with intervals that wide shows calibration, not sharpness. Report both.
- The probabilistic increment did **not** improve point accuracy (paired against base, all windows: -0.130 RMSE, 26/45 better, p = 0.785).
- This is on the legacy array. No coverage or width was found for P3 or P9. NB-likelihood runs in EXP-050 report point forecasts only (MISSING for the corrected data).

## Anchor-head formula (checked against `seirgnn2/models.py::_physics`, lines 351-358)

Code, with `p_z` the persistence column in fold-z log1p space (`std`, `mean` from the training weeks), `st0[..., 0]` the susceptible fraction S_i at the start of the window, `pop` the district population N_i, and `rho` the reporting rate (1/11, optionally rescaled by exp(rho_scale) when `state_fit`):

    last = clamp_min(expm1(p_z[..., :1] * std + mean), 0.5)
    lam_hat = last / (7 * rho * pop * S)
    lam = lam_hat * exp(clamp(raw, -10, 10))

LaTeX:

```latex
\hat{\lambda}_{i} \;=\; \frac{\max\!\left(c_{i,t-1},\, 0.5\right)}{7\,\rho\, N_i\, S_i(t)},
\qquad
\lambda_{i,h} \;=\; \hat{\lambda}_{i}\,\exp\!\big(\mathrm{clip}(r_{i,h},\,-10,\,10)\big)
```

where c is the last reported weekly count, rho the reporting fraction, N_i the district population, S_i(t) the susceptible fraction at the start of the window, r_{i,h} the network output for district i and horizon week h, and lambda a per-day per-susceptible rate. The factor 7 converts days to a week: lambda_hat is the constant daily rate whose seven days of infections, reported at rho, add up to last week's count. Then the code multiplies lambda by a spatial import factor exp(0.1 beta times the within-window-centred log of neighbour infectious fraction), and integrates the SEIR system for 7 substeps per week with omega = 0.7/7 per day and gamma = 1/7 per day.
**The draft's wording "lambda scaled to the rate that sustains last week's count" is correct.** The formula must not be written as lambda_persist * exp(raw) without the floor and the clip. Note that `raw = 0` reproduces last week's count only approximately (the SEIR state does not start in equilibrium), the code comment says "about persistence".

## GAN augmentation and seroprevalence

- **GAN augmentation was tested, three times, and not adopted.**
  1. EXP-042 (main, `seirgnn2/results/augment.json`, 84 rows, P3, rebuilt data): TimeGAN and an SEIR simulator. TimeGAN reproduces the bulk of the data but loses the tail (largest synthetic week 882 in the fidelity check vs real 2,631; EXP-050 reports 2,036 vs 2,631). In the EXP-050 rerun: G2 TimeGAN val 16.11 / test 38.06 against B 15.51 / 37.54. No arm beat the control.
  2. `origin/Maleesha-Dev:results/phase4_gan_results.csv` (legacy array, 3 folds, 3 seeds, WGAN-GP): **harmful**. Fold 2 RMSE 124.13 (WGAN-GP) against 43.71 (none) and 41.74 (jitter + warp); fold 1 81.92 vs 27.63; fold 3 81.72 vs 66.93. Persistence 26.88 / 38.89 / 68.62. Commit 5cc903b.
  3. `origin/Maleesha-Dev:results/master_ablation_table.csv`: GCN baseline 45.03, + adaptive 45.11, + physics loss 45.06, + augmentation 45.46, all three 45.72, persistence 44.80.
- **Seroprevalence was used in validation only.** EXP-058 (`analysis/_build/run_s8_seroprevalence.py`): Spearman rho = 0.6833, p = 0.0424 across nine districts with cumulative implied infection at week 496 (2022-12-17), reporting fraction 1/11. It supersedes EXP-030 (rho 0.25, p 0.52), which used nine values typed inline, four of them for districts the survey never sampled. Under rule R4 the survey is validation-only. **No final model uses seroprevalence as input.**
- **Anything newer than EXP-061:** none found. `origin/main` and `origin/fix/seir-adaptive-audit` stop at EXP-061. The unmerged branches carry only EXP-001 to EXP-005 era entries (their own numbering). The pre-registered `audit()` grid (46 configs, about 2.5 hours) is stated as "not yet run" in the audit sheet.

---

# Part B. Ledger

Columns: ID | what | value | metric, horizon | protocol | source | commit | status. Per-horizon values are the mean of that horizon's RMSE over runs. "n" = number of runs.

## B1. Data and settings

| ID | What | Value | Metric / horizon | Protocol | Source | Commit | Status |
|---|---|---|---|---|---|---|---|
| EV-001 | Corrected series length | 559 weeks, 2013-W26 to 2024-W10, 7 missing weeks left as NaN, 25 districts | n/a | rebuilt | `full_paper/outputs/kaggle_run/notebook_output.txt` l.17 | 65c437c | VERIFIED |
| EV-002 | Legacy array | (459, 25, 11), 2013 to 2022, target = channel 5 (cases); 11 channels incl. 3 air-temperature, humidity, soil moisture, canopy, 3 precipitation, min NDVI | n/a | legacy | `docs/DATA.md` | cf2d43e | LOG-ONLY |
| EV-003 | Causal lags | cases 1 week, climate 2 weeks, NDVI as-of 0, population 0 | n/a | rebuilt | `analysis/lib/corrected_data.py` LAGS; `tests/test_no_future_leakage.py` | 8373b8c | VERIFIED (code) |
| EV-004 | Lag-1 predictability | r² 0.8893; best causal climate r² 0.0291 | r² | rebuilt | `kaggle_run/results.json` | 65c437c | VERIFIED |
| EV-005 | Same on benchmark array | 0.85 and 0.02 | r² | legacy | `CLAUDE.md` | n/a | LOG-ONLY (conflicts in value with EV-004 only because the data differ) |
| EV-006 | Legacy persistence with / without week-395 spike | 44.7953 / 29.5210 | RMSE | PL | `kaggle_run/results.json` array_floor; `docs/ARRAY_AUDIT.md` | 8373b8c | VERIFIED |
| EV-007 | Window and horizon | window 3 weeks, horizon 3 weeks | n/a | all | `seirgnn2/core.py` l.28 | 91d0778 | VERIFIED (code) |
| EV-008 | Fold construction | cut = origin x (number of windows); 30 validation windows before the cut; test = next `test_frac` of windows; windows touching a missing week dropped; mean/std of log1p cases from training weeks only | n/a | P3/P9/PK9 | `seirgnn2/core.py::build_folds` | 91d0778 | VERIFIED (code) |
| EV-009 | Training settings | Adam, lr 3e-3, weight decay 1e-4, batch 32, patience 40, 300 epochs (400 for NB arms), hidden 64, 2 layers, dropout 0.1, loss mse on z-scored log1p | n/a | P3/P9 | `seirgnn2/train.py` defaults; EXP-050 config | 91d0778 (train.py), 65c437c (run) | VERIFIED (code), LOG-ONLY (epochs for NB) |
| EV-010 | Graph | 25 districts; border adjacency, 141 directed edges with self loops. **Contains two non-real edges** (Kandy to Ampara, Kegalle to Kalutara) on main; corrected file has 139 | n/a | rebuilt | `notebooks/baseline/sri_lanka_adj_list.json`; `praveen:` same path; commit 6273b45 | 6273b45 | CONFLICT (main vs branch) |
| EV-011 | SEIR constants | omega = 0.7/7 per day, gamma = 1/7 per day, rho = 1/11, 7 substeps per week | n/a | rebuilt | `seirgnn2/models.py::_physics`; EXP-058 | 91d0778 | VERIFIED (code) |
| EV-012 | Likelihood | `dist="nb"`: negative binomial mean mu is the point forecast; `point`: squared error on log1p | n/a | rebuilt | `seirgnn2/models.py` DISTS comment | 91d0778 | VERIFIED (code) |
| EV-013 | Adaptive graph settings (post-fix) | Graph WaveNet init, randn embeddings of 10 dims, one shared adjacency, softmax(ReLU(E1 E2^T)), embeddings exempt from weight decay | n/a | P9 | `seirgnn2/models.py` l.70-110, `train.py::_optimizer` | 91d0778 | VERIFIED (code) |
| EV-014 | Phase-2 settings (adaptive graph + spatial loss) | emb_dim 10, hidden 64, dropout 0.1, lr 1e-3, wd 5e-4, 120 epochs, patience 25, residual, log transform, grad clip 5, lambda_phys in {0, 0.01, 0.1, 1.0}, 3 folds x 3 seeds | n/a | PL | `docs/EXPERIMENT_LOG.md` EXP-004 config | 91d0778 (file); run on the removed Phase-2 code | LOG-ONLY |
| EV-015 | WGAN-GP settings | conditional generator and critic with gradient penalty; GAN epochs default 40 per fold; downstream 120 epochs; seeds 0,1,2. Network widths, penalty weight: NOT FOUND in a log (read from code if needed) | n/a | PL | `origin/Maleesha-Dev:scripts/run_phase4_gan.py`, `src/dengue_gnn/gan.py` | 5cc903b | VERIFIED (script defaults), partial |
| EV-016 | TimeGAN settings | NOT FOUND in a config file in this audit; the EXP-042 entry gives fidelity numbers only | n/a | P3 | `docs/EXPERIMENT_LOG.md` EXP-042 | e44c1cd | MISSING |
| EV-017 | Seasonal features | sin/cos of day-of-year at 1 and 2 cycles per year (4 channels). Code comment cites "EDA finding F9", which was **retracted** (shape r = -0.065, 0/25 districts significant). Cite the empirical gain, not F9 | n/a | all | `seirgnn2/core.py::seasonal_features`; `CLAUDE.md` | 91d0778 | CONFLICT (rationale vs retraction) |

## B2. Persistence and the original GNN baseline at every horizon

Mean RMSE per horizon, raw counts. sd over runs (persistence: over origins).

| ID | Model | h1 | h2 | h3 | Overall test RMSE | Protocol | Source | Commit | Status |
|---|---|---|---|---|---|---|---|---|---|
| EV-020 | **Persistence** | 29.62 ± 9.85 | 35.13 ± 11.74 | 42.11 ± 15.38 | **36.02 ± 12.41** (val 17.86 ± 5.41, MAE 15.97) | P3, n = 3 origins | derived: `seirgnn2.sweep.persistence_rows()`; overall also in `kaggle_run/results.json` | 65c437c | VERIFIED (overall), derived (horizons) |
| EV-021 | **Persistence** | 23.47 ± 9.56 | 27.94 ± 12.38 | 33.26 ± 15.71 | **28.54 ± 12.71** (val 27.04 ± 12.40, MAE 13.94 ± 6.28) | P9, n = 9 origins | `seirgnn2/results/frozen9_plus_audit.json` | 91d0778 | VERIFIED |
| EV-022 | Persistence | n/a | n/a | n/a | 31.76 (val 41.21) | PK9 | `kaggle_run/results.json` persistence_9 | 65c437c | VERIFIED (horizons MISSING) |
| EV-023 | **Persistence** (legacy) | 39.34 ± 24.87 | 44.11 ± 22.09 | 49.79 ± 18.73 | 44.41 mean over folds (pooled 44.80) | PL, 3 folds x 3 seeds listed | `praveen:results/baseline_rolling_origin.csv` | d0143ae | VERIFIED. Per-fold h1-h3: fold 1 21.74/25.73/32.15; fold 2 28.49/37.98/47.77; fold 3 67.78/68.61/69.45 |
| EV-024 | **Original-style GCN baseline** (residual + log, graph = border adjacency) | 40.92 ± 20.88 | 46.33 ± 18.94 | 51.53 ± 16.09 | 46.26 mean over folds (log says 45.4 and 45.03 in other aggregations) | PL | `praveen:results/baseline_rolling_origin.csv` (EXP-002 label) | d0143ae | VERIFIED; **CONFLICT** across aggregations (log table 45.4, master ablation 45.03) |
| EV-025 | GCN direct (toy encoder, fixed graph) | 23.10 ± 9.79 | 27.82 ± 12.82 | 32.68 ± 16.01 | 28.17 ± 13.04 (val 25.64, MAE 13.66) | P9, n = 27 | `frozen9_plus_audit.json` | 91d0778 | VERIFIED. Δ vs persistence -0.37, 5/9 origins, p_adj 0.984 |
| EV-026 | GCN residual | 23.24 ± 9.95 | 27.79 ± 12.51 | 32.47 ± 15.48 | 28.12 ± 12.77 (val 25.55) | P9, n = 27 | same | 91d0778 | VERIFIED. Δ -0.42, 6/9, p_adj 0.781 |
| EV-027 | Graph control, toy encoder, 3 origins: graph = none / gcn / adaptive | test 34.57 ± 11.09 / 34.50 ± 10.41 / 34.17 ± 10.72; h1 28.54 / 28.28 / 28.44; h2 34.05 / 34.01 / 33.97; h3 40.08 / 40.13 / 39.20 | RMSE | P3, n = 9 each | `kaggle_run/runs.jsonl` (derived means) | 65c437c | VERIFIED. The graph is worth about 0.07 RMSE over no graph (EXP-050 log); within the spread |
| EV-028 | Published encoders, direct head, 3 origins: STGAT / A3TGCN / DCRNN broken | test 62.00 ± 25.54 / 60.40 ± 24.13 / 73.52 ± 32.90 (val 22.93 / 28.63 / 34.64) | RMSE | P3, n = 9 | `kaggle_run/runs.jsonl` (derived) | 65c437c | VERIFIED. h1-h3 STGAT 60.24/61.97/63.67 |
| EV-029 | Working published encoders, direct head: AAGCN / ASTGCN / LSTM | test 35.15 ± 11.53 / 35.61 ± 11.82 / 35.10 ± 11.84 (val 16.71 / 16.83 / 16.91); h1/h2/h3 AAGCN 29.34/34.79/40.36, ASTGCN 29.40/35.23/41.14, LSTM 28.89/34.66/40.66 | RMSE | P3, n = 9 | `kaggle_run/runs.jsonl` (derived) | 65c437c | VERIFIED. 0.4 to 0.9 below persistence 36.02 on test; no paired test at the origin unit is recorded for these arms, three origins cannot reach p < 0.25 |

## B3. Headline results on the corrected data (P3, EXP-050; n = 9 runs each)

| ID | What | Value | Metric / horizon | Protocol | Source | Commit | Status |
|---|---|---|---|---|---|---|---|
| EV-030 | Gated head, AAGCN (best test of the 30 encoder x head configs) | test 33.53 ± 10.14, val 16.57; h1/h2/h3 27.78/32.89/38.93 | RMSE | P3 | `runs.jsonl` derived; `validation_vs_test.csv` | 65c437c | VERIFIED. Validation picks "AAGCN+gated" and it is also best on test (rank corr val/test 0.873) |
| EV-031 | B = AAGCN direct, NB likelihood + season | val 15.51, test 37.54 ± 13.90; h1/h2/h3 32.17/37.91/41.79; MAE 16.46 | RMSE | P3 | `runs.jsonl` derived | 65c437c | VERIFIED. Validation picks B in 3 of 8 families and **B is worse than persistence on test (37.54 vs 36.02)** |
| EV-032 | B, squared-error twin | val 16.27, test 35.94 ± 12.14 | RMSE | P3 | same | 65c437c | VERIFIED |
| EV-033 | NB vs squared error (B) | val -0.7626, 9/9, p_adj 0.0137; test +1.61 | ΔRMSE | P3, unit origin_seed | `comparisons`/EXP-050 log | 65c437c | VERIFIED but **unit is origin_seed, not valid alone**; sign reverses on test |
| EV-034 | B vs persistence | val -2.3462 (9/9 origin_seed, p 0.0039); test +1.5274 (3/9, p 0.23). Nine origins: val -2.5784 (8/9, origin unit p 0.105); test +2.2390 (4/9, p 0.293) | ΔRMSE | P3 / PK9 | `kaggle_run/comparisons.csv` | 65c437c | VERIFIED. **Validation gain does not transfer to test** |
| EV-035 | Bare SEIR head (foi) vs direct | AAGCN 49.54, ASTGCN 51.75, LSTM 50.26, STGAT 52.37, A3TGCN 52.92, DCRNN 52.26 test; val deficit +7.30 / +7.38 / +7.41 on working three (0/9) | RMSE | P3 | `runs.jsonl`; log | 65c437c | VERIFIED |
| EV-036 | Gated SEIR (foi_res) repair of broken encoders | STGAT 35.84, A3TGCN 35.50, DCRNN 35.52 vs direct 62.00 / 60.40 / 73.52; val -5.30 / -10.95 / -16.96 (9/9) | RMSE | P3 | `runs.jsonl`; EXP-050 | 65c437c | VERIFIED |
| EV-037 | Rescue control: part of the repair due to the physics | credited to physics: none (`credited_to_physics: []`); share from anchor+gate 1.124 / 1.000 / 0.960 | share | P3 | `kaggle_run/results.json`, `rescue.csv` | 65c437c | VERIFIED |
| EV-038 | Gated SEIR-GNN vs SEIR-LSTM | 3 origins val ASTGCN -0.5351 (9/9), AAGCN -0.1402 (8/9), test +0.136 / +0.021 (5/9). Nine origins (origin unit) ASTGCN val -0.6180 (8/9, p 0.0117), test +0.0666 (6/9, p 0.88); AAGCN val +0.1834, test -0.3034 (p 0.078) | ΔRMSE | P3 / PK9 | `kaggle_run/comparisons.csv` | 65c437c | VERIFIED. Graph advantage is small and does not hold on both splits |
| EV-039 | Validation vs test agreement | In 5 of 8 families the validation pick is worse than persistence on test; in 6 of 8 the best-on-test config is anchored to last week | count | P3 | `kaggle_run/validation_vs_test.csv`; EXP-050 | 65c437c | VERIFIED |
| EV-040 | Interim audit (see C5/C6 table below) | | | P9 | `seirgnn2/results/frozen9_plus_audit.json` | 91d0778 | VERIFIED |

## B4. P9 audit table (EXP-059 + EXP-061; 9 origins, n = 27 runs per arm; paired test on origin means, BH over 30 arms)

Persistence: val 27.04, test 28.54, MAE 13.94. Test sd over 27 runs.

| ID | Arm | val | Δval (p_adj) | test ± sd | Δtest (p_adj) | h1 / h2 / h3 (test) | Status |
|---|---|---|---|---|---|---|---|
| EV-041 | adaptive_gwn + residual | 25.42 | -1.62 (0.023) | 27.94 ± 12.91 | -0.60 (0.694) | 23.07 / 27.78 / 32.15 | VERIFIED |
| EV-042 | adaptive_gwn + foi_res anchor E0enc | 25.53 | -1.51 (0.047) | 27.45 ± 11.98 | -1.10 (0.576) | 22.45 / 27.07 / 31.88 | VERIFIED |
| EV-043 | adaptive_gwn + foi_res anchor | 25.74 | -1.30 (0.090) | 27.45 ± 12.24 | -1.09 (0.576) | 22.51 / 27.09 / 31.85 | VERIFIED |
| EV-044 | gcn + foi_res anchor | 25.83 | -1.20 (0.158) | 27.62 ± 12.26 | -0.92 (0.576) | 22.49 / 27.38 / 32.04 | VERIFIED |
| EV-045 | adaptive_gwn + foi_res mass | 26.93 | -0.11 (1.000) | 28.39 ± 12.94 | -0.16 (0.984) | 22.90 / 28.02 / 33.19 | VERIFIED |
| EV-046 | adaptive_gwn + foi anchor (bare) | 27.99 | +0.95 (0.591) | 29.80 ± 14.74 | +1.26 (0.781) | 23.99 / 28.81 / 35.29 | VERIFIED |
| EV-047 | gcn + foi (pre-fix, bare) | 38.08 | +11.04 (0.023) | 40.77 ± 22.01 | +12.23 (0.033) | 27.58 / 40.81 / 50.14 | VERIFIED |
| EV-048 | ASTGCN + residual (best val) | 25.23 | -1.80 (0.023) | 28.65 ± 13.78 | +0.11 (0.984) | 23.07 / 28.08 / 33.66 | VERIFIED |
| EV-049 | LSTM + direct | 25.45 | -1.59 (0.023) | 28.73 ± 14.27 | +0.19 (0.984) | 23.65 / 28.22 / 33.38 | VERIFIED |
| EV-050 | Adaptive graph entropy (1.0 = uniform) | mean 0.4693, min 0.3150, max 0.6182, n = 135 runs | normalized row entropy | P9 | `adj_entropy` in `frozen9_plus_audit.json` | 91d0778 | VERIFIED |

Everything in B4 beats persistence by at most 1.1 RMSE on test, with p_adj above 0.5. The pre-fix bare foi arms are significantly worse. Persistence is the hard floor.

## B5. Experiments with a different design (negative or null; protocols as stated)

| ID | What | Value | Metric | Protocol | Source | Commit | Status |
|---|---|---|---|---|---|---|---|
| EV-060 | EXP-060 FoI regression, reachable cells, features: log cases / + log pop / + log pop and log S | 0.363,0.384,0.353 / 0.512,0.515,0.444 / 0.589,0.660,0.674 (origins 0.55, 0.70, 0.85). All cells: 0.216-0.259 / 0.218-0.260 / 0.224-0.273 | out-of-sample r² | three origins, training windows | EXP-060; `seirgnn2/diagnose_foi.py` | 91d0778 | LOG-ONLY |
| EV-061 | Share of cells pinned at lambda = 0 | 39-41% | share | same | EXP-060; `seirgnn2/diagnose_foi.py` | 91d0778 | LOG-ONLY |
| EV-062 | Renewal-physics test: log R_t predictability | own past 0.263 (0.365 independent replication); climate -0.025 / -0.028 | r², contiguous blocks | rebuilt, 8005 district-weeks | `praveen:docs/decisions/0005...` quoting EXP-024 | 2fd689d | LOG-ONLY (EXP-024 in the log; replication on branch) |
| EV-063 | Climate lags 2-4 / 2-13 / 2-25 added to B | test 38.05 / 37.49 / 37.07 vs B 37.54 | RMSE | P3 | `runs.jsonl` | 65c437c | VERIFIED. No gain |
| EV-064 | Spatial penalty P1 / P2; metapopulation SEIR P4 | 37.52 / 37.72 / 35.03 | RMSE | P3 | `runs.jsonl` | 65c437c | VERIFIED. P4 val 17.34 vs B 15.51 |
| EV-065 | Data: TimeGAN G2 / SEIR pretraining G3 | val 16.11 / 15.54, test 38.06 / 38.46 vs B 15.51 / 37.54 | RMSE | P3 | `runs.jsonl` | 65c437c | VERIFIED. No gain |
| EV-066 | Learning curve of B at 25 / 50 / 75 / 100% of training | test 36.54 / 38.01 / 37.40 / 37.54 | RMSE | P3 | `runs.jsonl` | 65c437c | VERIFIED. More data does not help |
| EV-067 | k-NN (R4b) and gradient boosting (K6, K7) | test 34.34 ± 10.14; 36.49; 38.96 | RMSE | P3 | `runs.jsonl` | 65c437c | VERIFIED. k-NN is the best single non-neural arm on test |
| EV-068 | COVID policy and mobility covariates | base 30.0487, policy 30.0637, policy_mobility 30.0693; Holm p 1.0, 1.0. Run-to-run floor about 0.11 | test RMSE, 9 origins x 3 seeds | P9-like (9 origins 0.50-0.90, SEIR-STGAT v2, 400 epochs) | `results/covid_covariates_summary.csv`; EXP-056 | 7af8226 | VERIFIED |
| EV-069 | Movement-restriction prior | base 30.0744; prior g0.5/g1/g2 30.5934 / 31.2768 / 32.4286; p 0.14 / 0.047 / 0.031 (significantly worse) | test RMSE | same | `analysis/results/seir_gnn/s5_v2/stringency_prior_runs.csv`; EXP-057 | 8346b4d | VERIFIED (CSV) |
| EV-070 | Seroprevalence rank check | Spearman rho 0.6833, p 0.0424, 9 districts | rho | week 496 | EXP-058; `data/external/seroprevalence_nine_districts.csv`; `analysis/_build/run_s8_seroprevalence.py` | 8373b8c / 8b75c94 | LOG-ONLY. Validation-only |
| EV-071 | SEIR layer vs same encoder without it (old v2 SEIR-STGAT, MSE) | with SEIR 30.04, without 49.45, Δ -19.41, 8/9 origins, p 0.0078 | test RMSE | rebuilt, 9 disjoint origins x 3 seeds, different harness from P9 | EXP-052; `analysis/results/seir_gnn/s5_v2/` | f104646 | LOG-ONLY. Compares against a no-physics arm that fails to extrapolate, not against persistence (30.04 vs 28.54 persistence) |
| EV-072 | Damped mechanistic correction | 29.15 vs 30.01 undamped; with waning 28.74 / MAE 14.14 vs persistence 28.54 / 13.94 (+0.20, p 0.90) | test RMSE | same family | EXP-054; `persistence_plus_runs.csv` | 2dd4e76 | LOG-ONLY (CSV exists) |
| EV-073 | Horizon sweep physics vs persistence | h3 29.60 vs 28.54; h6 38.32 vs 36.04; h12 46.31 vs 45.08 | test RMSE | same family | EXP-053; `horizon_sweep_runs.csv` | 606c121 | LOG-ONLY (CSV exists) |

## B6. Legacy-array (PL) experiments from the proposal era

| ID | What | Value | Metric | Protocol | Source | Commit | Status |
|---|---|---|---|---|---|---|---|
| EV-080 | Lambda_phys sweep (spatial loss) | 0.0: 44.85; 0.01: 44.80; 0.1: 44.78; 1.0: 45.24. Fold 2: 39.95 / 39.68 / 38.65 / 38.99 | RMSE | PL | `praveen:results/exp004_lambda_sweep.csv` | d0143ae | VERIFIED. Differences of 0.07 are inside any seed spread |
| EV-081 | Adaptive GCN (EXP-003) vs GCN baseline vs persistence, fold 1/2/3 | adaptive 27.15 / 39.95 / 67.44; GCN 26.96 / 42.71 / 69.12; persistence 26.54 / 38.08 / 68.62 (means over seeds and horizons) | RMSE | PL | `praveen:results/exp003_adaptive_gcn.csv`, `baseline_rolling_origin.csv` | d0143ae | VERIFIED (derived means) |
| EV-082 | PIAG-Net (adaptive + physics 0.1) fold 1/2/3 | 27.27 / 38.65 / 68.42; mean 44.78 | RMSE | PL | `praveen:results/exp005_combined_proposed.csv` | d0143ae | VERIFIED |
| EV-083 | Corrected-adjacency rerun, 5 configs | pooled adaptive 44.64, dense_fixed 45.47, adaptive lam 0.01 / 0.1 / 1.0 = 45.50 / 46.04 / 67.39; persistence 44.80. Per horizon (adaptive) 39.28 / 44.23 / 49.24 | RMSE | PL, adjacency without 2 edges | `origin/adaptive-graph:results/phase2_runs.csv` (192 rows) | 22e6404 | VERIFIED. Paired t vs fixed graph p about 0.10-0.14 (commit message), not significant |
| EV-084 | Phase-3 Stage C (SEIR-SEI GNN, legacy array, origins 0.70 and 0.85, 40 epochs) | AdaptiveGCN 53.32 test / 20.33 val; persistence 53.75 / 20.93; SEIR-GNN lambda 0: 68.35 / 31.52 ± 10.44; lambda 1.0: 71.85 / 33.83 ± 12.48 | RMSE | PL, 2 origins | `praveen:results/phase3_stageC.json`; ADR 0005 | abf3416 | VERIFIED (quoted in ADR; JSON present) |
| EV-085 | Phase-1 per-origin baselines (Maleesha run) | persistence 26.88 / 38.89 / 68.62; AdaptiveGCN 28.84 / 39.73 / 67.97; PIAG-Net combined 29.52 / 40.91 / 68.04 | RMSE | PL, 3 folds | `origin/Maleesha-Dev:results/phase2_adaptive_results.csv`, `phase3_physics_results.csv` | 5cc903b | VERIFIED. **CONFLICT** with EV-081/082 (same protocol label, different numbers: this is a different code path, run by a different person) |
| EV-086 | WGAN-GP augmentation | fold 1/2/3: none 27.63 / 43.71 / 66.93; jitter+warp 29.97 / 41.74 / 65.32; WGAN-GP 81.92 / 124.13 / 81.72 | RMSE | PL | `origin/Maleesha-Dev:results/phase4_gan_results.csv` | 5cc903b | VERIFIED |
| EV-087 | Master ablation | persistence 44.80; GCN 45.03; + adaptive 45.11; + physics 45.06; + augmentation 45.46; all 45.72 (MAE 15.72 / 15.92 / 15.93 / 15.84 / 16.09 / 15.92) | RMSE | PL | `origin/Maleesha-Dev:results/master_ablation_table.csv` | 5cc903b | VERIFIED |
| EV-088 | Probabilistic head coverage and width (five architectures) | PICP 0.947-0.962; MPIW 122.8-249.9 | coverage, cases | PL | `analysis/results/improved_sweep.md` | 2fd689d | VERIFIED |
| EV-089 | Reproduction of Weng et al. STGAT | 24.01 ± 1.32 MAE / 42.31 ± 2.47 RMSE vs published 25.38 / 44.78; persistence on identical slices 18.58 / 34.67 | MAE / RMSE | published protocol | `crosscheck/FINDINGS.md` F1.0 | cf2d43e | VERIFIED (document) |
| EV-090 | Old S5 leaderboard (defective; **do not cite**) | persistence 33.87, LSTM 65.91, A3T-GCN 63.15, STGAT 83.99, SEIR-GNN 71.80 | test RMSE | PL-S5 | deleted `full_paper/outputs/csv/stage_s5_leaderboard.csv` (65c437c) | 65c437c | CONFLICT with EV-035..036. Excluded |

# Part D. Code constants checked by reading the source (IEEE draft)

Each row was read from the file named; nothing was run. Status VERIFIED (code).

| ID | What | Value | Source | Status |
|---|---|---|---|---|
| EV-170 | SEIR initial state ("lagged" seeding): I0 = c(t-1)/(rho N), E0 = c(t-2)/(rho N), both clamped to [1e-6, 0.5]; S0 = clamp(s0 - cumulative cases/(rho N), 0.01, 1) with s0 = 1 - 0.682 = 0.318; R0 = 1 - S0 - E0 - I0 | as stated | `seirgnn2/models.py::seir_state` | VERIFIED (code) |
| EV-171 | Free-rate force of infection: lambda = exp(clamp(r + b, -25, log(1/7))), b initialised at -9.36 | as stated | `seirgnn2/models.py::_physics`, LAM_LOG0, LOG_LAMBDA_MAX | VERIFIED (code) |
| EV-172 | Anchored force of infection: lambda_hat = max(c(t-1), 0.5)/(7 rho N S0); lambda = lambda_hat exp(clamp(r, -10, 10)) | as stated | `seirgnn2/models.py::_physics` l.351-358 | VERIFIED (code) |
| EV-173 | E0 is multiplied by 2 sigmoid(g), g a learned scalar (state_fit True) or a linear function of the district encoding, zero-initialised (state_fit "encoder"); rho is multiplied by exp(learned scalar) when state_fit is on | as stated | `seirgnn2/models.py::_seed`, `Net.__init__` | VERIFIED (code) |
| EV-174 | Spatial import: lambda is multiplied by exp(0.1 beta_s x), x = log of the neighbour-weighted infectious fraction, centred over districts within the window; beta_s a learned scalar initialised at 0 | as stated | `seirgnn2/models.py::_physics` | VERIFIED (code) |
| EV-175 | Training: Adam, lr 3e-3, weight decay 1e-4, batch 32, cosine annealing over the epoch budget, gradient clipping at 5, early stopping on validation RMSE in counts with patience 40 and restore of the best weights, 300 epochs (400 for the NB runs) | as stated | `seirgnn2/train.py::run_fold`; EV-009 | VERIFIED (code) |
| EV-176 | NB head: mu = expm1(clamp(z_hat sigma + mu_bar, -1, 12)); alpha = 1e-4 + (2 - 1e-4) sigmoid(d), d a linear read-out; NB2 negative log-likelihood, Var = mu + alpha mu^2 | as stated | `seirgnn2/train.py::_loss`, `analysis/lib/count_loss.py::nb_nll` | VERIFIED (code) |
| EV-177 | Architecture sizes: hidden 64, 2 graph layers, dropout 0.1, input projection Linear-ReLU-Dropout, residual connection around each graph layer, adaptive embeddings of 10 dimensions drawn N(0,1), shared by all layers, no weight decay on the embeddings | as stated | `seirgnn2/models.py::Net, GraphLayer, adaptive_embeddings`, `train.py::_optimizer` | VERIFIED (code) |
| EV-178 | Gate: gated and SEIR-gated heads output p + sigmoid(a)(x - p), a a learned scalar initialised at -2 | as stated | `seirgnn2/models.py::Net.forward` | VERIFIED (code) |
| EV-179 | Nine-origin arms use only the 3 weeks of log cases (no climate, no NDVI, no season). The 30 arms are 6 encoders (GCN, LSTM, STGAT, A3TGCN, ASTGCN, AAGCN) x 4 heads (direct, residual, SEIR free rate, gated SEIR free rate) plus the 6 hand-picked post-fix arms | as stated | `seirgnn2/grids.py::audit_quick`, `train.py` defaults; EXP-059, EXP-061 | VERIFIED (code and log) |
| EV-180 | Model B = AAGCN encoder, direct head, NB likelihood, 4 seasonal Fourier features (sin, cos of day-of-year at 1 and 2 cycles per year), 400 epochs | as stated | `full_paper/kaggle/src/70_experiments.py` | VERIFIED (code) |
| EV-181 | Smallest attainable two-sided exact sign-flip p: 0.25 with 3 origins, 0.0039 with 9 origins | 0.25; 0.0039 | `seirgnn2/stats.py` docstring and `sign_flip_p`; EXP-050 comparisons.csv (p = 0.00390625) | VERIFIED (code) |
| EV-182 | Fold construction: windows indexed in time; cut = origin x number of windows; the 30 windows before the cut are validation; test is the next fraction of windows; windows touching a missing week are dropped | as stated | `seirgnn2/core.py::build_folds` | VERIFIED (code) |
| EV-183 | Row-normalised border adjacency with self-loops, 25 districts | as stated | `seirgnn2/core.py::adjacency` | VERIFIED (code) |
| EV-184 | The six post-fix arms were chosen by the person auditing the code, not blind to likely RMSE, not pre-registered; the 46-configuration audit grid was not run | as stated | `docs/EXPERIMENT_LOG.md` EXP-061 notes; `docs/SEIR_ADAPTIVE_AUDIT_RESULTS.md` section 8 | LOG-ONLY |
| EV-185 | Strongest learned-graph edges in the audit (adaptive encoder, all runs), as strings: Kalutara<-Kandy 0.918, Batticaloa<-Puttalam 0.812, Jaffna<-Kegalle 0.722, Trincomalee<-Kurunegala 0.912, Anuradhapura<-Puttalam 0.896; none of these pairs share a border | as stated | `docs/SEIR_ADAPTIVE_AUDIT_RESULTS.md` section 6 (field `adj_top` in `frozen9_plus_audit.json` holds the same strings) | LOG-ONLY |
| EV-186 | Protocol definitions: nine origins 0.50 to 0.90 in steps of 0.05 with test fraction 0.05 (the frozen protocol); three origins 0.55, 0.70, 0.85 with test fraction 0.15; seeds 0, 1, 2 | as stated | `seirgnn2/core.py` ORIGINS, TEST_FRAC, ORIGINS_F9, TEST_FRAC_F9; `docs/PROTOCOL.md` | VERIFIED (code) |
| EV-187 | Derived arithmetic: spread of the three graph controls on three origins, max minus min of test RMSE = 34.5728 - 34.1668 = 0.4060 (graph=none, graph=gcn 34.5010, graph=adaptive) | 0.41 | from EV-116, EV-117, EV-118 | derived |
| EV-188 | Interval baseline method: 2.5th and 97.5th percentiles of training-window residuals y[t+h] - y[t-1], one pair per horizon, pooled over districts, lower bound clipped at 0; scaled version divides residuals by sqrt(1 + y[t-1]) and multiplies back | as stated | `paper/ieee/scripts/interval_baseline.py` | VERIFIED (code) |
| EV-189 | Derived arithmetic: variance of this week's cases not explained by last week's cases, 1 - 0.8893 = 0.1107 | 11% | from EV-004 | derived |

| EV-191 | Baseline GNN design: residual over persistence, log1p target, rolling-origin CV, grad clip 5, training-fold normalisation (the v2 baseline). The v1 plain GCN on absolute counts lost to persistence on every metric. The v2 baseline matched, not beat, the persistence floor | as stated | `docs/decisions/0001-baseline-training-refinements.md`; `notebooks/baseline/dengue_baseline_GNN_v2.ipynb` | LOG-ONLY (document) |

| EV-192 | Week 395 of the legacy array is a source-table error, not a backlog: in the published table for Vol. 48 No. 2 (26 Dec 2020 to 1 Jan 2021) the weekly row has 15 of its 24 inner cells equal to the sum of the two cells before them (a spreadsheet formula error); its district sum is 7165; the year-to-date row, which equals the weekly count in the first week of the year, sums with the Kalmunai division to 351, the published national total | 15 of 24; 7165; 351 | `data/external/report_corrections.json` (fields evidence, reason); `analysis/_build/build_corrected_cases.py` docstring and `extract_week1_correction` | VERIFIED (saved file) |

<!-- BEGIN GENERATED -->

# Part C. Rows derived by paper/ieee/scripts (IEEE draft)

Generated by `paper/ieee/scripts/make_tables.py` from the saved files named in each row. Do not edit by hand; rerun the script. Persistence per horizon for the three-origin protocol is copied from EV-020 (LOG-ONLY).

| ID | What | Value | Protocol | Source | Status |
|---|---|---|---|---|---|
| EV-100 | Persistence, 9 origins: val / h1 / h2 / h3 / test (mean ± sd over 9 origins) | 27.0363 / 23.4735 / 27.9365 / 33.2566 / 28.5410 ± 12.7078 | P9 | frozen9_plus_audit.json (derived) | VERIFIED (derived by script) |
| EV-101 | GCN, direct: val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val) | 25.6381 / 23.0968 / 27.8210 / 32.6780 / 28.1733 ± 13.0449; -0.3677, 5/9, 0.9844; -1.3982, 0.1582 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms) | VERIFIED (derived by script) |
| EV-102 | GCN, residual (baseline GNN): val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val) | 25.5493 / 23.2428 / 27.7889 / 32.4685 / 28.1194 ± 12.7701; -0.4216, 6/9, 0.7812; -1.4870, 0.0806 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms) | VERIFIED (derived by script) |
| EV-103 | LSTM, direct: val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val) | 25.4450 / 23.6498 / 28.2151 / 33.3829 / 28.7296 ± 14.2658; +0.1885, 6/9, 0.9844; -1.5913, 0.0234 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms) | VERIFIED (derived by script) |
| EV-104 | ASTGCN, direct: val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val) | 25.3547 / 23.9231 / 28.5390 / 33.4489 / 28.9397 ± 13.7720; +0.3986, 6/9, 0.9844; -1.6816, 0.0234 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms) | VERIFIED (derived by script) |
| EV-105 | AAGCN, direct: val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val) | 25.4608 / 24.1054 / 29.0988 / 33.6634 / 29.2589 ± 14.5070; +0.7179, 5/9, 0.9651; -1.5755, 0.0320 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms) | VERIFIED (derived by script) |
| EV-106 | STGAT, direct: val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val) | 35.9871 / 42.5994 / 44.5469 / 46.7529 / 44.7128 ± 25.4718; +16.1718, 1/9, 0.0335; +8.9508, 0.0335 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms) | VERIFIED (derived by script) |
| EV-107 | ASTGCN, residual: val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val) | 25.2348 / 23.0689 / 28.0754 / 33.6600 / 28.6486 ± 13.7768; +0.1076, 6/9, 0.9844; -1.8015, 0.0234 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms) | VERIFIED (derived by script) |
| EV-108 | Adaptive GCN, residual: val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val) | 25.4177 / 23.0733 / 27.7778 / 32.1540 / 27.9450 ± 12.9089; -0.5960, 7/9, 0.6941; -1.6186, 0.0234 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms) | VERIFIED (derived by script) |
| EV-109 | GCN, SEIR, free rate: val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val) | 38.0801 / 27.5809 / 40.8144 / 50.1441 / 40.7742 ± 22.0075; +12.2331, 1/9, 0.0335; +11.0438, 0.0234 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms) | VERIFIED (derived by script) |
| EV-110 | GCN, gated SEIR, free rate: val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val) | 26.9175 / 22.8603 / 28.0361 / 33.2912 / 28.4274 ± 13.0828; -0.1136, 5/9, 0.9844; -0.1187, 1.0000 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms) | VERIFIED (derived by script) |
| EV-111 | Adaptive GCN, SEIR, anchored: val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val) | 27.9858 / 23.9920 / 28.8106 / 35.2875 / 29.7981 ± 14.7447; +1.2570, 3/9, 0.7812; +0.9495, 0.5913 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms) | VERIFIED (derived by script) |
| EV-190 | Adaptive GCN, gated SEIR, mass-action: val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val) | 26.9260 / 22.8964 / 28.0155 / 33.1881 / 28.3858 ± 12.9402; -0.1552, 5/9, 0.9844; -0.1103, 1.0000 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms) | VERIFIED (derived by script) |
| EV-112 | GCN, gated SEIR, anchored: val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val) | 25.8330 / 22.4868 / 27.3827 / 32.0368 / 27.6216 ± 12.2569; -0.9194, 7/9, 0.5762; -1.2033, 0.1582 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms) | VERIFIED (derived by script) |
| EV-113 | Adaptive GCN, gated SEIR, anchored: val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val) | 25.7367 / 22.5076 / 27.0857 / 31.8453 / 27.4496 ± 12.2434; -1.0914, 8/9, 0.5762; -1.2996, 0.0896 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms) | VERIFIED (derived by script) |
| EV-114 | Adaptive GCN, gated SEIR, anchored, learned $E_0$: val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val) | 25.5297 / 22.4472 / 27.0737 / 31.8803 / 27.4451 ± 11.9766; -1.0959, 7/9, 0.5762; -1.5066, 0.0469 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms) | VERIFIED (derived by script) |
| EV-115 | Persistence, 3 origins: val / h1 / h2 / h3 / test ± sd (3 origins) | 17.8561 / 29.6185 / 35.131 / 42.114 / 36.0157 ± 12.4088 | P3 | results.json persistence_3 (val, test); horizons LOG-ONLY from EV-020 | LOG-ONLY (horizons) |
| EV-116 | GCN, residual (graph control): val / h1 / h2 / h3 / test ± sd (9 runs); Δ vs graph=gcn, wins of 9 paired runs | 16.8375 / 28.2772 / 34.0075 / 40.1275 / 34.5010 ± 10.4140; n/a | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-117 | No graph, residual: val / h1 / h2 / h3 / test ± sd (9 runs); Δ vs graph=gcn, wins of 9 paired runs | 16.8998 / 28.5407 / 34.0474 / 40.0838 / 34.5728 ± 11.0876; +0.0718, 6/9 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-118 | Adaptive GCN (simple), residual: val / h1 / h2 / h3 / test ± sd (9 runs); Δ vs graph=gcn, wins of 9 paired runs | 16.6928 / 28.4397 / 33.9719 / 39.2049 / 34.1668 ± 10.7187; -0.3341, 7/9 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-119 | AAGCN, direct: val / h1 / h2 / h3 / test ± sd (9 runs); Δ vs graph=gcn, wins of 9 paired runs | 16.7091 / 29.3399 / 34.7894 / 40.3592 / 35.1471 ± 11.5256; +0.6461, 4/9 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-120 | AAGCN, gated: val / h1 / h2 / h3 / test ± sd (9 runs); Δ vs graph=gcn, wins of 9 paired runs | 16.5679 / 27.7845 / 32.8868 / 38.9266 / 33.5312 ± 10.1387; -0.9698, 9/9 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-121 | AAGCN, SEIR, free rate: val / h1 / h2 / h3 / test ± sd (9 runs); Δ vs graph=gcn, wins of 9 paired runs | 24.0060 / 35.0448 / 50.2914 / 59.9167 / 49.5376 ± 18.4876; +15.0366, 0/9 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-122 | AAGCN, gated SEIR, free rate: val / h1 / h2 / h3 / test ± sd (9 runs); Δ vs graph=gcn, wins of 9 paired runs | 17.4308 / 28.6832 / 35.3330 / 41.4630 / 35.5814 ± 11.5015; +1.0804, 3/9 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-123 | STGAT, direct: val / h1 / h2 / h3 / test ± sd (9 runs); Δ vs graph=gcn, wins of 9 paired runs | 22.9318 / 60.2405 / 61.9711 / 63.6701 / 61.9966 ± 25.5411; +27.4956, 0/9 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-124 | STGAT, gated SEIR, free rate: val / h1 / h2 / h3 / test ± sd (9 runs); Δ vs graph=gcn, wins of 9 paired runs | 17.6331 / 28.3388 / 35.2952 / 42.3856 / 35.8381 ± 11.6463; +1.3371, 3/9 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-125 | DCRNN, direct: val / h1 / h2 / h3 / test ± sd (9 runs); Δ vs graph=gcn, wins of 9 paired runs | 34.6368 / 73.1922 / 73.5342 / 73.8420 / 73.5243 ± 32.8980; +39.0234, 0/9 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-126 | DCRNN, gated SEIR, free rate: val / h1 / h2 / h3 / test ± sd (9 runs); Δ vs graph=gcn, wins of 9 paired runs | 17.6722 / 28.3441 / 34.9354 / 41.8782 / 35.5184 ± 11.4217; +1.0174, 3/9 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-127 | AAGCN, direct, NB + season (B): val / h1 / h2 / h3 / test ± sd (9 runs); Δ vs graph=gcn, wins of 9 paired runs | 15.5098 / 32.1715 / 37.9056 / 41.7925 / 37.5431 ± 13.9041; +3.0422, 3/9 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-128 | AAGCN, direct, squared error + season: val / h1 / h2 / h3 / test ± sd (9 runs); Δ vs graph=gcn, wins of 9 paired runs | 16.2724 / 30.5233 / 35.5707 / 40.9010 / 35.9354 ± 12.1449; +1.4344, 3/9 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-129 | B: val, test | 15.5098, 37.5431 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-130 | TimeGAN augmentation vs B (AAGCN direct NB season): val, test, Δtest vs B, wins of 9 paired runs | 16.1096, 38.0625, +0.5193, 3/9; B val 15.5098 test 37.5431 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-131 | SEIR-simulator pre-training vs B (AAGCN direct NB season): val, test, Δtest vs B, wins of 9 paired runs | 15.5371, 38.4624, +0.9193, 5/9; B val 15.5098 test 37.5431 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-132 | Climate lags 2--4 weeks vs B (AAGCN direct NB season): val, test, Δtest vs B, wins of 9 paired runs | 15.6925, 38.0501, +0.5069, 3/9; B val 15.5098 test 37.5431 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-133 | Climate lags 2--13 weeks vs B (AAGCN direct NB season): val, test, Δtest vs B, wins of 9 paired runs | 15.6952, 37.4947, -0.0484, 5/9; B val 15.5098 test 37.5431 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-134 | Climate lags 2--25 weeks vs B (AAGCN direct NB season): val, test, Δtest vs B, wins of 9 paired runs | 15.9827, 37.0711, -0.4720, 8/9; B val 15.5098 test 37.5431 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-135 | Spatial penalty vs B (AAGCN direct NB season): val, test, Δtest vs B, wins of 9 paired runs | 15.4499, 37.5221, -0.0211, 7/9; B val 15.5098 test 37.5431 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-136 | Metapopulation SEIR head vs B (AAGCN direct NB season): val, test, Δtest vs B, wins of 9 paired runs | 17.3387, 35.0289, -2.5142, 6/9; B val 15.5098 test 37.5431 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-137 | District seasonal curves vs B (AAGCN direct NB season): val, test, Δtest vs B, wins of 9 paired runs | 15.4575, 37.6538, +0.1106, 7/9; B val 15.5098 test 37.5431 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-138 | Half of the training windows vs B (AAGCN direct NB season): val, test, Δtest vs B, wins of 9 paired runs | 15.9016, 38.0097, +0.4666, 4/9; B val 15.5098 test 37.5431 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-139 | RevIN normalization vs B (AAGCN direct NB season): val, test, Δtest vs B, wins of 9 paired runs | 16.9756, 37.1233, -0.4198, 5/9; B val 15.5098 test 37.5431 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-140 | District embedding vs B (AAGCN direct NB season): val, test, Δtest vs B, wins of 9 paired runs | 15.6956, 38.5233, +0.9802, 4/9; B val 15.5098 test 37.5431 | P3 | kaggle_run/runs.jsonl (derived) | VERIFIED (derived by script) |
| EV-141 | Persistence test RMSE by protocol: P3, P9, PK9, legacy all windows / without week 395 | 36.0157, 28.5410, 31.7619, 44.7953 / 29.5210 | all | results.json; frozen9_plus_audit.json | VERIFIED (derived by script) |
| EV-142 | Interval baseline additive: coverage, mean width (legacy folds, mean of 3 folds) | 0.9601, 98.64 | PL | paper/ieee/results/interval_baseline.json (derived by scripts/interval_baseline.py) | VERIFIED (derived by script) |
| EV-143 | Interval baseline scaled: coverage, mean width (legacy folds, mean of 3 folds) | 0.9629, 74.14 | PL | paper/ieee/results/interval_baseline.json (derived by scripts/interval_baseline.py) | VERIFIED (derived by script) |
| EV-144 | Gaussian head STGAT: PICP, MPIW (mean of 9 runs) | 0.9565, 122.84 | PL | analysis/results/improved_sweep.json | VERIFIED (derived by script) |
| EV-145 | Gaussian head A3TGCN: PICP, MPIW (mean of 9 runs) | 0.9617, 230.47 | PL | analysis/results/improved_sweep.json | VERIFIED (derived by script) |
| EV-146 | Gaussian head ASTGCN: PICP, MPIW (mean of 9 runs) | 0.9471, 128.43 | PL | analysis/results/improved_sweep.json | VERIFIED (derived by script) |
| EV-147 | Gaussian head DCRNN: PICP, MPIW (mean of 9 runs) | 0.9521, 249.92 | PL | analysis/results/improved_sweep.json | VERIFIED (derived by script) |
| EV-148 | Gaussian head AAGCN: PICP, MPIW (mean of 9 runs) | 0.9589, 145.36 | PL | analysis/results/improved_sweep.json | VERIFIED (derived by script) |
| EV-149 | Arms with BH-adjusted p < 0.05 against persistence on validation RMSE: better, worse, of arms compared | 7, 8, of 30 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py | VERIFIED (derived by script) |
| EV-150 | Arms with BH-adjusted p < 0.05 against persistence on test RMSE: better, worse, of arms compared | 0, 8, of 30 | P9 | frozen9_plus_audit.json via seirgnn2/stats.py | VERIFIED (derived by script) |
| EV-151 | Persistence minus gcn+direct test RMSE at h=1, 2, 3 (positive = model better) | +0.3766, +0.1155, +0.5787 | P9 | frozen9_plus_audit.json (arithmetic) | VERIFIED (derived by script) |
| EV-152 | Persistence minus adaptive_gwn+foi_res anchor test RMSE at h=1, 2, 3 (positive = model better) | +0.9659, +0.8508, +1.4114 | P9 | frozen9_plus_audit.json (arithmetic) | VERIFIED (derived by script) |
| EV-153 | Persistence minus adaptive_gwn+foi_res anchor E0enc test RMSE at h=1, 2, 3 (positive = model better) | +1.0263, +0.8628, +1.3763 | P9 | frozen9_plus_audit.json (arithmetic) | VERIFIED (derived by script) |
| EV-154 | Persistence minus gcn+foi_res anchor test RMSE at h=1, 2, 3 (positive = model better) | +0.9867, +0.5538, +1.2198 | P9 | frozen9_plus_audit.json (arithmetic) | VERIFIED (derived by script) |
| EV-155 | Persistence minus adaptive_gwn+residual test RMSE at h=1, 2, 3 (positive = model better) | +0.4002, +0.1587, +1.1026 | P9 | frozen9_plus_audit.json (arithmetic) | VERIFIED (derived by script) |
| EV-156 | Rescue control, AAGCN: SEIR minus gated head, val delta (wins), test delta (wins) | +0.8629 (1/9), +2.0502 (0/9); share of repair from anchor and gate -0.196 | P3 | kaggle_run/rescue.csv | VERIFIED (derived by script) |
| EV-157 | Rescue control, ASTGCN: SEIR minus gated head, val delta (wins), test delta (wins) | +0.7501 (0/9), +0.9537 (3/9); share of repair from anchor and gate -0.099 | P3 | kaggle_run/rescue.csv | VERIFIED (derived by script) |
| EV-158 | Rescue control, LSTM: SEIR minus gated head, val delta (wins), test delta (wins) | +0.7458 (0/9), -0.3257 (6/9); share of repair from anchor and gate 0.043 | P3 | kaggle_run/rescue.csv | VERIFIED (derived by script) |
| EV-159 | Rescue control, STGAT: SEIR minus gated head, val delta (wins), test delta (wins) | +0.6562 (1/9), -0.6194 (8/9); share of repair from anchor and gate 1.124 | P3 | kaggle_run/rescue.csv | VERIFIED (derived by script) |
| EV-160 | Rescue control, A3TGCN: SEIR minus gated head, val delta (wins), test delta (wins) | -0.0029 (6/9), -1.6551 (9/9); share of repair from anchor and gate 1.000 | P3 | kaggle_run/rescue.csv | VERIFIED (derived by script) |
| EV-161 | Rescue control, DCRNN: SEIR minus gated head, val delta (wins), test delta (wins) | -0.6796 (3/9), -3.5018 (9/9); share of repair from anchor and gate 0.960 | P3 | kaggle_run/rescue.csv | VERIFIED (derived by script) |
| EV-162 | Legacy array national cases: week 395, week 394, week 396, largest other week | 7165, 396, 414, 10535 | PL | notebooks/baseline/sri_lanka_2013-2022_shifted.npy channel 5 (summed over districts) | VERIFIED (derived by script) |
| EV-163 | Rebuilt series: weeks, weeks with no report, first and last week start date | 559, 7, 2013-05-15 to 2024-02-24 | rebuilt | data/corrected/rebuilt_cases.npy, rebuilt_index.csv | VERIFIED (derived by script) |
| EV-164 | Three-origin configurations: number, number below persistence on validation, of those above persistence on test; lowest-validation configuration (val, test) | 69, 56, 25; P1 spatial penalty (15.45, 37.52) | P3 | kaggle_run/runs.jsonl, results.json (figure_values.json) | VERIFIED (derived by script) |
| EV-165 | Persistence minus gcn+residual (baseline GNN) test RMSE at h=1, 2, 3 (positive = model better) | +0.2307, +0.1476, +0.7881 | P9 | frozen9_plus_audit.json (arithmetic) | VERIFIED (derived by script) |

<!-- END GENERATED -->
