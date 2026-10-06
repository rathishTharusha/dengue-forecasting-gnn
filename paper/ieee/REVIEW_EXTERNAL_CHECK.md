# Check of the external review (`REVIEW_EXTERNAL.md`)

Written on `paper/ieee-draft` against `main.tex` as committed after the week-395 rewording. Nothing was edited in the paper and nothing was trained for this check. Line numbers are lines of `paper/ieee/main.tex`.

Cheap computations on saved files were run for this check and are in `paper/ieee/results/r2_decomposition.json` (Q9). The 95% intervals below come from the nine origin means of `seirgnn2/results/frozen9_plus_audit.json`.

Cost key: **W** = wording only; **C** = cheap computation on saved outputs; **T** = needs new training or new data.

State of the data work this check depends on: branch `fix/week-index` is pushed (index fix, NDVI, ERA5 and COVID tables rebuilt). The Kaggle rerun has not started. The two local COVID and stringency-prior reruns finished **before** I checked Q1; they use the static graph (see Q1), so they are preliminary and will be discarded and rerun with the corrected graph.

---

# PART 1. Point by point

## Table rows (points 1 to 16)

**1. Main experiments use an incorrect graph.** Reviewer: two false borders, main runs not repeated. Lines 110 (141 directed edges) and 369 (Limitations).
- **Verdict: VALID.** The file on main has two one-directional edges (Kandy to Ampara, Kegalle to Kalutara) that are not shared borders. I re-checked this against the GADM 4.1 polygons (50 m buffer): the main file lists 59 undirected edges, 2 of them not borders, and no border is missing. The corrected file on `praveen` has 57 edges, all 57 real borders, none missing, symmetric (Q1).
- **Status:** not fixed. The corrected file exists only on `praveen` and `origin/adaptive-graph`. The paper states it as a limitation (line 369).
- **Fix:** merge the adjacency fix, add a test that the edge list equals the GADM shared-border list (store the border list in a committed file because `data/raw` is git-ignored), then T: rerun the 30 frozen nine-origin arms (810 runs, about 1.6 h at 6 workers locally), the Kaggle notebook (810 runs, about 3 h; the notebook reads the graph from the source dataset, so the fix goes into `20_data.py` as a documented, asserted removal of the two pairs, without re-uploading the dataset), and the COVID and stringency-prior arms (189 runs, about 2 h locally).

**2. Six arms were chosen after earlier results; the 46-arm grid never ran.** Lines 246 and 368.
- **Verdict: VALID** and already stated in the paper. The question "were test results seen first" is answered in Q2: yes. The frozen nine-origin test weeks had been seen through other protocols and other models for ten days before the six arms were chosen.
- **Status:** disclosed, not repaired.
- **Fix:** W to separate confirmatory from exploratory and state what was seen (Q2). A genuinely untouched test needs T: new weekly reports after 2024 week 10 (Q10, 127 reports available).

**3. The ablation that answers the title question is missing from the nine-origin protocol.** Line 296.
- **Verdict: VALID.** On the nine-origin protocol with the adaptive encoder (`adaptive_gwn`), the file has residual, gated SEIR (anchored, global E0 and learned E0), bare SEIR (anchored, global E0), and mass-action. It has **no `gated` head without SEIR**, and the bare SEIR head exists only with the global E0, not with the learned E0 that defines our best arm (Q3).
- **Status:** not fixed.
- **Fix:** T: two new arms, `adaptive_gwn+gated` and `adaptive_gwn+foi anchor E0enc`, 2 x 27 = 54 runs, about 10 minutes locally. If the corrected graph is merged first, the whole family is rerun anyway.

**4. Too many protocols and datasets mixed.** Lines 96 to 101, 227 to 233, 323 to 334.
- **Verdict: PARTLY VALID.** The paper labels every protocol and never puts two in one table, but it does present five bodies of evidence (nine-origin main, three-origin ablation, legacy-array intervals and older sections, local COVID runs, EXP-050 notebook). The reviewer's fix (one final protocol) is right.
- **Status:** partly addressed by labels and the week-index rerun.
- **Fix:** W to move legacy and older COVID results into one labelled exploratory subsection. C for the interval baseline on the rebuilt data (persistence-residual interval needs no training). The Gaussian-head interval has no rebuilt-data counterpart; T only if wanted (negative-binomial intervals from the NB arms need saved forecasts, which are not kept; see Q4).

**5. Weak statistics for small gains.** Lines 25 to 26, 367.
- **Verdict: VALID.** Computed here: the best arm (Adaptive GCN, gated SEIR, anchored, learned E0) has a mean paired difference from persistence of -1.10 with a 95% t-interval [-2.84, +0.65] and a bootstrap interval [-2.48, +0.32] over the nine origin means. The interval rules out an improvement larger than about 2.8 cases and cannot rule out zero or a small loss. GCN residual (baseline GNN): -0.42 [-1.54, +0.70]. Adaptive residual: -0.60 [-1.81, +0.61].
- **Status:** not fixed in the paper; the intervals are not yet reported.
- **Fix:** C for the intervals and effect sizes, W for the sentence that non-significance is not equivalence. More origins is T: moving from 9 to, for example, 17 origins doubles the runs (about 1,600 for the same 30 arms, about 3 h locally) but the test windows overlap more, so the gain in independent evidence is smaller than the run count suggests.

**6. Persistence is the only simple baseline.** Section V-B (lines 236 to 246).
- **Verdict: VALID.** No seasonal naive, no AR, no count regression on the nine-origin protocol. Some classical models already exist but only on the three-origin protocol and are not in the paper: NB-GLM (test 42.85), k-NN analogues (34.34), gradient-boosted trees without and with climate (38.96, 36.49) in `kaggle_run/runs.jsonl`.
- **Status:** not fixed.
- **Fix:** T-light: seasonal naive needs no training; an AR model per district on log(1+cases) with the lag order chosen on validation, and an NB regression with lagged cases and seasonal terms, are minutes of CPU on 9 origins (statsmodels). Needs the pre-registration first.

**7. Over-interpreting the 89%.** Lines 24, 47, 104, 345, 346 ("11% of the variance left to explain").
- **Verdict: VALID.** The 89% is the squared Pearson correlation between cases(t-1) and cases(t) pooled over all district-weeks, in sample, on counts (`80_analysis.py::r2_pooled`). Recomputed on the fixed data: pooled 0.889; within district (cases demeaned per district) 0.857; per-district median 0.685 (range 0.449 to 0.899); on log(1+cases) 0.802 pooled and 0.639 within district; out of sample (OLS fitted on the first 70% of weeks, evaluated on the last 30%) 0.764. The high pooled value is partly driven by large outbreak weeks and large districts.
- **Status:** not fixed.
- **Fix:** W plus the numbers above (C, done).

**8. "Climate explains only 3%".** Lines 24, 104, 345.
- **Verdict: VALID.** The 3% is the largest of 42 pooled in-sample squared correlations (6 variables x lags 2 to 8). After the week-index fix it is 0.0293 (soil moisture, lag 5; 0.0291 before). Within district (both variables demeaned) the best is 0.0101 (relative humidity, lag 6). Taking a maximum over 42 candidates also biases it upward.
- **Status:** number recomputed; wording not fixed.
- **Fix:** W: "the strongest single climate variable had r2 = 0.029"; let the climate-lag ablations carry the predictive claim (and those are being rerun).

**9. SEIR is simplified; title and conclusion are broad.** Lines 14, 74, 376 to 378.
- **Verdict: PARTLY VALID.** The paper already says human-only SEIR (line 74), but the title says "epidemic physics" and the discussion says "physics" in several places. The fixed values are listed in Q8.
- **Status:** not fixed.
- **Fix:** W to narrow to "this SEIR-informed architecture". A sensitivity analysis is T: reporting fraction, omega and gamma one at a time at 3 literature-based values on the Adaptive SEIR-GNN only: 7 distinct settings (centre plus 2 x 3) x 27 runs = 189 runs, about 0.4 h locally.

**10. The "conditional median" sentence is wrong.** Line 207.
- **Verdict: VALID.** Squared error on log(1+y) estimates E[log(1+Y) given X]. The exponential of that is the conditional median only if the conditional distribution of log(1+Y) is symmetric, which is not assumed. The same wrong statement is in a code comment (`seirgnn2/models.py`, the DISTS comment, and `train.py`); the comment is outside this task.
- **Status:** not fixed.
- **Fix:** W only.

**11. "Reporting backlog" stated without evidence.** Lines 51, 96 to 100.
- **Verdict: VALID for the earlier draft; ALREADY FIXED in wording.** The current text says the week is a formula error in the published table (Vol. 48 No. 2), with the evidence in EV-192. The evidence is stronger than the reviewer assumed: `report_corrections.json` records that the weekly row's district values sum to 7165 while the same row's printed national total is 35, 15 of 24 inner cells equal the sum of their two neighbours, and the year-to-date row (equal to the weekly count in the first week of the year) sums, with the Kalmunai division, to 351, the printed national total (Q7).
- **Remaining gap:** the paper names the report in text but has no `\cite` and no bib entry for it.
- **Fix:** W plus one bib entry (the URL is in `report_corrections.json`; the entry is not on the approved list in PAPER_PLAN.md, so it needs your approval).

**12. RMSE only.** Table II (`tables/main_p9.tex`), Section VI.
- **Verdict: VALID.**
- **Fix:** MAE is already saved for every run (field `MAE` in the result files), so MAE and a run-level skill score (1 - RMSE of arm / RMSE of persistence, per origin) are C. Per-district summaries and a district-level skill score need forecasts, which are **not saved** for the nine-origin arms or for EXP-050 (Q4); this needs T (rerun with forecasts kept, part of the 810-run rerun).

**13. The plus/minus values are easy to misread.** `tables/main_p9.tex`, `tables/ablation_p3.tex`, captions.
- **Verdict: VALID.** The standard deviation is over 27 runs and is dominated by origins.
- **Fix:** C: report the paired difference from persistence with its 95% interval (computed above for four arms), and seed standard deviation within origin as a separate number.

**14. 69 configurations, reader cannot reconstruct them.** Line 304 and Fig. 3.
- **Verdict: VALID.** Table III shows 14 rows and Table IV 11; the other 44 are only in `full_paper/outputs/kaggle_run/runs.jsonl` and `levers.csv`.
- **Fix:** C: generate a full results table for a supplement from `runs.jsonl` (cheap). The numbers will change after the rerun, so generate it last.

**15. Covariate provenance is thin.** Section III.
- **Verdict: VALID.** `docs/DATA_PROVENANCE.md` has most of it; the paper has none. Q6 gives the table content and flags two lags that may be shorter than the real publication delay (population, NDVI).
- **Fix:** W plus a compact table.

**16. STGAT and DCRNN, 62 and 73.** Lines 262 and 282, Table III.
- **Verdict: PARTLY VALID.** The numbers are right and come from the EXP-050 three-origin run (EV-123, EV-125), not from the deleted S5 leaderboard (Q5). The implementations are transcriptions verified against the reference code by `tests/test_kaggle_architectures.py`. But the logs show a sign of a collapsed fit: for DCRNN, STGAT and A3T-GCN with a direct head the best epoch is 1 to 9 in most runs and the forecast is almost flat across horizons (RMSE at h=3 minus h=1 of 0.65 for DCRNN and 3.4 to 3.5 for STGAT and A3T-GCN, against 11 to 12 for the working encoders). All models share one fixed hyperparameter set and none was tuned. Line 262 still writes "fails", which breaks the rule we set.
- **Fix:** W for the wording; T for fairness tuning (Part 3, item M6); the parameter counts are in Q5.

## Abstract and title (points 17 to 19)

**17. Abstract claims "no measurable test gain" for four additions from exploratory runs.** Line 29.
- **Verdict: VALID.** Learned adjacency (three-origin and nine-origin residual), augmentation, climate lags (three-origin) and COVID covariates (older code) come from different protocols; none is a nine-origin confirmatory test.
- **Fix:** W: "exploratory ablations found no clear gain", in a separate subsection. The climate and COVID parts are being rerun.

**18. Interval result in the abstract uses the legacy array.** Line 30.
- **Verdict: VALID.** It is flagged "legacy-array folds" in parentheses but sits in the abstract.
- **Fix:** W: remove from the abstract; optionally C for a rebuilt-data version of the simple interval.

**19. Title.** Line 14.
- **Verdict: PARTLY VALID.** "Epidemic Physics" is broader than a human-only SEIR head on a graph encoder. The reviewer's suggestion matches the title direction already planned.
- **Fix:** W.

## Reviewer's "four things to fix first"
(1) corrected graph and rerun: point 1; (2) missing nine-origin ablation: point 3; (3) separate exploratory from confirmatory and protect an untouched test: points 2, 4, 17; (4) seasonal naive and an AR or count baseline: point 6.

---

# PART 2. Fact questions

## Q1 Graph
Which arms use the static adjacency (`notebooks/baseline/sri_lanka_adj_list.json`, passed as `fixed` and `edge_index`):

| Component | Static graph used? |
|---|---|
| `gcn`, `gat` (alias of gcn) toy encoders | yes, as `A` in every layer |
| `none`, `uniform`, `linear` | no |
| `adaptive` (including `adaptive_gwn`) message passing | no, learned matrix only |
| `hybrid` | yes, 0.5 static plus 0.5 learned (used only in early screens, not in the 30 nine-origin arms) |
| STGAT, A3T-GCN, ASTGCN, AAGCN, DCRNN | yes, through `edge_index` |
| LSTM | no |
| Every SEIR head with a free or anchored rate (`foi`, `foi_res`) | **yes, through the spatial import term** `nb = I @ fixed.T` in `Net._physics`, with a learned scalar weight initialised at 0 (`models.py`) |
| SEIR head with `lam_param="mass"` | no import term |
| `foi_meta` (metapopulation) | yes, as the coupling prior |
| Spatial penalty arms (P1, P2, P5) | yes, binary version |

So the Adaptive SEIR-GNN (adaptive encoder, gated SEIR, anchored) mixes the learned graph in the encoder with the static graph in the SEIR import term. Only `adaptive_gwn+residual` and `adaptive_gwn+foi_res mass` avoid the static graph entirely among the post-fix arms. The COVID and stringency-prior arms use SEIR-STGAT, which uses the static graph twice (GATConv edge list and the import term), so they depend on it. They had already finished when I checked, so I did not need to stop them; their outputs (in the worktree `../dengue-wf`, untracked) are preliminary and will be rerun after the graph is corrected.

Corrected file: `notebooks/baseline/sri_lanka_adj_list.json` on `praveen` and `origin/adaptive-graph` (commit 6273b45, with a symmetry check added to `build_fixed_adjacency`). Verified here against `gadm41_LKA_1.json` (50 m buffer): 57 undirected edges, symmetric, every edge a real shared border, and no real border missing. The file on main has 59 edges, of which exactly the two named are not borders.

## Q2 Selection
What had been seen before the six EXP-061 arms were chosen:

| What | Date | Overlap with the nine-origin test weeks |
|---|---|---|
| EXP-032 to EXP-049 (three-origin test blocks, 15% each, from 0.55 to 1.0) | 2026-09-23 to 09-24 | covers 0.55 to 0.95 of the nine-origin test range |
| EXP-038, EXP-047, EXP-050 (nine origins 0.40 + k/15, blocks of 1/15) | 2026-09-23, 09-24, 09-25 | covers the whole range |
| EXP-051 to EXP-055 (SEIR-STGAT on **exactly** the nine origins 0.50 to 0.90, test RMSE logged) | 2026-09-23 to 09-24 | identical |
| EXP-056, EXP-057 (same nine origins, COVID and stringency arms) | 2026-10-03 | identical |
| EXP-059 (24-arm grid on the nine origins, test RMSE logged) | 2026-10-04 | identical |

EXP-059 and EXP-061 are in the same commit (91d0778, 2026-10-04 11:04 +0530), so the log does not give the order within that day. EXP-061 itself says the six arms were chosen by engineering judgement and were not blind to likely RMSE. Answer: yes, nine-origin test results (other models) had been seen from 2026-09-23, and overlapping test weeks from other protocols from the same day. The nine-origin protocol itself (`docs/PROTOCOL.md`, fingerprint e9afbdfb0528) predates the six arms, but its test numbers were not unseen.

## Q3 Ablation on the nine-origin protocol, Adaptive SEIR-GNN encoder, seeds 0 to 2
In `frozen9_plus_audit.json`:
- gated head without SEIR (`adaptive_gwn+gated`): **missing**
- gated SEIR: present (`foi_res anchor`, global E0; and `foi_res anchor E0enc`, learned E0)
- bare SEIR: present only as `foi anchor` (global E0). **Missing:** bare SEIR with learned E0.
- also present: residual, mass-action gated.

## Q4 Saved forecasts
- Nine-origin arms (`seirgnn2/results/*.json`): aggregate only, per run: RMSE, MAE, per-horizon RMSE, validation RMSE. The `keep=True` option writes `results/<grid>_preds.pkl`, which is git-ignored and absent on disk.
- EXP-050 (`kaggle_run/runs.jsonl`): aggregate only. The notebook keeps forecasts only for 6 configurations (the ceiling analysis), and that pickle is not in the repository outputs.
- Forecasts exist as `.npz` for the SEIR-STGAT runs of EXP-051 to EXP-057 (`analysis/results/...`).
- Computable without retraining: MAE (already saved), a run-level skill score against persistence, the paired differences and intervals. Persistence per district per week can be recomputed from the data. **Not computable:** per-district or per-week model summaries, forecast-versus-actual figures, and any recomputation of the paired tests from forecasts (the planned check "from saved per-district forecasts" needs a rerun with forecasts kept).

## Q5 STGAT and DCRNN
- Source of 62.00 and 73.52: `full_paper/outputs/kaggle_run/runs.jsonl`, names `STGAT+direct` and `DCRNN+direct`, three-origin protocol, 9 runs each (EV-123, EV-125), validation 22.93 and 34.64. The nine-origin run of the same heads gave 44.71 for STGAT direct. They are not from the deleted S5 leaderboard (that had 83.99 for STGAT on a different build and protocol).
- Implementation: `analysis/lib/reproduced.py` (reference wiring from the Weng et al. code, with a dropout-at-inference bug fixed and batched edges) and its plain-PyTorch transcription in `full_paper/kaggle/src/40_architectures.py`, checked weight for weight in `tests/test_kaggle_architectures.py`.
- Encoder parameter counts (hidden 64 as used; head excluded): STGAT 160,699; A3T-GCN 8,547; ASTGCN 50,574; AAGCN 2,678; DCRNN 827,648; LSTM 50,432. Whole network with a direct head for toy encoders: GCN 10,371, adaptive 10,271; STGAT 160,894.
- Hyperparameters, the same for every model: hidden 64, dropout 0.1, Adam lr 3e-3, weight decay 1e-4, batch 32, 300 epochs (400 with NB), patience 40, gradient clip 5. **Tuning budget: none for any model**, ours included. (Phase-2 sweeps of hidden size and learning rate were run on the old toy GCN on the legacy array, not on these encoders.)
- Signs of a broken run (best epoch, flat horizon profile) are in point 16: DCRNN best epoch 1 to 33 (mostly 1 to 9), STGAT best epoch 1 to 81 (four runs at 1 or 2), A3T-GCN mostly 3 to 4. Loss curves were not saved, so divergence cannot be confirmed; the flat horizon profile (almost the same RMSE at h=1, 2, 3) suggests a near-constant forecast. This is consistent with the repository's own diagnosis (EXP-039: A3T-GCN restarts its recurrent state each week; STGAT's LSTM sees the whole country as one vector, so a district is identified only by position).

## Q6 Covariates

| Variable | Source | Resolution | Date range | Lag used | Missing values | Real publication delay | Lag shorter than delay? |
|---|---|---|---|---|---|---|---|
| Cases | Epidemiology Unit weekly reports, Table 1 | district, weekly | 2013 W26 to 2024 W10 (559 reports, 7 absent) | 1 week | 7 weeks left missing, windows touching them dropped | next report | no |
| ERA5 (6 variables) | Open-Meteo archive API (ERA5) | one interior point per district, daily, aggregated to 7 days from `week_start` | 2013-01-01 to 2024-03-31 | 2 weeks | none | ERA5 is released about 5 days after the day (per `corrected_data.py`); the archive API serves the final reanalysis, so the initial version a forecaster would have seen may differ (not checked) | no |
| NDVI | MODIS MOD13Q1 v061 via ORNL subset | 250 m, 16-day composite, mean over a plus or minus 5 km box | composites from 2012-09 | as-of: composite start + 32 days before the week start | none; as-of join, no interpolation | **assumed** 16 days after the composite ends, not measured | cannot say; assumption |
| Stringency | OxCGRT, national `NAT_TOTAL` | national, daily, weekly mean | 2019-12-28 to 2022-12-31 (158 weeks) | 2 weeks | pre-2020 and gaps: neutral gate | days | no |
| Workplace mobility | Google COVID-19 Community Mobility | **national only**, daily, weekly mean | 2020-02-15 to 2022-10-15 (140 weeks) | 2 weeks | neutral gate | a few days | no; archived vintages not verified |
| Population | DCS 2012 Census (2013 to 2014); DCS mid-year estimates for year Y-1 from 2015 | district, annual | 2013 to 2024 | previous year's figure | none | **unknown**: a mid-year estimate for Y-1 is normally published during year Y, so it may not exist in week 1 of Y | possibly yes; not verified in the repository |
| Seasonal features | day of year of `week_start` | weekly | all | 0 (calendar) | none | none | no |

Flags: population (publication date of the Y-1 mid-year estimate is not documented) and NDVI (delay assumed). Both are unaffected for the nine-origin arms except the SEIR heads' population scale.

## Q7 Week 395
The table is the district table in the Epidemiology Unit's Weekly Epidemiological Report, Vol. 48 No. 2, reporting 26 December 2020 to 1 January 2021 ("1st week"), https://www.epid.gov.lk/storage/post/pdfs/vol_48_no_02-english_1.pdf (archived copy used in `fetch_sources.py`). Evidence (`data/external/report_corrections.json`): the weekly row has 15 of its 24 inner cells equal to the sum of the two cells before them (a spreadsheet formula error); its district values sum to 7165 while the same row's printed national total is 35; the year-to-date row, which equals the weekly count in the first week of the year, sums with the Kalmunai division to 351 and equals its national total of 351. The legacy array holds the first row (national sum 7165). Written to the corrections file by `extract_week1_correction`.

## Q8 SEIR settings

| Parameter | Value | Source | Cited in the paper? |
|---|---|---|---|
| Reporting fraction rho | 1/11, times a learned global factor exp(s) when E0 is fitted | Liu et al. scale reported cases by 11 | Liu is cited; the 11 is not stated as theirs |
| omega (E to I) | 0.7/7 per day (0.1/day) | Liu et al. (WHO guidance); intrinsic incubation alone is 5.9 days (Chan and Johansson 2012, `seir_parameters.json`) | Liu only; Chan and Johansson not in the bib |
| gamma (I to R) | 1/7 per day | Liu et al.; WHO range 2 to 7 days; Phaijoo and Gurung 3.04 days | Liu only |
| Initial susceptible fraction s0 | 1 - 0.682 = 0.318, the same for every district, minus cumulative reported cases scaled by rho N | suburban Colombo seroprevalence, all ages, 2013 to 2014 (`seir_parameters.json`, key `colombo_sero2015`) | **no** (and not on the approved list) |
| E0, I0 | from the last two reported weeks | design choice | described |
| Spatial import weight | 0.1 times a learned scalar | design choice | no |
| Free-rate bias | -9.36 (median inverted lambda) | measured on training windows (`diagnose_foi.py`) | no |
| Clamps | lambda at most 1/7; anchored raw within plus or minus 10; floor 0.5 cases | design choices | partly |

Note: the project's own source file says the ascertainment evidence spans about an order of magnitude and should be a sensitivity parameter, yet the code fixes it at 1/11 (with the learned scale). Liu et al. (2025) is in the literature manifest and fetch log (status "present", 1,184,756 bytes, sha256 prefix 38365da496bc50fe) but the PDF itself is not on disk in this clone (`literature/papers/` is git-ignored); it is not in `full_paper/references`.

## Q9 The 89% and the 3%
- 89%: squared Pearson correlation of cases(t-1) with cases(t), pooled over all district-weeks, in sample, on raw counts (value 0.8893). Fixed-data recomputation (`paper/ieee/results/r2_decomposition.json`): pooled 0.889; **within district 0.857**; per-district median 0.685 (0.449 to 0.899); log scale 0.802 pooled and 0.639 within district; out of sample 0.764.
- 3%: maximum over 6 climate variables and lags 2 to 8 of the pooled in-sample squared correlation with cases: 0.0293 (soil moisture, lag 5) on the fixed data, 0.0291 on the old index. Within district (both demeaned): 0.0101 (relative humidity, lag 6).

## Q10 New data
- Cases: yes. The Epidemiology Unit page lists weekly reports to **Vol. 53 No. 33** (printed 3 to 9 August 2026, week 32), the same Table 1 layout (checked on that PDF). After the last week in our series (Vol. 51 No. 10, 2024 W10) there are 42 reports in Vol. 51 (nos. 11 to 52), 52 in Vol. 52 and 33 in Vol. 53: **127 weeks**. The site refuses a plain request (HTTP 403) but serves the listing with a browser User-Agent. File names carry a hash, so the listing must be scraped. The repository has a Table 1 parser that was verified against 553 reports; it must be rerun with a check that district values plus Kalmunai equal the printed national total (the same integrity check as the current build).
- ERA5: yes (Open-Meteo returns data to at least 2026-09). NDVI: yes (ORNL composites listed to 2026-08-13; needs the 32-day availability rule). Population for 2025 and 2026: the DCS mid-year file would have to be checked; not verified.
- Design for an untouched test: freeze the arm list, the settings, the corrected graph and the baselines in a committed pre-registration before downloading or opening any data after 2024 W10. Train on all data to 2024 W10 (validation = the last 30 windows), then evaluate the frozen arms on forecast origins in the new period, for example three origins at the start of 2025, mid 2025 and the start of 2026, each with a test block of about 30 weeks and retraining at each origin, seeds 0 to 2, with persistence, seasonal naive and the classical baselines scored on the same weeks. Work estimate: 2 days of parsing, checking and rebuilding the index and covariates (686 weeks; the weekly grid is `2013-06-15 + 7 x row`), 1 to 2 h ERA5 download, and about 0.5 h of training for 8 arms x 3 origins x 3 seeds (about 72 runs).

---

# PART 3. Run plan (nothing started)

Estimates are for 6 workers on this machine unless stated; Kaggle numbers come from EXP-050 (810 runs in 3.01 h).

## Must do
| # | Item | Runs | Time |
|---|---|---|---|
| M1 | Merge the adjacency fix into `fix/week-index`; test that the edge list equals the GADM shared-border list (store the list in a committed file); put the corrected graph into the Kaggle notebook source (`20_data.py`, asserted removal of the two pairs) and rebuild the notebook | 0 | 2 to 3 h of work |
| M2 | Pre-register in `docs/EXPERIMENT_LOG.md` (data, arms, protocol, comparison families F1 to F3, the replace-whatever-it-shows rule), commit before any run | 0 | 1 h |
| M3 | Rerun the 30 frozen nine-origin arms on the corrected graph, forecasts saved per run | 810 | about 1.6 h |
| M4 | F2 ablation: add `adaptive_gwn+gated` and `adaptive_gwn+foi anchor E0enc` | 54 | about 0.2 h |
| M5 | Classical baselines on the same nine origins and horizons, fitted on training weeks only: seasonal naive, AR on log(1+cases) per district with lag order chosen on validation, NB regression with lagged cases and seasonal terms | 9 origins x 3 models (deterministic) | under 0.5 h |
| M6 | STGAT and DCRNN fairness: log the cause of the flat, early-stopped fits; give STGAT (direct and gated), DCRNN and our model the same validation-only tuning grid, fixed in the pre-registration (for example 6 settings of learning rate and hidden size on 3 origins x 3 seeds, 54 runs per arm per setting group, about 0.7 h for the four arms) | about 650 | about 1.5 h |
| M7 | Kaggle rerun with both fixes (week index and corrected graph), all 810 runs, one run | 810 | about 3 h on Kaggle (you start it) |
| M8 | COVID and stringency-prior reruns on the corrected graph (the finished runs on the old graph are discarded) | 189 | about 2 h |
| M9 | Statistics from saved forecasts: RMSE, MAE, skill against persistence, paired difference with 95% interval over origins, seed spread separately; recompute the 89% and the climate figure | 0 | 1 h |

## Should do
| # | Item | Runs | Time |
|---|---|---|---|
| S1 | SEIR sensitivity on the Adaptive SEIR-GNN only: reporting fraction, omega, gamma one at a time at 3 literature-based values (7 distinct settings including the centre) | 189 | about 0.4 h |
| S2 | Interval comparison on the rebuilt data: persistence-residual intervals (no training); Gaussian or NB head intervals only if forecasts are saved | 0 to 135 | under 0.5 h to 1 h |
| S3 | Supplementary table of all 69 three-origin configurations from the rerun output | 0 | 0.5 h |

## Optional
| # | Item | Runs | Time |
|---|---|---|---|
| O1 | Untouched test on the 127 new weeks (design in Q10) | about 72 plus baselines | 2 days of data work plus 0.5 h of training |
| O2 | More forecast origins (for example 17) to raise power | about 1,600 | about 3 h |

Dependencies: M1 and M2 first; M3 to M6 after M2; M7 after M1; M8 after M1; M9 after M3 to M6. Totals: about 2,500 local runs (about 6 to 7 h) plus one 3 h Kaggle run for the must-do list.

I have not started any of this. The corrected graph is not merged and the Kaggle notebook has not been changed for it.
