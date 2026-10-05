# Check of the ACM short paper against EVIDENCE.md

Files read: `paper/sections/00_abstract.tex` to `06_discussion_conclusion.tex`, `05b_mechanism_corrected.tex`. The ACM paper was **not edited**. It mostly rests on the legacy 459-week array (protocol PL in EVIDENCE.md); only section 05b uses the rebuilt 559-week data.

Sources checked: `analysis/results/{renewal_feasibility,mechanistic_r,paper_measurements,physics_final,physics_claims_check,outbreak_signal,improved_sweep}.json/.md`, `crosscheck/FINDINGS.md`, `docs/EXPERIMENT_LOG.md`, `analysis/README.md`.

## Numbers that match a saved file (no action)

| ACM claim | Source |
|---|---|
| Persistence 44.80 with the backlog, 29.52 without | EV-006, `improved_sweep.md` |
| AAGCN 40.99 against 44.80 (3.8 lead); ASTGCN 43.44 (1.4); artifact-free AAGCN 29.84, STGAT 29.35, A3TGCN 29.05 | `improved_sweep.md` |
| Increment p-values 0.078, 0.077, 0.462, 0.304 (deltas +0.526, +0.245, +0.128, +0.280) | `improved_sweep.md` |
| Coverage 0.947 to 0.962 | EV-088 |
| Oracle replay RMSE 13.44 vs persistence 55.03 on the same windows; mean generation interval 3.30 weeks | `renewal_feasibility.json` |
| log R_t predictable r² 0.263 from own past; climate -0.025 linear, -0.028 hump; depletion adds 0.263 to 0.263 | `mechanistic_r.json`; EV-062 |
| Anchors 35.45 (force), 33.05 (damped ratio), 50.01 (R-hat x force) vs 29.52 | `paper_measurements.json` |
| log R-hat correlations +0.798 past growth, -0.217 future growth, -0.106 with absolute future growth; sd(log R) 0.749 | `paper_measurements.json` |
| 1.90x per step = exp(0.749 x sqrt(1 - 0.263)) | arithmetic, 0.643 |
| Spatial term on STGAT -0.044, 57/60, p < 0.0001; envelope -0.186 (5/9, p 0.238) at n = 9 becoming -0.042 (38/75, p 0.69); STGAT 29.550 to 29.506 against 29.52 (p 0.70) | `physics_final.json`, `physics_claims_check.json` |
| Outbreak ranking AUC 0.807 (level) to 0.826 (+R-hat), growth +0.004, neighbours +0.006, base rate 14.4% | `paper_measurements.json` detection |
| Section 05b: SEIR-STGAT 50.13 to 29.86 (9/9, p 0.0039); with 30.04 vs without 49.45 (8/9, p 0.0078); +1.3 RMSE vs persistence (p 0.92); 28.74 vs 28.54; horizon 3/6/12 deltas; stringency prior 30.59/31.28/32.43 vs 30.07; COVID covariates 30.05/30.06/30.07, Holm p 1.0, noise floor 0.109 | EV-068, EV-069, EV-071 to EV-073 |

## Problems

| # | Where | Problem | Severity |
|---|---|---|---|
| A1 | Abstract and Intro, "naive persistence beats every model in the column the source paper reports" | `crosscheck/FINDINGS.md` F1.0 says this holds for the Cross Validated column only. In the Full Dataset column A3TGCN (30.55) and ASTGCN (32.41) beat persistence (33.48) on RMSE, and the crosscheck explicitly warns against the unqualified sentence. Section 04 says "the column labelled cross validated", which is correct; the abstract and the contributions list drop the qualifier. | Medium |
| A2 | Abstract, "reproduce five published ST-GNNs ... to within 7.3%" | 7.3% is the worst deviation over both columns (`analysis/README.md`). `crosscheck/FINDINGS.md` gives ~5% for STGAT. Fine, but say "worst of twenty comparisons". | Low |
| A3 | Results, Gaussian head called "the operationally useful output of the whole study", coverage 0.947 to 0.962 | The mean interval widths are 122.8 to 249.9 cases (`improved_sweep.md`) against point RMSE of about 41 to 47 on the same data. Coverage near 95% with intervals this wide shows calibration, not usefulness. No comparison with a persistence-residual interval was run. The paper's own result (the Gaussian head gives +0.280 RMSE, p 0.304) also does not support calling it a win. Per D7 we do not claim this. | High |
| A4 | Limitations, "We do not evaluate GAN augmentation" and Intro "(b) is left to the full study" | Stale. GAN augmentation exists in the repo: TimeGAN and an SEIR simulator on rebuilt data (EXP-042, EV-065), WGAN-GP on the legacy array (EV-086). Neither helps. | Medium |
| A5 | Intro, skewness 8.6, "top 1% of district-weeks hold 18.3% of cases" | Not found in `analysis/results/eda_findings.csv`, the handbook or EXPERIMENT_LOG by search. UNSUPPORTED until a source file is named. Not used in the IEEE draft. | Medium |
| A6 | Intro, "2017 outbreak alone recorded over 186,000 cases and 440 deaths" | 186,000 is in `docs/handbook/01_problem_and_history.md`. The 440 deaths figure was not found in the repo. Cited to `tissera2020severe`, whose DOI is NEEDS CHECK. Not used in the IEEE draft. | Medium |
| A7 | Intro, "Cases at t-1 explain r² = 0.85" | This is the legacy-array value. On the rebuilt data it is 0.8893 (EV-004). Both are right on their own data, but the ACM paper never says which array. Section 05b switches to rebuilt data without changing the 0.85 claim. | Low |
| A8 | Results, "independently on each origin (p = 0.040, < 0.0001, 0.025)" and "alpha converges to exactly zero in all 108 runs" | Not found in a saved file during this check (the 108-row CSV exists: `stringency_prior_runs.csv`, EXP-057 and EXP-056 notes). Marked UNVERIFIED, not used in the IEEE draft. | Low |
| A9 | Results, the spatial term "holds" | The 57/60 result is for one architecture (STGAT) on the legacy array. On the rebuilt data's B model the effect is -0.06 with 6/9 better (EXPERIMENT_LOG, EXP-046 to EXP-047 lines), and EXP-050 does not adopt the spatial penalty (EV-064: 37.52 vs B 37.54). The ACM text does not state that the result is single-architecture, legacy-array. | Medium |
| A10 | Whole paper | Legacy array contains the week-395 backlog and `0`-filled weather channels. All headline architecture comparisons are on it. The paper says so for the backlog but the adjacency also carries two non-real edges on this array (EV-010). | Low |
| A11 | Section 05b, "better on 9 of 9 origins, p = 0.0039" | Compares the corrected arm with the old defective S5 arm (EXP-051). It is a fix-versus-bug comparison, not evidence for the physics. Same sign, still not better than persistence; the text says so. | Low |

## Overall
No number in the ACM paper contradicts a saved result file. Four claims go further than the evidence (A1, A3, A4, A9) and three could not be traced (A5, A6, A8). The IEEE draft avoids all seven.

## Added after the week-index and week-395 findings

| # | Where | Problem | Severity |
|---|---|---|---|
| A12 | Abstract, Introduction (contribution 2), Framework (two places), Results (figure caption and text): "reporting backlog", "administrative reporting backlog", "19x reporting backlog", "backlogged reporting" | `data/external/report_corrections.json` shows that week 395 of the legacy array is a spreadsheet formula error in the published table (Vol. 48 No. 2), not a backlog: 15 of 24 inner cells of the weekly row equal the sum of two neighbours, and the year-to-date row sums to the published national total of 351 (the array's value is 7165). The ACM paper's wording and its explanation of the mechanism are therefore wrong. The numbers (persistence 44.80 and 29.52) are unaffected. | High (wording) |
| A13 | Section "The mechanism, specified correctly" (COVID covariates, stringency prior) | Both experiments join policy and mobility to cases through `rebuilt_index.csv`, whose `week_start` was wrong on 8 of 559 rows. Rows 346 and 388 change stringency (up to 28.3 points) and rows 378 and 388 change mobility. The reported values (30.05, 30.06, 30.07; 30.59, 31.28, 32.43) come from the mis-joined data and are being re-run. The climate and AUC statements in the ACM paper use the legacy array's own climate channels and are not affected by this index. | High until the re-run is done |

Correction to A6: the 440 deaths in the 2017 outbreak is recorded in `data/external/seir_parameters.json` (`annual_reported_cases`, sources epi_review2021 and tissera2020), so that figure is traceable. The 186,000 cases figure is in the same file as 186101. The skewness and top-1% figures (A5) are still untraced.
