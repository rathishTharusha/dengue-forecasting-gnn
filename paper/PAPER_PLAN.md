> **Codex status, 2026-10-05:** Earlier narrative and decisions below are historical. The current IEEE draft reports development evidence only. Corrected-graph/calendar reruns and untouched-period confirmation remain pending; target windows overlap partition boundaries (EV-218). Read `CODEX_HANDOVER_AUDIT.md`, `FIGURE_AUDIT.md` and `FINAL_AUDIT_CODEX.md` for the latest status. No replacement result is invented.

# Paper plan (IEEE conference, IEEEtran)

Written for: the project owner (Praveen De Silva) to approve before any paper text is written.
Placeholders still open: page limit, venue, authors. This plan assumes **6 pages plus references** until told otherwise (see Q1). Evidence IDs (EV-xxx) refer to `paper/EVIDENCE.md`.

## 1. The core story

Forecasting weekly dengue cases for 25 Sri Lankan districts three weeks ahead looks like a job for a spatio-temporal graph neural network with an epidemic model inside it. We test that idea under a leakage-controlled protocol and find that a naive baseline, repeating last week's count, is very hard to beat: last week's cases explain 89% of this week's variance, and the best climate variable explains 3% (EV-004). On the corrected data the best learned models beat this baseline by about 1.5 RMSE on validation and by 0.6 to 1.1 RMSE on test, and the test gain is not statistically significant (EV-041 to EV-044). An SEIR (epidemic-compartment) output head fed with a persistence anchor repairs two failing encoders, but the repair comes from the anchor and not from the epidemic physics (EV-036, EV-037). A bare SEIR head loses by 12 RMSE until we add the population scale it was missing, after which it sits within 1.3 RMSE of the baseline and still does not beat it (EV-046, EV-047). A learned adjacency, GAN-generated training data, longer climate lags, lockdown covariates and a spatial penalty gave no measurable gain (EV-027, EV-063, EV-065, EV-068, EV-086). The paper's value is a defensible negative result: what to check before claiming that physics or graphs help in dengue forecasting.

## 2. Title options

1. Does Physics Help a Graph Network Forecast Dengue? A Leakage-Controlled Test Against Persistence in Sri Lanka
2. Beating Last Week's Count: SEIR-Informed Graph Networks for District-Level Dengue Forecasting
3. Where Epidemic Physics and Learned Graphs Help, and Where They Do Not: Weekly Dengue Forecasts for 25 Sri Lankan Districts

Recommendation: option 1. Option 2 over-promises; the evidence does not show we beat the baseline.

## 3. Contributions (for the Introduction list)

1. A leakage-controlled rolling-origin protocol for district-level dengue forecasting, with a persistence baseline reported at every horizon and enforced causal lags (cases 1 week, climate 2 weeks). Evidence: EV-003, EV-007, EV-008, EV-020, EV-021, EV-004.
2. A measurement of how reporting backlogs inflate published-style results: the persistence floor falls from 44.80 to 29.52 once one backlog week is removed, and an earlier benchmark paper reports no naive baseline. Evidence: EV-006, EV-089, EV-088.
3. A controlled comparison of output heads and encoders showing that the SEIR head's gain over broken encoders is the persistence anchor, not the physics, and that a population-scaled force-of-infection parameterization brings the bare head from +12 RMSE to near the baseline. Evidence: EV-035 to EV-037, EV-041 to EV-047, EV-060, EV-061.
4. Negative results for four commonly proposed additions (learned adjacency, GAN or simulator augmentation, climate lags and COVID covariates, spatial and mechanistic priors), each tested with matched origins and seeds. Evidence: EV-027, EV-050, EV-063 to EV-069, EV-080 to EV-087.

## 4. Section outline and word budget (about 4,600 words, 6 pages with 3 tables and 2 figures)

| Section | Key points | Evidence IDs | Words |
|---|---|---|---|
| Abstract | Problem, protocol, persistence floor numbers, SEIR head result, what did not help. No citations or equations. | EV-004, EV-021, EV-041, EV-042 | 200 |
| Index Terms | dengue, graph neural networks, physics-informed learning, rolling-origin evaluation, SEIR model, time series forecasting (alphabetical) | | 10 |
| I Introduction | Dengue burden in Sri Lanka; GNN and physics-informed claims; persistence is rarely reported; our contributions list (the only list). | EV-004, EV-089 | 550 |
| II Related Work | Dengue GNNs (Weng, DengueGNN), epidemic GNNs (EINN, CausalGNN, MepoGNN), SEIR-LSTM (Liu), PINN failure modes, GAN augmentation. | EV-089 | 400 |
| III Data and Problem Setup | 559 weeks 2013-W26 to 2024-W10, 25 districts, 7 missing weeks kept as missing, covariates and lags, backlog week 395, task: window 3, horizon 3. | EV-001 to EV-007 | 450 |
| IV Methods | Encoders (GCN with border graph, adaptive graph, published encoders), heads (direct, residual, gated, SEIR, gated SEIR), anchored force of infection (one equation), negative-binomial likelihood, augmentation arms. Each equation followed by a plain-words sentence. | EV-009 to EV-013, EV-011, formula in EVIDENCE Part A | 800 |
| V Experimental Setup | Rolling-origin cross-validation in plain words; protocols P3 and P9; seeds; selection on validation only; paired sign-flip test over origins; Benjamini-Hochberg correction; metrics. | EV-008, EV-009, EV-020, EV-021 | 450 |
| VI Results | (a) Baseline floor (b) head and encoder comparison on P3 (c) nine-origin significance on P9 (d) validation versus test (e) what did not help. | EV-020 to EV-050, EV-063 to EV-073 | 1,000 |
| VII Discussion and Limitations | Why the floor is hard (lag-1 r² 0.89, climate 0.03); anchor not physics; unreachable lambda = 0 cells; three origins cannot reach p < 0.25; validation-test disagreement; hand-picked arms; adjacency edges; legacy-array results limited. | EV-004, EV-039, EV-060, EV-061, EV-010 | 550 |
| VIII Conclusion | Two to three sentences with the numbers; what to do next. | EV-041, EV-034 | 150 |
| Acknowledgment | Funding or course; AI-tools statement (Claude Code used for repository audit, drafting support and code review; authors verified all numbers against saved outputs). | | 80 |
| References | About 25 entries. | | n/a |

## 5. Tables and figures

All numbers will be generated by a new script `paper/_build/make_tables.py` that reads only the saved result files named below, so every table cell traces to an EV row. Figures are redrawn as vector PDFs by `paper/_build/make_figures.py` (to be written after approval). Existing PNGs in `full_paper/outputs/kaggle_run/` show the layouts but are not reused (raster, wrong style).

| Item | Shows | Data file | Script |
|---|---|---|---|
| Table I | Datasets and protocols (P3, P9, PK9): origins, test fraction, persistence RMSE, counts | `seirgnn2/results/frozen9_plus_audit.json`, `full_paper/outputs/kaggle_run/results.json`, `docs/PROTOCOL.md` | make_tables.py |
| Table II | P3 ablation: persistence, none/gcn/adaptive graph, six encoders x direct/residual/gated/SEIR/gated SEIR, B (NB + season). RMSE at h1/h2/h3, mean ± sd, 3 origins x 3 seeds | `full_paper/outputs/kaggle_run/runs.jsonl` | make_tables.py |
| Table III | P9 nine-origin audit: val and test RMSE, Δ vs persistence, wins out of 9, BH p; arms from EV-041 to EV-049 | `seirgnn2/results/frozen9_plus_audit.json` via the logic of `seirgnn2/stats.py` | make_tables.py |
| Table IV | Things that did not help: GAN/WGAN-GP/simulator, climate lags, COVID covariates, spatial penalty, metapopulation | `runs.jsonl`, `origin/Maleesha-Dev:results/phase4_gan_results.csv`, `results/covid_covariates_summary.csv` | make_tables.py |
| Fig. 1 | National weekly cases 2013-2024 with the week-395 backlog marked in the legacy array against the rebuilt series | `data/corrected/rebuilt_*.npy`, `notebooks/baseline/sri_lanka_2013-2022_shifted.npy` | make_figures.py |
| Fig. 2 | Model architecture: encoder, heads, anchored SEIR decoder (schematic) | none (TikZ) | hand-drawn in the .tex |
| Fig. 3 | Test RMSE minus persistence, per arm, with per-origin spread (forest plot) for P9 | `frozen9_plus_audit.json` | make_figures.py |
| Fig. 4 (optional, page permitting) | Validation RMSE against test RMSE for all configurations: shows validation picks do not transfer | `full_paper/outputs/kaggle_run/validation_vs_test.csv`, `runs.jsonl` | make_figures.py |

## 6. Decisions: what goes in the paper

### MAIN RESULTS (on main, same protocol as baseline, saved outputs)

| Item | Reason |
|---|---|
| Persistence floor at all horizons, P3 and P9 (EV-020, EV-021) | Parameter-free, reproducible to four decimals, saved in JSON. |
| EXP-050 P3 ablation of heads and encoders, 69 configs (EV-027 to EV-039) | One notebook, one environment, saved `runs.jsonl`, baseline and variants share the protocol. |
| NB likelihood and the validation-test mismatch (EV-031 to EV-034, EV-039) | Largest controlled validation gain, and the honest finding that it does not transfer to test. |
| Rescue control: gain is anchor, not physics (EV-036, EV-037) | Core claim of contribution 3, saved in `rescue.csv`. |
| P9 audit table incl. post-fix anchored head and adaptive graph (EV-041 to EV-050) | On main, saved JSON, nine origins so significance is reachable. Caveat: six arms were hand-picked (see Q4). |
| Backlog effect and persistence-vs-published-protocol finding (EV-006, EV-089) | Measured and saved; supports contribution 2. |
| Foi-regression diagnostic (EV-060, EV-061) | Log-only, but cheap and explains why the bare head fails. State as "regression diagnostic", not as a head ablation. Needs a saved output file before use (see Q9). |

### ADDITIONAL RESULTS (only on a branch; your decision)

| Item | Reason / condition |
|---|---|
| Phase-2 adaptive graph and spatial-loss sweep, raw CSVs on `praveen` (EV-080 to EV-083) | Legacy array with backlog artifacts, 3 folds, different protocol (PL). Include only as a one-paragraph negative result with its own protocol label, never in the P3 or P9 tables. Corrected-adjacency rerun is the cleaner version. |
| WGAN-GP augmentation (`origin/Maleesha-Dev`, EV-086, EV-087) | This is the only test of the GAN named in the proposal, and the effect is large (fold 2: 124 vs 44 RMSE) so seed noise cannot explain it. Legacy array, different code path. Include as a negative result with the PL label. Not reproduced on corrected data except TimeGAN (EV-065). |
| Phase-3 SEIR-SEI Stage C (`praveen`, EV-084) | Two origins, 40 epochs, legacy array. Use at most as a sentence. |
| Climate-informed force of infection (ADR 0005) | **Exclude**: smoke test only, status "Proposed, evidence against". The refuting measurement (climate r² about -0.03, EV-062) is the usable part. |

### NEGATIVE RESULTS to keep (they teach something)

Bare SEIR head loses by 12 RMSE (EV-035, EV-047); anchored head and adaptive graph tie the baseline on test (EV-041 to EV-044); TimeGAN, SEIR pretraining, climate lags 2-25, spatial penalty, metapopulation head, more training data (EV-063 to EV-066); COVID policy and mobility, movement-restriction prior (EV-068, EV-069); damped correction and horizon sweep (EV-072, EV-073, only if space); lambda = 0 floor unreachable for 39-41% of cells (EV-061).

### EXCLUDE

| Item | Reason |
|---|---|
| Old S5 leaderboard, 71.80 vs 83.99 (EV-090) | Defective script, deleted source, not citable (`CLAUDE.md`). |
| "Peak outbreak 38.65 vs 42.10" (C7) | 42.10 not found; one fold; baseline persistence 38.08 is lower than 38.65. |
| EXP-027 style claims (one origin against pooled persistence) | Compares mismatched windows (EXP-049). |
| `origin_seed` p-values | Count seeds as independent evidence; invalid alone. |
| Seroprevalence as a result (EV-070) | Validation-only by rule R4. May appear as one sentence in Discussion if wanted. |
| Climate-informed FoI results, smoke tests, EXP-001 to EXP-014 debug runs, pre-fix foi/foi_res arms (as results) | Smoke or superseded. The pre-fix bare foi arms stay only as the "before" row of the fix. |
| 71.80 and 27.45 in one table | Different data builds and protocols. |

## 7. References (only from the repo)

Source lists: `full_paper/overleaf/refs.bib` (36 entries, used below), `paper/refs.bib` (ACM paper; not compared entry by entry), `literature/manifest.csv`, `docs/SEIR_ADAPTIVE_AUDIT_RESULTS.md`. DOI/URL is given when the repo has one; everything else is NEEDS CHECK for DOI before submission. Full author lists are in the .bib; here first author and venue only.

| Key | Citation | DOI / URL in repo | Cited for | Status |
|---|---|---|---|---|
| weng2024graph | Weng et al., Graph Representation Learning for Dengue Forecasting, IEEE Big Data 2024 | none (local PDF only; code github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2 @ 45f1c08) | Benchmark data and baseline; no persistence baseline reported | NEEDS CHECK (DOI) |
| gulmohamed2026denguegnn | GulMohamed et al., DengueGNN, Scientific Reports 16, 2026 | 10.1038/s41598-026-43073-y | Related dengue GNN | OK |
| liu2025seirlstm | Liu et al., explainable covariate compartmental model, PLoS Comput. Biol. 21, 2025 | 10.1371/journal.pcbi.1013540 | SEIR-LSTM design, SEIR constants, 11x reporting scale | OK |
| phaijoo2018sensitivity | Phaijoo and Gurung, Sensitivity analysis of SEIR-SEI model of dengue, GAMS J. 6, 2018 | none | SEIR-SEI model | NEEDS CHECK |
| seirsei2022 | Hasan et al., Vector-host SEIR-SEI dengue model, Int. J. Anal. Appl. 20, 2022 | none | SEIR-SEI | NEEDS CHECK |
| huang2019stgat | Huang, Bi, Li, Mao, Wang (Yingfan Huang, Huikun Bi, Zhaoxin Li, Tianlu Mao, Zhaoqi Wang), STGAT: Modeling Spatial-Temporal Interactions for Human Trajectory Prediction, ICCV 2019, pp. 6271-6280 | doi 10.1109/ICCV.2019.00637 | STGAT encoder (our code transcribes the dengue adaptation in Weng et al.: a graph-attention layer then two LSTM layers) | CONFIRMED against Crossref on 2026-10-05 (Crossref gives pages 6271-6280; the owner's message said 6272-6281, Crossref was used) |
| guo2019astgcn | Guo et al., ASTGCN, AAAI 33, 2019 | ojs.aaai.org/index.php/AAAI/article/download/3881/3759 | Encoder | OK |
| bai2021a3tgcn | Bai et al., A3T-GCN, ISPRS IJGI 10, 2021 | arxiv.org/abs/2006.11583 | Encoder | OK |
| shi2019aagcn | Shi et al., Two-stream adaptive GCN, CVPR 2019 | arxiv.org/abs/1805.07694 | Encoder (AAGCN) | OK |
| li2018dcrnn | Li et al., DCRNN, ICLR 2018 | arxiv.org/abs/1707.01926 | Encoder | OK |
| wu2019graphwavenet | Wu et al., Graph WaveNet, IJCAI 2019 | arxiv.org/abs/1906.00121 (audit sheet) | Adaptive adjacency | NEEDS CHECK (not in bib with URL) |
| kipf2017gcn | Kipf and Welling, GCN, ICLR 2017 | none | Graph convolution | NEEDS CHECK |
| velickovic2018gat | Velickovic et al., GAT, ICLR 2018 | none | Only if we mention that our "gat" arm is an alias of GCN, otherwise drop | NEEDS CHECK |
| rodriguez2023einns | Rodriguez et al., EINNs, AAAI 2023 | arxiv.org/abs/2202.10446 | Physics-informed epidemic learning | OK |
| wang2022causalgnn | Wang et al., CausalGNN, AAAI 2022 | cdn.aaai.org/ojs/21479 | Epidemic GNN | OK |
| cao2023mepognn | Cao et al., MepoGNN, 2023 | arxiv.org/abs/2306.14857 | Metapopulation GNN | OK |
| deng2020colagnn | Deng et al., Cola-GNN, CIKM 2020 | arxiv.org/abs/1912.10202 | Epidemic GNN | OK |
| raissi2019pinn | Raissi et al., PINNs, J. Comput. Phys. 378, 2019 | none | Physics-informed loss | NEEDS CHECK |
| krishnapriyan2021failure | Krishnapriyan et al., PINN failure modes, NeurIPS 2021 | none | Why physics losses can fail | NEEDS CHECK |
| yoon2019timegan | Yoon et al., TimeGAN, NeurIPS 2019 | papers.nips.cc (manifest) | GAN augmentation arm | OK |
| esteban2017rcgan | Esteban et al., RCGAN, arXiv 1706.02633 | arxiv.org/abs/1706.02633 | GAN background | OK |
| kim2022revin | Kim et al., RevIN, ICLR 2022 | openreview.net/pdf?id=cGDAkQo1C0p | Normalization arm | OK |
| shao2022stid | Shao et al., STID, CIKM 2022 | arxiv.org/abs/2208.05233 | District-embedding arm | OK |
| salinas2020deepar | Salinas et al., DeepAR, Int. J. Forecasting 36, 2020 | arxiv.org/abs/1704.04110 | Likelihood-based forecasting | OK |
| bracher2021evaluating | Bracher et al., Evaluating epidemic forecasts in an interval format, PLoS CB 17, 2021 | arxiv.org/abs/2005.12881 | Interval metrics | OK |
| tissera2020severe | Tissera and Jayamanne, Severe dengue epidemic, Sri Lanka, 2017, EID 26, 2020 | none | 2017 epidemic context | NEEDS CHECK |
| clarke2024opendengue | Clarke et al., global dengue count dataset, Sci. Data 11, 2024 | none | Data context only if used | NEEDS CHECK |
| (not in any .bib) Chan and Johansson 2012 | Intrinsic incubation period of dengue | pmc.ncbi.nlm.nih.gov/articles/PMC3511440 (audit sheet) | Incubation 5.9 days comment | NEEDS CHECK: no bib entry, title and venue incomplete |
| wallinga2007generation | Wallinga and Lipsitch, generation intervals, Proc. R. Soc. B 274, 2007 | none | Only if renewal analysis (r² of log R_t) is cited | NEEDS CHECK |

Not proposed for citation: keys in the bib that the full paper does not cite (xie2022epignn, wang2022ntk, rusch2023oversmoothing, daw2021pidgan, hyndman2021forecasting, chen2016xgboost, wang2021gradient, wu2020mtgnn, nguyen2024mppinn) unless a section needs them.

## 8. Questions and gaps for you

1. **Paper settings are still placeholders.** Page limit, venue and author list were left in brackets. I assumed 6 pages plus references. Please fill them in.
2. **Graph file.** `origin/main` still uses the adjacency with two non-real edges (Kandy to Ampara, Kegalle to Kalutara). The fix (commit 6273b45, verified against GADM polygons) lives only on `adaptive-graph` and `praveen`. Every graph result in the plan uses the uncorrected graph. Options: re-run on the corrected graph (not allowed in this session), or state the limitation. Which do you want?
3. **Which protocol is the main table?** P3 (EXP-050, 3 origins, full 69-config ablation, cannot reach p < 0.25 at the origin unit) or P9 (9 origins, only 30 arms, post-fix code)? My proposal: P3 for the ablation tables, P9 for the significance claims, never mixed in one table. Confirm.
4. **EXP-061 arms were hand-picked** and the 46-config audit grid was never run. Using them as main results needs a sentence admitting this. Alternatively run the grid before submission.
5. **Branch-only results.** Decide whether the Phase-2/3 raw results and WGAN-GP on the legacy array go in (my suggestion: one short negative-results paragraph, PL label). They are the only evidence for the adaptive-graph + physics loss + GAN design in the original proposal.
6. **Five commits exist only on your local `praveen` branch** (Phase 3, ADR 0005; tip a0a5883). They are on no remote. Push or back them up if you want them kept.
7. **Draft claims that must change** (details in EVIDENCE Part A): C1 (the 0.36 to 0.67 is a no-training regression, not the anchor head), C3 (no published paper reports beating persistence; Weng et al. report no naive baseline), C5/C6 (25.42/0.023 and 25.53/0.047 belong to two different models, and both p-values are validation; test p is 0.694 and 0.576), C7 (42.10 not found; baseline is 42.7; one fold; persistence is 38.08), C8 (from a deleted, defective leaderboard), C9 (add widths 122.8 to 249.9).
8. **Statistical correction.** `docs/PROTOCOL.md` says Holm; `seirgnn2/stats.py` (which produced the numbers) uses Benjamini-Hochberg over 30 arms. Which should the paper state? I will state BH because that is what was run.
9. **Missing outputs.** (a) Saved file for the foi regression (EXP-060, only in the log). (b) Interval coverage and width on the corrected data (only legacy array). (c) TimeGAN hyperparameters. (d) Persistence per horizon for PK9. (e) The "original GNN baseline" on the corrected data: I used GCN direct on P9 (EV-025) and the six published encoders with direct heads on P3 (EV-028/EV-029). Confirm that is what you mean.
10. **Std convention.** The ± in the ledger is the sd across runs, dominated by differences between origins. For significance I propose reporting the paired delta and its sd across origins instead. Say if you prefer the other.
11. **Existing papers.** `paper/` already holds the ACM short paper (with `refs.bib` and `main.tex`) and `full_paper/` holds an 8-page paper. I created only new files in `paper/` and did not touch those. The IEEE draft will need its own subfolder or a rename (`paper/ieee/`) to avoid overwriting `paper/refs.bib` and `paper/main.tex`. Which?
12. **Local `main` is 167 commits behind `origin/main`.** I branched from `origin/main`. Say if you meant the stale local `main`.
13. **Seasonal-feature rationale.** The code cites a retracted finding (F9). The paper must justify the feature by its measured gain (val -0.76, 6/9 origins, not adopted) and not by F9.

### Primary sources added after approval (each confirmed from the document or the DOI record on 2026-10-05)

| Key | Source | What was checked | Cited for |
|---|---|---|---|
| epid2021wer0248 | Epidemiology Unit, Ministry of Health, Nutrition and Indigenous Medicine, Weekly Epidemiological Report Vol. 48 No. 02 | Opened the PDF (web.archive copy of the epid.gov.lk file): header "Vol. 48 No. 02", Table 1 dated "26th - 01st Jan 2021 (1st Week)". The file's PDF creation date is 2022-07-26 (a re-upload), so the year is the report's own, 2021 | The week-395 formula error (EV-192) |
| jeewandara2015sero | Jeewandara et al., Change in Dengue and Japanese Encephalitis Seroprevalence Rates in Sri Lanka, PLOS ONE 10(12): e0144799, 2015, doi 10.1371/journal.pone.0144799 | Crossref record: title, 11 authors, volume, article number. The value 0.682 (all ages, suburban Colombo, 2013 to 2014, n = 1689) is taken from `data/external/seir_parameters.json` (key colombo_sero2015); I did not re-read the paper's table | Initial susceptible fraction S0 = 1 - 0.682 |
| dcs2015census | Department of Census and Statistics, Census of Population and Housing 2012, Key Findings (ISBN 978-955-577-906-7) | Opened the PDF: title page, ISBN, publisher. Year 2015 is the PDF creation date (2015-05-08); the printed year was not found | 2012 Census district population (reference date 2012-03-20) |
| dcs2024midyear | Department of Census and Statistics, Mid-year population by district and sex, 2014 to 2024 | Downloaded: 3-page PDF, created and Last-Modified 2024-09-19, estimates 2014 to 2024. A 2025 file (one page, 2025* estimates, Last-Modified 2025-11-03) is also used; it is not in the bib yet | Mid-year population estimates and their publication dates |
| hersbach2020era5 | Hersbach et al., The ERA5 global reanalysis, QJRMS 146(730):1999-2049, 2020, doi 10.1002/qj.3803 | Crossref record; the Copernicus CDS ERA5 page lists this as its key reference. The dataset DOI 10.24381/cds.adbb2d47 (ERA5 hourly data on single levels, C3S CDS) resolves | ERA5 reanalysis |
| zippenfenig2024openmeteo | Zippenfenig, Open-Meteo.com Weather API, Zenodo, doi 10.5281/zenodo.7970649 | DOI record (year shown is the latest version, 2024) | How ERA5 was obtained |
| didan2021mod13q1 | Didan, MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN Grid V061, NASA LP DAAC, 2021, doi 10.5067/MODIS/MOD13Q1.061 | DOI record. MOD13Q1 processing delay was measured from the ORNL subset service (`proc_date`), not taken from this record | NDVI (not used by the preregistered arms) |
| gadm41 | GADM, database of Global Administrative Areas, version 4.1, https://gadm.org | Site states "The current version is 4.1". No release year was found on the page, so none is given | District polygons for the shared-border check |
| liu2025seirlstm | Liu et al., PLOS Comput. Biol. 21(9): e1013540, 2025 (already in the list) | Crossref record matches authors, volume, article number, year | SEIR design; the 11x reporting scale and the rates omega and gamma |

No entry was added for Chan and Johansson 2012 or for Yi et al. 2021 because the paper does not yet rely on them.

## Codex presentation continuation

The current title and contributions are deliberately narrower than the older core story above. Pooled association is not explained predictive variance; gated-versus-SEIR results are mixed; date-dependent additions are pending replacement. No final scientific conclusion is frozen by the presentation revision. Previously approved citations remain approved; metadata corrections and primary-source links are recorded in REFERENCE_AUDIT.md. The proposed eighteen-block new-period design remains pending. Author resolution of the newly verified target-overlap issue is required before freezing a fresh evaluation.


Final handoff: supplied ICITR authors incorporated; main is six pages including references, historical supplement two pages. See FINAL_AUDIT_CODEX.md for authoritative final status and blockers.
