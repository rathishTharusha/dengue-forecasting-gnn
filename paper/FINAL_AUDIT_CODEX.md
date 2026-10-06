# Final correction audit ? 5 October 2026

Current paper: **Data Quality and Evaluation Design in Graph-Based Dengue Forecasting**.

The requested fallback is implemented as an evaluation/data-quality study. No final model-superiority claim is made. The classification was completed in `REVIEW_FIX_PLAN.md` before rewriting. The main IEEE PDF is six pages including references; the optional historical supplement is two pages. Sources remain canonical under `paper/ieee/`, with a tested `paper/main.tex` entrypoint and synchronized root artifacts. The old ACM entrypoint/bibliography are archived rather than overwritten without recovery.

## BLOCKERS

- **Human scientific approval and scope decision:** all authors must verify and approve the revised audit framing, results, source report evidence and AI disclosure. No approval is claimed in the paper. Confirm that this methodological study fits the intended ICITR track and whether a supplement is permitted.
- **A final model paper is still blocked:** no completed corrected-graph/embargoed/equally tuned/untouched-period study exists. Those experiments cannot be implied complete by wording. They are not needed to establish the narrow completed audit findings, but they are required if authors restore a claim about final model superiority or isolate a causal SEIR benefit.
- The current committed rules do not jointly specify the intended final target embargo, constant Census population, irregular-report scoring, fair tuning grid/tie rule and requested 18 confirmation blocks. An invented replacement workflow would violate the instruction to use only the frozen protocol.

## MAJOR ISSUES

1. Historical forecast targets overlap by two weeks at every training/validation and validation/evaluation boundary across nine origins. Early stopping therefore sees some outcomes in the first evaluation forecasts. Quantifying the impact requires new controlled runs; this paper only establishes the defect.
2. Historical scores use 116 directed neighbor entries plus 25 loader self-loops. The corrected graph has 57 shared borders, 114 directed entries and 25 self-loops (139 total), and passes the stored GADM border/symmetry test. Scores were not relabeled as corrected-graph results.
3. Shared fixed training settings do not demonstrate fair tuning. No architecture-inferiority claim survives. Retrospective arm selection and temporal dependence limit all reported inference.
4. The documented protocol calls for Holm, while historical software uses BH. The new table follows Holm, preserving raw and BH values separately. On the same 30-arm family, neither improvements nor degradations are significant under Holm on validation or evaluation. Earlier BH significance claims are no longer presented as protocol-compliant.
5. Reporting fraction, fixed progression rates and initial susceptibility are assumptions. The local serosurvey is not national susceptibility evidence. Cumulative initialization omits missing contributions and does not model serotypes/waning; no nine-origin sensitivity analysis establishes robustness.
6. The official source-report URL returned HTTP 403 during fresh web retrieval. The paper uses stored report-correction and extraction evidence. Authors should retain and visually verify the original source PDF in their research archive.

## MINOR ISSUES

- Main PDF compiles without warnings, overflow, missing citations or Type 3 fonts. The supplement retains one underfull paragraph warning with no clipping or overflow.
- All figure body labels are Times New Roman at 9 pt; a mathematical subscript in the forest plot is 6.3 pt. Vector diagrams remain editable in draw.io; arrows have visible shafts and filled heads.
- References begin on page 5 and continue onto page 6. The limit is six pages including references, not a requirement to fill all remaining space.
- The supplement is historical context, not central evidence. Its submission eligibility requires checking with the venue.

## EXPERIMENTS COMPLETED

No new model training was performed during this correction. Completed analyses and verifications are:

- Reproduced legacy persistence sensitivity directly from the original array and original forecast-window rule. Mean across three origins: 44.7953157332 versus 29.5210097089 after excluding six windows touching row 395, a 34.097998% reduction. Last-fold RMSE: 68.6183948927 versus 22.7954768198. This is an exclusion diagnostic, not corrected-data retraining.
- Verified the corrected graph at commit `3878e70` against its stored GADM border list and ran `tests/test_adjacency_borders.py` in `fix/week-index`: two tests pass. Removed entries are Kandy to Ampara and Kegalle to Kalutara, with no added borders.
- Rechecked nine-origin target-set intersections and saved `partition_overlap.json`.
- Reanalyzed 819 stored rows (30 neural arms plus persistence), retaining original scores and origin-level seed aggregation. Source SHA-256: `9e30a1d4d665bb00a86a441d9b49bb91d4953931db100f5e0fa8b2a3ce632584`.
- Independently checked the new Holm implementation against `statsmodels.stats.multitest.multipletests(method="holm")` for both validation and evaluation families; all 30 adjusted values agree.
- Regenerated horizon/overall RMSE, MAE, normalized skill, paired differences, descriptive t intervals and seed variability as CSV/JSON and tables. GCN residual RMSE 28.1194 versus persistence 28.5410; adaptive anchored E0 variant RMSE 27.4451, delta -1.0959, interval [-2.8426, 0.6508], raw p 0.19921875, BH p 0.576171875, Holm p 1.0. Unrounded intervals are authoritative in the JSON.
- Verified all 15 active references against publisher-deposited metadata and/or primary author/publisher records; see `REFERENCE_AUDIT.md` for the primary-report access limitation.

## EXPERIMENTS STILL MISSING

- Completed EXP-062 calendar-dependent replacement outputs and an accepted rerun summary.
- Corrected-graph neural comparison with an explicit target embargo.
- Equal validation-only tuning budgets and all frozen selection/tie-breaking rules.
- Classical seasonal-naive, autoregressive and NB/count-regression baselines under the corrected final protocol. Archived legacy classical results cannot be inserted into the nine-origin table.
- Matched adaptive gated non-SEIR, bare SEIR and gated SEIR controls with shared data/training settings; parameter sensitivity on the same final protocol.
- Final population/report-timing rules, parsed/QC-certified new-period inputs, frozen selection and untouched-period confirmation.

Repository-wide branch/output checks found no superseding final evidence. EXP-062 remains pending at `008887d`; `docs/RERUN_SUMMARY.md` and prepared `data/new_weeks` outputs are absent. External untracked COVID week-fix output declares itself exploratory, has unverified vintage and does not certify the corrected final graph/protocol. Older files called `confirm` or `s9_confirmatory` use historical periods. Exact existing commands and missing workflow decisions are recorded in `REVIEW_FIX_PLAN.md`; no new-period cases were inspected.

## CLAIMS REMOVED OR WEAKENED

- Replaced model-centered development framing with a completed audit question and three completed contributions.
- Removed abstract/conclusion promises of core experiments as though the model study were finished.
- Replaced ambiguous ?without that week? wording with the exact affected-window exclusion rule and estimator.
- Removed inference that earlier gains were caused by the audit defects; the measured source sensitivity and unmeasured leakage effect are distinguished.
- Removed BH-based significance language from protocol-compliant conclusions; explicitly documented the correction discrepancy.
- Removed unarchived force-of-infection diagnostic claims, isolated SEIR-benefit language, universal susceptibility interpretations, and irrelevant legacy augmentation claims from the main narrative.
- Retained unavailable corrected and confirmatory evidence in a single scope/limitations discussion. No limitation was hidden as a positive result.
- Replaced the printed ?author approval pending? sentence with a factual disclosure; approval remains an explicit human action here.

## FIGURES CHANGED

- Source-series figure moved into the main paper to support the audit contribution (Fig. 1).
- Architecture separates non-SEIR and SEIR/gated-SEIR branches, persistence anchor, population/state inputs, daily S -> E -> I -> R path, forecast output and training-only loss (Fig. 2). Editable native draw.io source retained.
- Protocol distinguishes the observed overlapping-target design from a clearly labeled recommendation to freeze settings before untouched confirmation (Fig. 3). It does not portray confirmation as completed.
- Paired-difference forest plot retained with descriptive uncertainty and explicit zero meaning (Fig. 4).
- All figures remain vector, serif, readable at final widths and interpretable in grayscale. Final renders and compiled pages were inspected.

## TABLES CHANGED

- Main historical table regenerated from saved structured outputs; adjusted values now use Holm across all 30 arms. Caption states provenance, seed/origin averaging, uncertainty and the historical BH archive. No best-score bolding.
- CSV/JSON retain raw p, historical BH p, Holm p, horizon metrics, MAE, paired differences/intervals, skill and within-origin seed SD.
- Supplement preserves incompatible three-origin and legacy protocols separately. Root `paper/tables/` mirrors the canonical generated files.

## REFERENCES CORRECTED

- Weng: added published eighth author Mahi Pasarkar and page range 4448?4456; documented the seven-author local prepublication version.
- A3T-GCN: corrected Jiandong Bai, Yujiao Song and Zhixiang Hou; added Haifeng Li and DOI 10.3390/ijgi10070485.
- DengueGNN: completed issue metadata. Removed unused entries from the active bibliography while preserving original archives.
- IEEE AI guidance checked; acknowledgment names actual tools, drafting throughout the text, source/statistical checking and figure/table assistance without inventing human approval.

## FINAL VERDICT

**NOT READY FOR SUBMISSION**

The corrected audit manuscript is compiled and ready for the authors to review. Submission still requires their scientific approval, acceptance of the audit framing and venue/disclosure checks. The missing final model experiments are disclosed and excluded from the completed contributions; a model-superiority paper remains scientifically incomplete. This verdict does not imply that new model training is necessary to substantiate the narrow source/split/statistical audit findings already reported.

Validation: `review_analysis.py`, `make_figures.py`, `make_tables.py`, `make_bib.py`, `check_paper.py`, `audit_publication.py`; independent Holm cross-check; two geographic-border tests; direct LaTeX/BibTeX builds. Root and canonical PDFs have identical extracted text and six pages. Original scientific checkout, external worktree and saved model outputs are unchanged. No model training, confirmatory access, remote push or submission was performed.
