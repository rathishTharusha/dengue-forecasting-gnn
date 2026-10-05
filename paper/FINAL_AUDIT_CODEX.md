# Final Codex audit ? 5 October 2026

## Outcome

The IEEE development manuscript is compiled and reviewed: **6 pages including references**, with the supplied six authors, University of Moratuwa affiliation and email addresses, for the intended ICITR venue. The optional historical supplement is **2 pages**. This is a defensible historical development draft, not a submission-ready claim of independent confirmation. No model was trained, no new-period case values inspected, and no stored scores overwritten.

Work is isolated on `paper/codex-final`, based on `paper/ieee-draft` (`ef22ed3`), in `.codex-paper-review`. The original `fix/week-index` checkout, user files and external `dengue-wf` worktree are preserved. No push or submission is performed.

## Review verdicts

| Area | Status | Evidence or remaining limitation |
|---|---|---|
| Scientific correctness | Needs attention | Stored arithmetic and code descriptions verified; original fold boundary leakage and missing corrected runs prevent clean held-out conclusions. |
| Experimental consistency | Needs attention | Legacy, rebuilt development and pending confirmation are distinguished. Corrected calendar/graph/population experiments are incomplete. |
| Figures | Pass | Four vector assets regenerated, final sizes and fonts checked, color/grayscale and compiled-page inspection completed. Three appear in main; data figure is supplementary. |
| Tables | Pass for historical reporting | Original statistics preserved; paired effects, descriptive intervals, MAE and separate seed variability added without changing significance testing. Protocols are not pooled. |
| References | Needs author review | Duplicate/missing keys: zero; all cited entries have DOI or URL. Approved metadata corrections checked against primary sources. Full reading/support verification remains an author responsibility; uncertain unused entries are documented separately. |
| IEEE formatting | Pass for six-page draft | Main 6 pages, references start on page 6, 207-word abstract, no LaTeX warnings, no Type 3 fonts. Venue-specific submission requirements still require author confirmation. |
| Writing | Pass for development scope | Unsupported novelty, causal/biological interpretation and explained-variance ceiling claims removed or narrowed. Human authors must approve the final text and AI-use statement. |

## Critical scientific blockers

1. **Fold targets overlap.** `build_folds` divides forecast starts without a horizon embargo. For horizon three, two target weeks are shared at every training/validation and validation/test boundary across all nine origins. Early stopping therefore sees target weeks present in initial test forecasts. `ieee/results/partition_overlap.json` records the enumeration. Decide and freeze an embargoed protocol before rerunning; do not present current intervals or adjusted p-values as curing this design problem.
2. **Corrected-data evidence is pending.** The manuscript scores retain historical graph/calendar provenance. Corrected graph, climate/season alignment, population timing and current report rules require EXP-062/063 completion. The requested 18 confirmatory blocks, fixed 2012 population, irregular-report scoring rule, QC and frozen configuration are not a completed committed evaluation.
3. **Independent confirmation is absent.** Files named confirmatory in older runs are not evidence of the newly intended untouched-period experiment. Freeze configuration, preprocessing and evaluation before that experiment; preserve its untouched status.
4. **Mechanistic attribution is unresolved.** Equal validation-only tuning, matched gated-head controls, meaningful classical baselines and SEIR sensitivity/ablation remain incomplete. Reported failures under shared settings do not establish architecture inferiority or irrelevance of epidemic physics.

## Verified numerical results

The saved nine-origin source SHA-256 is `9e30a1d4d665bb00a86a441d9b49bb91d4953931db100f5e0fa8b2a3ce632584`. It contains 31 arms including persistence, 819 rows, nine origins and three seeds for neural arms. The unchanged exact sign-flip/BH procedure gives zero test improvements and eight degradations among 30 comparisons; validation gives seven improvements and eight degradations. Persistence test RMSE is 28.5410 and MAE 13.9386. The GCN residual baseline has RMSE 28.1194, paired difference -0.4216, descriptive 95% t interval [-1.5391, 0.6959], adjusted p=0.78125. These are retrospective development summaries with correlated origins and selection limitations.

## Changes delivered

- Main manuscript rewritten around development evidence and comparison with persistence; supplied authors added and content reduced to six pages including references.
- Architecture, evaluation timeline, paired-difference forest and data comparison regenerated as reproducible vector figures. Validation/test scatter omitted; historical material moved to the supplement.
- Main table regenerated with RMSE, MAE, paired differences, descriptive intervals and unchanged adjusted p-values. Full descriptive statistics and seed variation are available as CSV/JSON and supplementary table.
- Methods corrected against code: transformed conditional mean, biased graph layer, effective reporting fraction, learned exposed-state initialization, centered spatial term, mass-action dynamics, shuffled training minibatches and target overlap.
- Citation metadata corrected with documented provenance; evidence ledger extended and earlier plans marked historical.
- Rebuild entrypoints, source hashes, PDF/citation checks and audit scripts added. Historical generators retained as archives.

## Validation performed

`make_figures.py`, `make_tables.py`, `make_bib.py`, `check_paper.py` and `audit_publication.py` were run during this review. Direct pdflatex/BibTeX builds succeed; latexmk cannot run because this installation lacks Perl. Final main build has no warnings, undefined references/citations, overflow or Type 3 fonts. Supplement has one underfull paragraph warning and no overflow or missing references. Every included figure has zero raster images; body labels are 9 pt, with a mathematical subscript at 6.3 pt. All six main pages, both final supplementary pages, and grayscale figures were visually inspected. `git diff --check` passes. Generated PDF builds are intentionally ignored by the repository; figure PDFs and all sources/audits are committed.

## Exact files for manual review

1. `ieee/main.pdf` and `ieee/main.tex`: final six-page manuscript, especially abstract, protocol limitation, results and conclusion.
2. `ieee/supplement.pdf` and `ieee/supplement.tex`: internal historical supplement; decide whether the venue permits submission of it.
3. `CODEX_HANDOVER_AUDIT.md`: pre-edit scientific findings and branch/source inventory.
4. `FIGURE_AUDIT.md`: figure/table redesign decisions and final layout.
5. `REFERENCE_AUDIT.md`: primary-source corrections and remaining bibliographic uncertainty.
6. `EVIDENCE.md`, `ieee/results/development_statistics.csv`, `ieee/results/partition_overlap.json` and `ieee/results/publication_checks.json`: claim provenance and machine-readable checks.

Before submission, the authors must resolve the fold embargo and final evaluation protocol, complete the corrected and untouched-period experiments, verify the manuscript independently, approve the AI-use statement, and check ICITR instructions. None of these unresolved scientific decisions is silently assumed complete.


## Follow-up diagram redesign

At the user's request, the architecture and evaluation chronology were redesigned with editable native draw.io sources (`ieee/diagrams/architecture.drawio` and `ieee/diagrams/evaluation_protocol.drawio`). Rounded boxes, consistent spacing, restrained fill colors and routed arrows improve the hierarchy. The architecture distinguishes forecasting, conditional inputs and training-only loss; the protocol explicitly retains target overlap and pending confirmation. `drawio_diagrams.py` reads the editable XML and exports vector PDFs; the normal figure entrypoint delegates to it and preserves existing source edits. Both diagrams and their grayscale previews were inspected, followed by the compiled pages 3 and 4. Main remains six pages, warning-free, with 9-point diagram labels and no raster images. Stored scientific results are unchanged.
