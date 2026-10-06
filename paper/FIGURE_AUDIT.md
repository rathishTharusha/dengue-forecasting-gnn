# Figure and table audit

2026-10-05. Scope: the IEEE draft in `paper/ieee/`; the ACM paper remains an archive and is not silently restyled. All final figures are reproducible vector PDFs. Scientific scores are unchanged.

| Figure | Scientific purpose | Source data/code | Source script | Original PDF dimensions | Strengths | Problems | Decision | Reason | Final target |
|---|---|---|---|---|---|---|---|---|---|
| Fig. 1 / fig_data | Identify source error and display rebuilt case chronology | Legacy channel 5; rebuilt_cases.npy; repaired report calendar | make_figures.py ? publication.py | 3.34 x 2.29 in | Good paired source comparison | Old date labels; tiny labels; old backlog annotation | Redesign | Use repaired plot calendar without changing historical scores | 3.5 x 3.25 in; 9 pt; PDF |
| Fig. 2 / fig_arch | Explain encoder, selected head, persistence and SEIR inputs | models.py, train.py, seir_sim.py | fig_arch.py ? publication.architecture | 2.75 x 1.93 in | Contains encoder/head distinctions | Tiny colored cards; crossing paths; missing direct anchor path; training loss mixed with inference | Redesign | Separate conditional inputs and training-only loss | 7.16 x 2.55 in; 9 pt; PDF |
| Omitted forest ? Fig. 4 / fig_forest | Show magnitude and uncertainty of paired differences | frozen9_plus_audit.json; development_statistics.json | publication.statistics | 3.48 x 3.30 in | Original shows origin variation | 13 long labels at under 6 pt; no intervals | Redesign and include | Four baseline/anchored comparisons with descriptive t intervals; selection is explicit | 3.5 x 2.75 in; 9 pt; PDF |
| Old Fig. 3 / fig_valtest | Show validation/test disagreement | kaggle_run/runs.jsonl; results.json | historical_figures.fig_valtest | 3.33 x 2.14 in | Uses all 69 configurations | Date-affected season/climate arms; color-only families; 6?7 pt labels | Remove from manuscript, preserve archive | Pending corrected-data replacements; not final evidence | Do not include until reruns are verified |
| New Fig. 3 / evaluation_protocol | Explain development versus planned confirmation | core.build_folds; pasted frozen-design decisions; no new target values | publication.protocol | New vector PDF | Explicit pending state; patterned and outlined regions | Target overlap must be stated even though forecast-start regions are adjacent | Add | Protocol status is central to interpretation | 3.5 x 2.0 in; 9 pt; PDF |

## Tables

| Table | Purpose/source | Problem at handover | Decision and final target |
|---|---|---|---|
| I: protocols | P9 and P3 definitions; docs/PROTOCOL.md and core.py | Legacy/rebuilt baselines shared one table; tiny type | Remove legacy numerical row; list rebuilt development protocols without mixed scores; booktabs |
| II: main_p9 | Stored nine-origin models; frozen9_plus_audit.json | Seed and origin SD conflated; no MAE or intervals | Preserve all displayed model scores, add MAE and descriptive paired 95% intervals; unchanged sign-flip/BH over thirty arms; no bold winner |
| III: ablation_p3 | Historical three-origin head/encoder comparison; runs.jsonl | Seasonal rows need reruns; wins count seed replicates, not independent origins | Remove two seasonal rows; label historical and untested; preserve horizon scores; rename GCN residual consistently |
| Omitted negative_p3 | Changes to NB+season model B; runs.jsonl | Control itself uses bad date joins | Remove from manuscript; retain historical table and source |
| Omitted intervals | Legacy residual versus Gaussian interval comparison | Legacy-array protocol, not final rebuilt evaluation | Keep archive; text always pairs coverage with width and labels source-table-error folds |
| New seed_variability (supplement) | Within-origin SD and origin-normalized skill; frozen9_plus_audit.json | Seed spread not separated | Generate optional booktabs table and CSV for every stored arm; main body uses paired intervals |

## ACM figure archive (preserved)

The following paired PDF/PNG assets are not included in the IEEE manuscript and remain unchanged: `fig_artifact` (source anomaly), `fig_results` (architecture performance), `fig_physics` (physics additions), `fig_ceiling` (mechanistic ceiling), `fig_detection` (early-warning comparison), `fig_scoring_gap` (loss/score mismatch). Their existing generator is `paper/_build/figures.py`; the scoring-gap asset has no function in that generator and needs provenance checking before reuse. Existing `ACM_CHECK.md` remains authoritative for the separate ACM draft?s unresolved issues. None is promoted to final IEEE evidence.

## Actual-size and grayscale review

Every included PDF was rendered at its final physical width, then checked in the compiled two-column manuscript. Arial/DejaVu Sans body labels are 9 pt. The learned exposed-state subscript is smaller mathematical typography, not a general label. PDF text sizes and physical dimensions are recorded by `audit_publication.py`; all included figures contain no raster images. Grayscale renders use black outlines, dashed boundaries, hatching or direct labels rather than color-only encoding. The data panels show a single series each. The forest plot uses a single point/interval style and model labels. The protocol hatch labels use a white backing for readability.

No geographic graph hairball, cherry-picked outbreak example, decorative schematic, or unverified learned-transmission interpretation is added. Confidence intervals are explicitly descriptive, not evidence that the original split or retrospective selection has become valid.


## Final six-page layout (supersedes earlier placement notes)

The final main manuscript contains Figure 1 (architecture, double column), Figure 2 (evaluation protocol, single column), Figure 3 (paired development differences, single column), and Table I (nine-origin historical comparison). The data-series figure, descriptive protocols table, three-origin ablation, legacy interval comparison and seed-variation table are in the separate two-page supplement. The validation/test scatter and date-sensitive negative-ablation claims are omitted. All six main pages and both final supplementary pages were visually inspected. Main compilation has no warnings; the supplement has one harmless underfull paragraph warning, with no overflow or missing references. Supplement table placement is visible in the rendered PDF and acceptable for internal author review.


## Follow-up diagram redesign

At the user's request, the architecture and evaluation chronology were redesigned with editable native draw.io sources (`ieee/diagrams/architecture.drawio` and `ieee/diagrams/evaluation_protocol.drawio`). Rounded boxes, consistent spacing, restrained fill colors and routed arrows improve the hierarchy. The architecture distinguishes forecasting, conditional inputs and training-only loss; the protocol explicitly retains target overlap and pending confirmation. `drawio_diagrams.py` reads the editable XML and exports vector PDFs; the normal figure entrypoint delegates to it and preserves existing source edits. Both diagrams and their grayscale previews were inspected, followed by the compiled pages 3 and 4. Main remains six pages, warning-free, with 9-point diagram labels and no raster images. Stored scientific results are unchanged.


## Typography and arrow correction

All four included figures now use Times New Roman serif labels, with STIX mathematical glyphs in plots, matching the manuscript?s Times-style text. This supersedes the earlier Arial font notes. Both editable draw.io files were updated along with their generator defaults. Diagram arrows now use dark 1.15-point strokes, larger filled heads rendered above box borders, and the protocol has wider inter-box gaps. Color previews of all four figures and the compiled protocol page were visually inspected. Main PDF remains six pages with no LaTeX warnings; vector/text checks pass.


## Final review layout and content (authoritative)

The methodological-audit paper has four main figures: source series (single column), architecture (double column), observed/recommended protocol (single column), and paired-difference forest (single column). Architecture now separates non-SEIR and SEIR/gated-SEIR branches and shows S?E?I?R. The protocol explicitly labels its lower sequence as a recommendation with no confirmation result reported. Existing serif labels, vector export and dark arrows are retained. Main Table I reports historical scores with Holm rather than BH values; no incompatible protocols are pooled. The supplement now contains historical interval, protocol, three-origin ablation and seed-statistic tables, without duplicating the source-series figure. Both PDF entrypoints compile the same six-page main paper.
