# Codex handover audit

Audit date: 2026-10-05. This audit precedes manuscript edits.

## Repository and completed work

The original workspace is on `fix/week-index`, tip `008887d`, three commits ahead of its remote. User-owned untracked data/download files are present and are preserved. The IEEE draft is on `paper/ieee-draft`, tip `ef22ed3`, in `C:/Users/ASUS/Desktop/dengue-wf`. That worktree contains untracked week-index rerun outputs. The isolated review branch `paper/codex-final` starts at `ef22ed3` inside the authorized workspace. No branches were reset, merged, pushed or deleted.

The ACM manuscript is `paper/main.tex`; the IEEE manuscript is `paper/ieee/main.tex`. Preserve the ACM manuscript. Existing IEEE work includes a compiling draft, approved bibliography, reproducible figure/table scripts, interval comparison, evidence ledger, plan, style guide, repository map and external review with a point-by-point check. The handover is incomplete, not a finished confirmatory study.

## Partial and outstanding work

The data branch contains the repaired week index (`ac89792`), rebuilt covariates (`e2360c8`, `b0ae6d8`), corrected graph (`3878e70`), population/NDVI availability changes (`ce2f90b`), and frozen new-report parsing rules (`008887d`). EXP-062 in that branch's experiment log is explicitly pending. `docs/RERUN_SUMMARY.md`, a committed EXP-063 entry, new-week QC report, frozen development-selected settings, and new-period confirmatory outputs were not found. The older file named `s9_confirmatory_results.json` is not the newly proposed untouched-period evaluation.

The latest pasted decisions require eighteen confirmatory blocks, irregular-report scoring rules, identical missing-data treatment, constant 2012 Census population, validation-only tuning and a graph/self-loop check. These decisions are not implemented in the committed new-week rules: R4 still derives a seven-day start from the end date and population still uses publication-date vintages. Do not silently claim that the desired evaluation has run. Do not inspect new case values or launch training to finish a presentation task.

## Authority and supersession

`seirgnn2/results/frozen9_plus_audit.json` is the saved source of the draft's historical nine-origin results, not an untouched confirmatory result. `full_paper/outputs/kaggle_run/` is the saved EXP-050 three-origin/wide-nine run. Keep these values intact and label them development or historical. Both remain scientifically limited by the graph; season/climate/policy results also await week-index reruns. No final replacement result is available. Pending replacement is distinct from a completed superseding experiment.

The defective S5 leaderboard is excluded. Old legacy-array results are never interchangeable with rebuilt-series results. EXP-059 pre-fix physics arms are historical controls where EXP-061 post-fix arms exist. EXP-050 and EXP-056/057 date-dependent results are pending replacement under EXP-062; they must not support final claims. Later graph/population changes additionally require the approved reruns.

## Unsupported or overstated manuscript claims found before edits

- Pooled squared correlation is called explained variance; the claim that only eleven percent remains to explain is unsupported.
- Squared error on log counts is said to target the conditional median without the necessary distribution assumption.
- SEIR formulas omit the learned reporting-scale multiplier and centered spatial-import term in the implemented decoder.
- The graph is described as border adjacency although two directed edges are false borders.
- Test results are described as independently read after selection although earlier test windows were already examined before the convenience arms were chosen.
- Poor direct-head fits are called encoder failures without equal validation-only tuning.
- Abstract claims of no gain for climate/policy/season-based additions rely on results awaiting replacement.
- The proposed untouched evaluation is absent; non-significance cannot establish equivalence.
- Authors, venue, page limit and author confirmation of the AI-use statement remain missing.

## Figures and tables at handover

The IEEE body includes `fig_data.pdf`, `fig_arch.pdf`, `fig_valtest.pdf`; `fig_forest.pdf` exists but is omitted. The data plot uses the old date index, the diagram uses approximately six-point text and unclear anchor paths, and the scatter distinguishes families only by color and contains date-affected configurations. The omitted forest has tiny labels and no confidence intervals. Tables included: protocol overview, nine-origin main comparison, three-origin ablation and model-B negative ablation. The interval table is generated but omitted. The protocol overview mixes incompatible legacy/rebuilt baselines; the negative table and seasonal ablation rows await reruns. Detailed per-item decisions are recorded in `FIGURE_AUDIT.md`.

## Scope of this continuation

Complete source-based corrections, label existing evidence honestly, improve reproducible vector figures and tables, compile and inspect the PDF. Leave missing confirmatory experiments and scientific approval decisions explicitly unresolved. Reports live under `paper/`; all IEEE changes live under `paper/ieee/`.

## Additional verified conflict before the corresponding manuscript change

`seirgnn2/core.py::build_folds` partitions forecast **start indices**, with no horizon embargo. Direct enumeration of the repaired missing mask and the frozen nine origins confirms **two shared target weeks at every train/validation boundary and two at every validation/test boundary**. Thus early stopping can use validation targets that also belong to the first test forecasts. A chronological split of starts is not a disjoint split of target weeks. This is a scientific blocker for a clean held-out evaluation; corrected reruns require an author-approved boundary rule before any new run. The stored scores are preserved, not silently recomputed.

`seirgnn2/train.py:233` permutes training windows with `torch.randperm(n)`. The statement that windows are never shuffled is incorrect as a description of minibatch training; partitions are chronological. This differs from the pasted pre-registration's no-shuffling wording and must be resolved before that workflow is frozen.
