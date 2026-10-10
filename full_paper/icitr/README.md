# ICITR 2026 version (IEEE, 6 pages, double-blind)

Rules: <https://icitr.uom.lk/forauthors/>. **Deadline 10 Oct 2026, 23:59 Sri Lanka time**,
Microsoft CMT, PDF only. IEEE two-column A4, **6 pages max including references**,
double-blind. Track: Artificial Intelligence (or Data Analytics & Decision Making).

This folder is the paper we submit. `../overleaf/` is the long ACM draft (9 pages) that
keeps the material cut from here: the benchmark audit table, every lever, the nine-origin
confirmation and the validation-test analysis.

## Layout

| File | Content |
|---|---|
| `main.tex` | IEEE template, title, anonymous author block, section order |
| `sections/00_abstract.tex` | abstract and index terms |
| `sections/01_introduction.tex` | introduction, written by the team |
| `sections/02_related_work.tex` | ST-GNNs, mechanistic hybrids, simple baselines |
| `sections/03_dataset.tex` | benchmark audit, corrected dataset, 2024-2026 extension |
| `sections/04_page.tex` | the PAGE framework: formulation, head, SEIR branch, loss, implementation |
| `sections/05_setup.tex` | dataset overview, baselines, protocol |
| `sections/06_results.tex` | comparative analysis, ablation, prospective test, tuning and cost |
| `sections/07_discussion_conclusion.tex` | discussion, limitations, conclusion |
| `figures/overview.tex` | Fig. 1, TikZ |
| `tables/*.tex` | generated; never edit by hand |

## Tables

```bash
python scripts/build_icitr_tables.py   # encoders, rescue, compute, prospective
```

It calls the same functions as the ACM draft's builders, on the same run outputs
(EXP-050 Kaggle run, EXP-063 prospective), so the two versions cannot disagree. The other
files in `tables/` (paired, tuning, levers, nine) are not used by this version.

## Build

```bash
cd full_paper/icitr && latexmk -pdf main.tex
```

Overleaf: zip this folder, New Project -> Upload Project. Compiler pdfLaTeX.

## Before submitting

- [ ] `\codeurl` in `main.tex` points at a real anonymous mirror (anonymous.4open.science)
- [ ] EXP-062 checked: the lines marked `% EXP-062` in `sections/06_results.tex`
- [ ] 6 pages or fewer, including references
- [ ] no names, no university, no "our earlier work"; PDF metadata has no author
- [ ] every number checked against the generated tables
- [ ] for grading: colour-highlighted copy with one colour per member (CS3631 brief, section 11)
