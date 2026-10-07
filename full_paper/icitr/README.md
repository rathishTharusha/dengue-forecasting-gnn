# ICITR 2026 version (IEEE, 6 pages, double-blind)

Rules: <https://icitr.uom.lk/forauthors/>. **Deadline 10 Oct 2026, 23:59 Sri Lanka time**,
Microsoft CMT, PDF only. IEEE two-column A4, **6 pages max including references**,
double-blind. Track: Artificial Intelligence (or Data Analytics & Decision Making).

The ACM version in `../overleaf/` is the long reference draft. This folder is the one we submit.

## How to write

Each file in `sections/` starts with a comment block: what the section must say, its
page budget, and the numbers you may use. Write in your own words where the red
`\writehere{...}` markers are, then delete the marker. Interpret results yourself;
use a tool only to polish language (Madam's rule).

| Section | File | Budget | Writer |
|---|---|---|---|
| Abstract (write last) | `00_abstract.tex` | 200 words | |
| Introduction | `01_introduction.tex` | 3/4 page | |
| Related work | `02_related_work.tex` | 1/2 column | |
| Dataset | `03_dataset.tex` | 1/2 column | |
| PAGE framework | `04_page.tex` | 1 page with Fig. 1 | |
| Experimental setup | `05_setup.tex` | 1/2 column | |
| Results | `06_results.tex` | 1.5 pages with 3 tables | |
| Discussion + conclusion | `07_discussion_conclusion.tex` | 1/3 page | |

## Tables and figure

Tables in `tables/` are generated; never edit them by hand:

```bash
python scripts/build_paper_assets.py        # encoders, rescue, levers, nine
python scripts/build_paper_extra_tables.py  # paired, tuning, compute, prospective
```

Fig. 1 is TikZ in `figures/overview.tex`.

## Build

```bash
cd full_paper/icitr && latexmk -pdf main.tex
```

Overleaf: zip this folder, New Project → Upload Project. Compiler pdfLaTeX.

## Before submitting

- [ ] `grep -n writehere sections/*.tex` prints nothing
- [ ] 6 pages or fewer, including references
- [ ] no names, no university, no "our previous work"; `\codeurl` is the anonymous mirror
- [ ] every number checked against the generated tables
- [ ] EXP-062 status checked (B, season and climate numbers)
- [ ] PDF passes IEEE PDF eXpress if the conference asks for it
