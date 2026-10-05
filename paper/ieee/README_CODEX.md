# Rebuilding the evaluation-audit paper

This is an **evaluation/data-quality audit** using historical development scores; it does not report independent confirmation. Read `../FINAL_AUDIT_CODEX.md` before using it for submission. The old ACM entrypoint and bibliography are preserved as `../main_acm_archive.tex` and `../refs_acm_archive.bib`. The current root `../main.tex` builds this IEEE paper.

Run from the repository root:

```text
python paper/ieee/scripts/review_analysis.py
python paper/ieee/scripts/make_figures.py
python paper/ieee/scripts/make_tables.py
python paper/ieee/scripts/make_bib.py
python paper/ieee/scripts/check_paper.py
```

Build in `paper/ieee/`:

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Then run from the repository root:

```text
python paper/ieee/scripts/audit_publication.py
python paper/scripts/publish_review.py
```

`publication.py` is the canonical entrypoint for the included figures and nine-origin table. The architecture and protocol diagrams are rendered from editable native draw.io files in `diagrams/` by `scripts/drawio_diagrams.py`; see `diagrams/README.md`. It reads only existing saved results, preserves the historical sign-flip/BH values, adds the protocol-specified Holm correction, and reports descriptive paired t intervals. No training or new-period case inspection occurs. Input hashes, unrounded statistics, and partition-overlap evidence are saved under `results/`. The dates in the data figure use the verified repaired calendar; historical model scores retain their original data/graph provenance.

`historical_figures.py` and `historical_tables.py` preserve the old generators for traceability; do not run them as final publication entrypoints. The current wrappers prevent included figures and tables from reverting to the old designs. Omitted negative-ablation and interval tables remain historical artifacts. The supplementary `tables/seed_variability.tex` and CSV statistics separate within-origin seed variation from between-origin uncertainty; the optional table is not included in the space-limited main manuscript.

The manuscript can be compiled from this directory on Overleaf with the already generated PDF figures and TeX tables. Regeneration requires the original saved source data from the repository. Standard Python dependencies: numpy, pandas, scipy, matplotlib and pymupdf. Figures use Times New Roman with a STIX serif fallback and STIX mathematical glyphs to match the manuscript's Times-style typography. The current environment's latexmk lacks Perl, so the direct pdflatex/BibTeX sequence is used.

Compile the optional supplement with two `pdflatex -interaction=nonstopmode -halt-on-error supplement.tex` passes in this directory. The source-audit script resolves the corrected graph at commit `3878e70`; retain repository history when reproducing that check. The main build is also supported from `paper/` using the root entrypoint and synchronized references/assets.
