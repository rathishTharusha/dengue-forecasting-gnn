# Rebuilding the development draft

This is a historical **development** manuscript, not a completed confirmatory paper. Read `../FINAL_AUDIT_CODEX.md` before using it for submission. The ACM paper outside this directory is preserved.

Run from the repository root:

```text
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
```

`publication.py` is the canonical entrypoint for the included figures and nine-origin table. The architecture and protocol diagrams are rendered from editable native draw.io files in `diagrams/` by `scripts/drawio_diagrams.py`; see `diagrams/README.md`. It reads only existing saved results, preserves the project sign-flip/BH calculations, and adds descriptive paired t intervals. No training or new-period case inspection occurs. Input hashes, unrounded statistics, and partition-overlap evidence are saved under `results/`. The dates in the data figure use the verified repaired calendar; historical model scores retain their original data/graph provenance.

`historical_figures.py` and `historical_tables.py` preserve the old generators for traceability; do not run them as final publication entrypoints. The current wrappers prevent included figures and tables from reverting to the old designs. Omitted negative-ablation and interval tables remain historical artifacts. The supplementary `tables/seed_variability.tex` and CSV statistics separate within-origin seed variation from between-origin uncertainty; the optional table is not included in the space-limited main manuscript.

The manuscript can be compiled from this directory on Overleaf with the already generated PDF figures and TeX tables. Regeneration requires the original saved source data from the repository. Standard Python dependencies: numpy, pandas, scipy, matplotlib and pymupdf. Figures use Times New Roman with a STIX serif fallback and STIX mathematical glyphs to match the manuscript?s Times-style typography. The current environment's latexmk lacks Perl, so the direct pdflatex/BibTeX sequence is used.
