# Current research paper

Current title: **Data Quality and Evaluation Design in Graph-Based Dengue Forecasting**.

Open `paper/main.pdf` or `paper/ieee/main.pdf`. The root `main.tex` routes to the canonical source `ieee/main.tex`; both build the same IEEE paper. The old ACM entrypoint and bibliography are preserved as `main_acm_archive.tex` and `refs_acm_archive.bib` with their original section/assets still in place. They are historical and are not this submission.

Read `REVIEW_FIX_PLAN.md` and `FINAL_AUDIT_CODEX.md` for the scientific scope and remaining decisions. The main document is an evaluation/data-quality audit. It does not claim completion of a corrected final model comparison or independent confirmation.

Rebuild instructions are in `ieee/README_CODEX.md`. After generation and the IEEE PDF build, run `python paper/scripts/publish_review.py` from the repository root to synchronize root-facing artifacts. Editable diagrams are in `ieee/diagrams/`. The separate supplement is optional and requires the venue's permission; all central audit findings are in the main paper.

The compiled `paper/main.pdf` is tracked for direct viewing on GitHub. After changing the manuscript, rebuild and synchronize it before committing. Other LaTeX build artifacts remain ignored.
