# Full paper and its reproduction package

**Title:** *Where Does Physics Help a Graph Network? A Leakage-Controlled Study of
SEIR-Informed Spatio-Temporal GNNs for Dengue Forecasting*
**Authors:** Group 05 (CS3631). The Phase-2 short paper (*Know When the Epidemic Comes*,
on the benchmark array) is kept unchanged in `paper/`.

Every number, table and figure in this paper comes from **one notebook run from original
source files**. The notebook reads no result computed elsewhere: it verifies the sources,
rebuilds the data, defines every model in plain PyTorch, trains all 810 runs and analyses
only those runs.

---

## Layout

```
full_paper/
├── full_paper.pdf                 # the compiled paper (8 pages with references)
├── overleaf.zip / overleaf/       # LaTeX source; upload the zip to Overleaf
├── REVIEW_NOTES.md                # notes for co-author review
├── kaggle/
│   ├── dengue_physics_gnn.ipynb   # THE reproduction notebook (generated; see below)
│   ├── src/*.py                   # its cell sources, one file per section
│   ├── fetch_sources.py           # downloads the original source files -> kaggle/dataset/
│   ├── dataset/                   # (git-ignored) the Kaggle dataset fetch_sources.py builds
│   └── kernel/                    # kernel-metadata.json + a copy of the notebook for `kaggle kernels push`
├── outputs/kaggle_run/            # the outputs of the run the paper reports (EXP-050)
├── data/                          # reference copies of the derived tables; the notebook does not read them
└── references/                    # full-text PDFs of the cited papers
```

---

## Run it on Kaggle

The notebook needs only its source dataset: CPU, internet off.

**1. Build the dataset from the original sources** (repository root, internet on):

```bash
python full_paper/kaggle/fetch_sources.py
```

This downloads each file from where it was originally obtained and checks its SHA-256
against the hash recorded at first retrieval (`data/external/source_manifest.csv`). It
also writes `SOURCES.csv` (file, bytes, hash, URL, date) and `dataset-metadata.json`.
The result is 58 files and 42.5 MB:

| folder | files | source |
|---|---|---|
| `graph/` | district adjacency, GADM 4.1 boundaries, district config | benchmark authors' repository `MLOpenSourceOpenScience/disease_modeling_MLOS2` |
| `benchmark/` | `sri_lanka_2013-2022_shifted.npy` (audited only, never trained on) | same repository |
| `cases/` | `output_Dengue Fever.csv`, the table parsed from the Weekly Epidemiological Reports | received from the benchmark authors; no public URL, copied byte for byte |
| `cases/` | WER Vol 48 No 02 PDF (the one corrected report) | Epidemiology Unit URL via its Internet Archive capture; the hash matches the original download |
| `climate/` | ERA5 daily series, one interior point per district | Open-Meteo Historical Weather API |
| `population/` | mid-year estimates 2014–2024 and 25 Census 2012 district reports | Department of Census and Statistics |
| `wheels/` | PyMuPDF 1.28.2, for reading the PDFs offline | PyPI |

**2. Upload the dataset** as a private Kaggle dataset, from the web UI (drag in
`full_paper/kaggle/dataset/`) or the CLI:

```bash
cd full_paper/kaggle && kaggle datasets create -p dataset -r zip
```

**3. Run the notebook.** Upload `full_paper/kaggle/dengue_physics_gnn.ipynb` as a
new notebook, attach the dataset, choose CPU and turn internet off, then **Save Version → Save
& Run All**. Or push the prepared kernel:

```bash
cd full_paper/kaggle && kaggle kernels push -p kernel
```

The notebook finds the dataset wherever Kaggle mounts it. The first code cell sets
`PROFILE = "full"` (810 runs, about 3 h on Kaggle's 4 CPU cores) or `"quick"` (a few
minutes, for checking the pipeline only; **never quote its numbers**). Outputs go to
`/kaggle/working/outputs/`. Every table is also printed in the notebook.

It also runs locally (`jupyter nbconvert --execute`) with the dataset at
`full_paper/kaggle/dataset/`, given numpy, pandas, matplotlib, torch and pymupdf.

### What the notebook does, in order

1. **Sources** — verifies the SHA-256 of all 58 files and installs PyMuPDF from the bundled wheel.
2. **Data** — rebuilds the 559-week case series (2013-W26 to 2024-W10, 7 missing weeks left
   missing), reading the week-395 correction from the printed PDF table; aggregates ERA5 to
   weeks; parses population from the census and mid-year PDFs; builds the graph and the
   frozen folds with causal lags (cases ≤ t−1, climate ≤ t−2).
3. **Benchmark-array audit** — dates the array's rows against the reports, and measures the
   2023 rows in training, the weather offset, the future-rain "lag 12" channel and the
   week-395 error.
4. **Architectures** — STGAT, A3TGCN, ASTGCN, AAGCN and DCRNN in plain PyTorch (dense graph
   operators, parameter names matching PyTorch Geometric Temporal), plus the SEIR-LSTM's LSTM.
   `tests/test_kaggle_architectures.py` loads reference weights into each and checks the
   outputs match the reference implementations.
5. **Heads and physics** — direct, residual, gated, SEIR decoder, gated SEIR, metapopulation
   SEIR; the daily SEIR simulator; the NB likelihood; the spatial penalty.
6. **SEIR-decoder diagnosis** — inverts the simulator for the required force of infection.
7. **Training** — 69 configurations × 3 origins × 3 seeds plus 7 × 9 origins × 3 seeds,
   with baselines (k-NN, gradient boosting) and augmentation (TimeGAN, SEIR-simulated).
   Runs resume from `runs.jsonl` if a session is interrupted.
8. **Analysis** — every paired test (sign-flip permutation, BH-adjusted) in the paper,
   computed from the runs just trained.
9. **Outputs** — CSVs, `results.json`, figures and a summary.

---

## The reported run (EXP-050)

`outputs/kaggle_run/` holds the downloaded outputs of the run the paper reports: Kaggle
CPU, 810 runs, 3.01 h, notebook at commit `180ffae`. `predictions.pkl` (88 MB, every
prediction) is git-ignored; rerun the notebook to regenerate it.
`scripts/build_paper_assets.py` builds the paper's tables and figures from these files
alone.

RMSE in weekly cases on the corrected data, three frozen origins × three seeds, pairs on
(origin, seed):

| | validation | test |
|---|---|---|
| persistence | 17.86 | 36.02 |
| direct head: AAGCN / ASTGCN / LSTM | 16.71 / 16.83 / 16.91 | |
| direct head: STGAT / A3TGCN / DCRNN | 22.93 / 28.63 / 34.64 | |
| SEIR decoder, all six encoders | 24.01–25.75 | |
| gated SEIR head, all six encoders | 17.43–17.69 | 35.50–35.84 |
| best model B (AAGCN, direct, NB, seasonal) | 15.51 | 37.54 |

- **Physics does not earn the repair.** The gated SEIR head repairs STGAT, A3TGCN and DCRNN
  (−5.30, −10.95, −16.97, 9/9 each), but the same head without the simulator carries
  96–112% of it. The simulator is significantly worse on STGAT (+0.66, 1/9, p_adj 0.023)
  and no better on the other two. Credited to physics: none.
- **NB likelihood** is the largest controlled gain: −0.76, 9/9, p_adj 0.014.
- **SEIR-GNN vs SEIR-LSTM** wins on three origins on validation (ASTGCN −0.54, 9/9) but is not
  robust on nine disjoint origins (ASTGCN −0.62 on validation, +0.07 on test; AAGCN +0.18
  / −0.30).
- **Ceiling:** the working models' residuals correlate 0.91–0.97 with B's; TimeGAN's largest
  week is 2,036 against a real 2,631; Moran's I of weekly log growth is −0.03.
- **Validation and test disagree:** in 5 of 8 families validation's pick is worse than
  persistence on test.

Full tables: `outputs/kaggle_run/*.csv`; the log entry is `docs/EXPERIMENT_LOG.md` EXP-050.

**Not in the notebook:** the reproduction of Weng et al.'s published numbers (paper §3.1) runs
the authors' own code in its pinned environment and lives in `reproduction/`. The earlier
spatial-penalty result on the benchmark array (−0.044 on STGAT) is cited from the Phase-2
study, not re-run.

---

## Rebuild the paper

```bash
python scripts/build_kaggle_notebook.py     # regenerate the notebook from kaggle/src/*.py
python scripts/build_paper_assets.py        # tables_generated.tex + fig_*.pdf/png from outputs/kaggle_run/
cd full_paper/overleaf && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Or upload `overleaf.zip` to Overleaf (pdfLaTeX, TeX Live 2023 or later). Edit
`kaggle/src/*.py`, never the `.ipynb`.

## References

Full-text PDFs of the cited papers are in `references/`, indexed in `references/README.md`.
