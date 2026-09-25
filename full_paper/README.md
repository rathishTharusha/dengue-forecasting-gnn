# Full Paper & Complete Reproducibility Package

**Paper Title**: *Where Does Physics Help a Graph Network? A Leakage-Controlled Study of SEIR-Informed Spatio-Temporal GNNs for Dengue Forecasting*
(rewritten 2026-09-24; the previous title, *Know When the Epidemic Comes*, was the Phase-2 short paper on the benchmark array, which is kept in `paper/`)  
**Authors**: Group 05 (CS3631)  
**Target Format**: IEEE / ACM Conference & Journal Full Paper Format  

---

## 📦 Standalone Package Overview

This directory (`full_paper/`) is a **completely self-contained, standalone distribution package**. Anyone can download only this directory and execute the entire experimental pipeline, reproduce all paper tables/figures, inspect the full-text reference papers, and compile the manuscript in LaTeX without relying on any external path or repository root setup.

---

## 📂 Directory Structure

```
full_paper/
├── README.md                      # Main package documentation & execution instructions
├── reproduce_full_paper.ipynb     # Master end-to-end interactive reproduction notebook
├── overleaf.zip                   # Complete, ready-to-upload Overleaf LaTeX source ZIP
├── overleaf/                      # Unzipped LaTeX source code, section files, & figures
│   ├── main.tex                   # Primary full paper LaTeX manuscript
│   ├── main-highlighted.tex       # Color-coded contribution build (for review/grading)
│   ├── refs.bib                   # BibTeX bibliography database
│   ├── tables_generated.tex       # LaTeX tables for the manuscript (renewal/physics), not the SEIR-GNN stages
│   ├── sections/                  # Modular paper sections (00_abstract to 06_discussion)
│   └── figures/                   # High-resolution vector PDF & raster PNG figures
├── data/                          # Standalone dataset directory
│   ├── dengue_cases_raw.csv       # Raw weekly 25-district epidemiological case series (2013-2024)
│   ├── era5_weekly_by_district.csv# ERA5 climate reanalysis features (temp, rainfall, humidity)
│   ├── modis_ndvi_weekly_by_district.csv # MODIS satellite vegetation index (NDVI)
│   ├── district_census_2012.csv   # Sri Lanka Census 2012 district population metadata
│   ├── seroprevalence_nine_districts.csv # 9-District empirical field survey dataset
│   └── seir_parameters.json       # Calibrated SEIR-SEI compartmental parameters
└── references/                    # Full-text PDFs of all 12 peer-reviewed primary papers
    ├── Bai_2021_A3TGCN.pdf
    ├── GulMohamed_2026_DengueGNN.pdf
    ├── Guo_2019_ASTGCN.pdf
    ├── Hasan_2022_SEIR_SEI_Dengue.pdf
    ├── Li_2018_DCRNN_Traffic_Forecasting.pdf
    ├── Nguyen_2024_MP_PINN_Epidemic_Forecasting.pdf
    ├── Phaijoo_Gurung_2018_Sensitivity_Analysis_SEIR_SEI.pdf
    ├── Raissi_2019_Physics_Informed_Neural_Networks.pdf
    ├── Shi_2019_AAGCN.pdf
    ├── Tissera_2020_Severe_Dengue_Sri_Lanka_2017.pdf
    ├── Weng_2024_Graph_Representation_Learning_Dengue.pdf
    ├── Wu_2019_Graph_WaveNet.pdf
    └── README.md                  # Literature index and citation mapping
```

---

## 🚀 Quick Start: LaTeX Compilation & Overleaf

1. **Upload to Overleaf**:
   - Log in to Overleaf $\to$ Click **New Project** $\to$ **Upload Project**.
   - Select `full_paper/overleaf.zip`.
2. **Compiler Configuration**:
   - Compiler: **pdfLaTeX** (or XeLaTeX)
   - TeX Live Version: **2023 or later**
3. **Manuscript Variants**:
   - **Standard Full Paper**: Compile `main.tex`.
   - **Contribution-Highlighted Variant**: Compile `main-highlighted.tex` (uses color coding for author contributions).

---

## 🧪 Quick Start: the reproduction notebook

`reproduce_full_paper.ipynb` reproduces **every finding in the paper and the experiment
log (EXP-032 … EXP-049)** in one run, in the order of the paper: the corrected data and
the persistence floor recomputed from the raw series, the benchmark-array audit, the six
encoders × SEIR heads, the SEIR-decoder diagnosis, the anchor/gate/physics control,
every lever against its own control, the best model against persistence, SEIR-GNN
against SEIR-LSTM on three and nine origins, the physics-structure arms, the
informational ceiling, the validation/test disagreement across every grid, and the audit
of the earlier "positive physics" runs. Every paired test is recomputed in the notebook
from the per-run rows in `outputs/results/`.

```bash
pip install numpy pandas matplotlib jupyter
jupyter notebook reproduce_full_paper.ipynb      # Run All: under a minute
```

- **Read without running:** `reproduce_full_paper.html` is the executed notebook.
- **Re-train instead of reading committed rows:** set `RERUN = True` and list grids in
  `RERUN_GRIDS` in the first code cell, and run it from inside the repository
  (hours on CPU; section 14 of the notebook lists the cost per grid).
- **Regenerate the notebook and its inputs:** `python scripts/build_full_paper_notebook.py --execute`
  from the repository root. It is generated; do not edit the `.ipynb` by hand.

---

## 📊 SEIR-GNN results

> **These replace the Stage S4–S9 table that stood here until 2026-09-23.** That table
> was withdrawn in full: every number in it traced to `analysis/_build/run_s5_seir_gnn.py`,
> which took one gradient step per epoch, minimised SMAPE while being scored by RMSE, held
> the force of infection constant across the horizon, and collapsed every covariate to a
> scalar through `nn.Linear(in_dim, 1)` before the graph saw it. The S9 row was worse than
> mistuned: `run_s9_confirmatory.py` is described as a nine-origin paired permutation test,
> but runs on the three origins S5 produced, subtracts hardcoded scalars instead of matching
> on origin, and reported an early-warning p-value (`p = 0.04`) that was a typed-in literal
> rather than a computed quantity. See `docs/EXPERIMENT_LOG.md` EXP-032 and EXP-037.
>
> **The manuscript in `overleaf/` was rewritten on 2026-09-24** around the corrected re-run.
> It now reports the array audit, the six-encoder x head comparison, every lever tried, and
> the nine-origin confirmations. Every number in its tables and figures is generated from
> `seirgnn2/results/*.json` by `scripts/build_paper_assets.py`, and the result files it reads
> are copied to `outputs/results/`.

The replacement is a leakage-controlled re-run (`seirgnn2/`) on **nine origins with disjoint
test spans**, five arms frozen before the run, three seeds each, selection on validation only.
Differences are paired at the origin unit, which is the only unit resampled from the data.

| comparison (validation RMSE) | Δ | origins won | p | p_adj |
|---|---|---|---|---|
| SEIR-GNN (ASTGCN+FOI) vs **SEIR-LSTM** | **−0.75** | **8/9** | **0.012** | 0.059 |
| SEIR-GNN vs **persistence** | −2.92 | 8/9 | 0.066 | 0.166 |
| best published GNN (AAGCN direct) vs persistence | −2.89 | 8/9 | 0.035 | 0.166 |
| SEIR-GNN vs best published GNN | −0.03 | 1/9 | 1.000 | 1.000 |
| SEIR-GNN vs SEIR-LSTM, **test** RMSE | +0.23 | 4/9 | 0.410 | 0.513 |

**What this supports, stated plainly:**

- **Against SEIR-LSTM** — on validation the proposed method wins 8 of 9 origins (raw p = 0.012,
  p_adj = 0.059, missing the pre-registered bar); on test it loses 4/9 (+0.23). A second
  nine-origin run with the AAGCN encoder (EXP-047) reverses the pattern: a tie on validation
  (−0.01) and a win on test (−0.41, 6/9, p_adj 0.125). **The advantage is not robust across
  encoders, metrics or runs; do not report it as a win.**
- **Against the naive persistence floor** — a small consistent edge on validation that does
  not survive correction, and nothing at all on test (31.70 vs 31.76). **Persistence is not
  beaten.**
- **Against the five published architectures** — comfortably better than STGAT, A3TGCN and
  DCRNN; **not** better than AAGCN or ASTGCN on the direct head. Behind the gated SEIR head
  all six encoders land within 0.25 of each other (EXP-034). **EXP-048 shows that repair is the
  persistence anchor, not the SEIR physics**: the same head without the simulator carries
  96–112% of it, and the simulator is significantly worse on STGAT (`docs/RESCUE_PLAN.md`).
- **Mean RMSE across these nine origins is not a usable summary.** Origin 0.40 is an outlier
  where every arm fails (140–227 RMSE against 13–48 elsewhere) and it dominates every mean.
  Read win counts and per-origin values; `seirgnn2/stats.py` prints wins beside every delta
  for this reason.

**Withdrawn without replacement:** the S7 early-warning significance claim. The AUC figures
themselves (0.807 → 0.826, from `analysis/_build/outbreak_signal.py`) are computed and stand;
the `p = 0.04` attached to them never was. S6 is not re-run here and is unverified under
the corrected harness. **S8 (EXP-030) is invalid**: `run_s8_seroprevalence.py` compares against
hardcoded "example" survey values that do not match the transcribed survey in
`data/external/seroprevalence_nine_districts.csv`.

Reproduce with:

```bash
python seirgnn2/sweep.py confirm --workers 6 --epochs 400
python seirgnn2/stats.py confirm --ref "LSTM+foi_res" --metric val_RMSE
```

Verified identical on Kaggle (144 rows both sides, every arm within 0.11 RMSE) via
`python scripts/build_seirgnn2_kernel.py --grid confirm`. See `seirgnn2/README.md` for the
harness, the protocol, and a table of every lever tried with its measured effect.

## 📚 Literature & Citation Integrity

All primary references cited in the full paper are archived as full-text PDFs in `full_paper/references/`:
* **Epidemiology & Physics-Informed Modeling**: Phaijoo & Gurung (2018), Hasan et al. (2022), Raissi et al. (2019), Nguyen et al. (2024).
* **Baseline Spatial GNNs**: STGAT (Huang et al.), ASTGCN (Guo et al., 2019), AAGCN (Shi et al., 2019), A3TGCN (Bai et al., 2021), DCRNN (Li et al., 2018), DengueGNN (GulMohamed et al., 2026).
* **Empirical Field Data**: Tissera et al. (2020) for Sri Lanka 2017 Dengue Outbreak data.
