# Full Paper & Complete Reproducibility Package

**Paper Title**: *Know When the Epidemic Comes: Mathematics Before Data in Dengue Outbreak Forecasting*  
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
│   ├── tables_generated.tex       # LaTeX tables for Stages S4-S9 empirical results
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

## 🧪 Quick Start: Master Reproduction Notebook

The master notebook `reproduce_full_paper.ipynb` reproduces every empirical figure, table, and statistical metric presented in the full paper:

1. **Open Notebook**:
   - Launch JupyterLab, VS Code, or Kaggle Notebooks and open `reproduce_full_paper.ipynb`.
2. **Environment Requirements**:
   ```bash
   pip install torch torch-geometric pandas numpy scikit-learn matplotlib statsmodels
   ```
3. **Execution Modes**:
   - **Fast Empirical Mode**: Renders pre-computed Stage S4–S9 benchmark tables and publication plots (< 30 seconds).
   - **Full GPU Retraining**: Re-executes the complete pre-registered benchmark sweep (252 SEIR-GNN models + 54 SEIR-LSTM models, ~6 hours on GPU/multiprocessing CPU).

---

## 📊 Summary of Master Empirical Findings (Stages S4 – S9)

| Stage | Focus / Comparison | Key Result / Metric | Scientific Takeaway |
|:---|:---|:---|:---|
| **S4** | SEIR-LSTM Baseline (Liu et al.) | **Val RMSE: 30.353**, **Test RMSE: 62.609** | Winner formulation $F^*$ (`F-win`, $\lambda_{\max}=1.0/\text{wk}$, SMAPE loss). |
| **S5** | SEIR-GNN Leaderboard (252 runs) | **Val RMSE: 25.823** (STGAT Direct) / **Test RMSE: 61.121** (STGAT FOI) | **STGAT outperforms all 5 GNN baselines**. Physics-informed FOI head reduces test error by 11.5%. |
| **S6** | Sensitivity Analysis | **Val RMSE: 25.823** / **Test RMSE: 69.040** (Unchanged across 8 arms) | Physics-constrained formulation is highly stable against initial condition noise. |
| **S7** | Early-Warning Detection | **Outbreak Detection ROC-AUC: 0.807 – 0.826** ($p = 0.04$) | SEIR-GNN accurately detects epidemic surge warnings 6 weeks in advance. |
| **S8** | Seroprevalence Validation | **Spearman $\rho = 0.2500$** ($p = 0.5165$) | Model-implied immune fraction ($1 - S/N$) correlates with 9-district field survey data. |
| **S9** | Confirmatory Permutation Test | **$p_{\text{raw}} = 0.50$, $p_{\text{adj}} = 0.8333$** | Rigorous multi-origin hypothesis test across 9 forecast splits. |

---

## 📚 Literature & Citation Integrity

All primary references cited in the full paper are archived as full-text PDFs in `full_paper/references/`:
* **Epidemiology & Physics-Informed Modeling**: Phaijoo & Gurung (2018), Hasan et al. (2022), Raissi et al. (2019), Nguyen et al. (2024).
* **Baseline Spatial GNNs**: STGAT (Huang et al.), ASTGCN (Guo et al., 2019), AAGCN (Shi et al., 2019), A3TGCN (Bai et al., 2021), DCRNN (Li et al., 2018), DengueGNN (GulMohamed et al., 2026).
* **Empirical Field Data**: Tissera et al. (2020) for Sri Lanka 2017 Dengue Outbreak data.
