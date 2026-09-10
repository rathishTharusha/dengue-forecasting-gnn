# CS3631 Phase 2 Short Paper Submission Package

**Paper Title**: *Know When the Epidemic Comes: Mathematics Before Data in Dengue Outbreak Forecasting*  
**Authors**: Group 05 (CS3631)  
**Target Venue / Format**: ACM `sigconf` format (Strict 4-page body limit excluding references)  

---

## Directory Structure

```
short_paper/
├── README.md                    # Package documentation and instructions
├── reproduce_short_paper.ipynb  # Master interactive reproduction notebook
├── overleaf.zip                 # Ready-to-upload complete Overleaf zip bundle
├── overleaf/                    # Full unzipped LaTeX source & figures
│   ├── main.tex                 # Primary submission document
│   ├── main-highlighted.tex     # Color-coded contribution build (for grading)
│   ├── refs.bib                 # BibTeX bibliography database
│   ├── ACM-Reference-Format.bst # ACM bibliography style
│   ├── tables_generated.tex     # Generated tables (Table 1)
│   ├── sections/                # Modular paper sections (00 to 06)
│   └── figures/                 # Vector PDF and PNG publication figures
└── references/                  # Full-text PDFs of all cited reference papers
    ├── Weng_2024_Graph_Representation_Learning_Dengue.pdf
    ├── GulMohamed_2026_DengueGNN.pdf
    ├── Phaijoo_Gurung_2018_Sensitivity_Analysis_SEIR_SEI.pdf
    ├── Hasan_2022_SEIR_SEI_Dengue.pdf
    ├── Tissera_2020_Severe_Dengue_Sri_Lanka_2017.pdf
    ├── Raissi_2019_Physics_Informed_Neural_Networks.pdf
    ├── Nguyen_2024_MP_PINN_Epidemic_Forecasting.pdf
    ├── Wu_2019_Graph_WaveNet.pdf
    ├── Li_2018_DCRNN_Traffic_Forecasting.pdf
    ├── Guo_2019_ASTGCN.pdf
    ├── Bai_2021_A3TGCN.pdf
    ├── Shi_2019_AAGCN.pdf
    └── README.md                # Literature index and citation rationale
```

---

## Quick Start: Overleaf Submission

1. **Upload Project**: In Overleaf, click **New Project** $\to$ **Upload Project** and select `overleaf.zip`.
2. **Compiler Settings**:
   - Compiler: **pdfLaTeX** (default)
   - TeX Live version: **2023 or later**
3. **The Two Builds**:
   - Clean Submission: Set main document to `main.tex`.
   - Contribution-Highlighted Build: Set main document to `main-highlighted.tex` (colors contributions per student).
4. **Page Count Compliance**:
   - The body ends cleanly on **page 4**, with the bibliography starting on **page 5** (strictly compliant with the 4-page body limit).

---

## Quick Start: Reproducibility Notebook

The master notebook `reproduce_short_paper.ipynb` reproduces every empirical number and figure in the paper:
1. Open `reproduce_short_paper.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
2. Select the Python environment containing `torch`, `scipy`, `pandas`, and `matplotlib`.
3. Run **Run All Cells** top-to-bottom.
4. Execution time is under 30 seconds.
5. All figures are verified and automatically updated in `short_paper/overleaf/figures/`.

---

## Scientific Rigor & Citation Statement

Every statement in the short paper is strictly grounded:
1. **Empirically Proven**: All performance figures, benchmark gaps, artifact isolation (Week 395), $R_t$ unpredictability ($r^2=0.263$), and Dirichlet spatial regularization ($-0.044$ RMSE, $p<0.0001$) are directly verified and generated from our reproducible experiments.
2. **Mathematical Truths**: Identified mechanisms (e.g., Jensen's inequality for per-window averaging, lag-1 autoregression floor).
3. **Primary Literature Backing**: All external domain facts (2017 epidemic figures, SEIR-SEI stage durations, PINN methods, baseline architectures) are accompanied by their peer-reviewed citations and archived as PDFs in `short_paper/references/`.
