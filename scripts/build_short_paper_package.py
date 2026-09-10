"""Assemble and package the complete CS3631 Phase 2 short paper submission package.

Builds:
- short_paper/
  ├── README.md
  ├── reproduce_short_paper.ipynb
  ├── overleaf/
  │   ├── main.tex
  │   ├── main-highlighted.tex
  │   ├── refs.bib
  │   ├── ACM-Reference-Format.bst
  │   ├── tables_generated.tex
  │   ├── sections/*.tex
  │   └── figures/*.pdf
  ├── overleaf.zip
  └── references/*.pdf
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper"
SHORT_PAPER = REPO / "short_paper"
OVERLEAF_DIR = SHORT_PAPER / "overleaf"
OVERLEAF_ZIP = SHORT_PAPER / "overleaf.zip"

OVERLEAF_FILES = [
    "main.tex",
    "refs.bib",
    "ACM-Reference-Format.bst",
    "tables_generated.tex",
    "sections/00_abstract.tex",
    "sections/01_introduction.tex",
    "sections/02_related_work.tex",
    "sections/03_framework.tex",
    "sections/04_experimental_setup.tex",
    "sections/05_results.tex",
    "sections/06_discussion_conclusion.tex",
]

HIGHLIGHTED_STUB = r"""% Overleaf: set this as the main document (Menu -> Main document) to build the
% contribution-highlighted version. It defines one flag and defers entirely to
% main.tex -- there is only ever one copy of the text.
\def\HIGHLIGHTON{1}
\input{main.tex}
"""

PACKAGE_README = """# CS3631 Phase 2 Short Paper Submission Package

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

1. **Upload Project**: In Overleaf, click **New Project** $\\to$ **Upload Project** and select `overleaf.zip`.
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
"""


def main() -> int:
    print("=== Building Short Paper Package ===")

    # 1. Regenerate paper figures and tables
    print("Step 1: Regenerating paper figures and tables...")
    subprocess.check_call([sys.executable, str(REPO / "paper" / "_build" / "figures.py")])

    # 2. Build reproducibility notebook
    print("Step 2: Building reproduce_short_paper.ipynb...")
    subprocess.check_call([sys.executable, str(REPO / "scripts" / "build_short_paper_notebook.py")])

    # 3. Assemble Overleaf folder
    print("Step 3: Assembling short_paper/overleaf/...")
    if OVERLEAF_DIR.exists():
        shutil.rmtree(OVERLEAF_DIR)
    OVERLEAF_DIR.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for rel in OVERLEAF_FILES:
        src = PAPER / rel
        dst = OVERLEAF_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)

    # Copy all generated figures (both PDF and PNG)
    fig_dir = PAPER / "figures"
    if fig_dir.is_dir():
        for fig_file in sorted(list(fig_dir.glob("*.pdf")) + list(fig_dir.glob("*.png"))):
            rel = f"figures/{fig_file.name}"
            dst = OVERLEAF_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fig_file, dst)
            copied.append(rel)

    (OVERLEAF_DIR / "main-highlighted.tex").write_text(HIGHLIGHTED_STUB, encoding="utf-8")
    copied.append("main-highlighted.tex")

    # 4. Package overleaf.zip
    print("Step 4: Packaging overleaf.zip...")
    if OVERLEAF_ZIP.exists():
        OVERLEAF_ZIP.unlink()

    with zipfile.ZipFile(OVERLEAF_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OVERLEAF_DIR.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OVERLEAF_DIR))

    print(f"Wrote {OVERLEAF_ZIP.relative_to(REPO)} ({OVERLEAF_ZIP.stat().st_size / 1024:.1f} KB)")

    # 5. Write package README.md
    print("Step 5: Writing package README.md...")
    (SHORT_PAPER / "README.md").write_text(PACKAGE_README, encoding="utf-8")

    # 6. Also mirror to root overleaf/ and overleaf.zip for convenience
    root_overleaf = REPO / "overleaf"
    if root_overleaf.exists():
        shutil.rmtree(root_overleaf)
    shutil.copytree(OVERLEAF_DIR, root_overleaf)
    shutil.copy2(OVERLEAF_ZIP, REPO / "overleaf.zip")

    # 7. Create Windows Directory Junction for 'short paper' if possible
    space_dir = REPO / "short paper"
    if not space_dir.exists():
        try:
            subprocess.run(["cmd", "/c", "mklink", "/J", str(space_dir), str(SHORT_PAPER)], check=False)
            print(f"Created directory junction: {space_dir.name} -> {SHORT_PAPER.name}")
        except Exception:
            pass

    print("\nPackage Assembly Complete:")
    print(f"  Folder:    {SHORT_PAPER}")
    print(f"  Overleaf:  {OVERLEAF_DIR} ({len(copied)} files)")
    print(f"  Zip:       {OVERLEAF_ZIP}")
    print(f"  Notebook:  {SHORT_PAPER / 'reproduce_short_paper.ipynb'}")
    print(f"  References:{SHORT_PAPER / 'references'} ({len(list((SHORT_PAPER / 'references').glob('*.pdf')))} PDFs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
