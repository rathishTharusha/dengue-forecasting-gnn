# %% [markdown]
# # Where does physics help a graph network? — the whole study, from the original sources
#
# District-level weekly dengue forecasting for Sri Lanka: 25 districts, a 3-week input
# window, a 3-week horizon. This notebook is the complete, self-contained reproduction
# of the study reported in the paper. It:
#
# 1. **verifies** every input file against the SHA-256 recorded when it was retrieved
#    from its original source (section 1);
# 2. **rebuilds** the weekly case series from the Epidemiology Unit's report table, the
#    weekly climate from ERA5 daily data, and the population from the Department of
#    Census and Statistics PDFs (section 2);
# 3. **audits** the benchmark array prior work used, to show why it is not used (section 3);
# 4. **defines** the five published spatio-temporal GNNs and the SEIR-LSTM's LSTM in plain
#    PyTorch, the SEIR simulator, and every forecast head (sections 4-5);
# 5. **trains** every configuration the paper reports under one leakage-controlled
#    protocol (sections 6-7); and
# 6. **analyses** only the rows it has just produced (section 8).
#
# Nothing is loaded from an earlier run: every number printed below is computed here.
#
# ## Running it on Kaggle
#
# 1. Upload `full_paper/kaggle/dataset/` as a Kaggle dataset (it is built by
#    `full_paper/kaggle/fetch_sources.py` and contains only original source files).
# 2. Create a notebook from this file, attach the dataset, choose a **CPU** accelerator
#    (the models are small and the work is many short runs; 4 CPU workers beat a GPU
#    here) and "Save Version → Save & Run All".
# 3. Everything the notebook produces is written to `/kaggle/working/outputs/`.
#
# Internet access is not needed: the one extra package (PyMuPDF, to read the PDFs) is
# installed from a wheel inside the dataset.

# %%
# ----------------------------------------------------------------- run settings --
#: "full" runs every configuration the paper reports (several hours on 4 CPUs).
#: "quick" is a smoke test -- one origin, one seed, 3 epochs -- whose numbers are
#: deliberately degraded and must never be quoted.
PROFILE = "full"
#: Parallel worker processes. Kaggle's CPU sessions have 4 cores.
WORKERS = 4

import os
import sys
import time

import matplotlib

if not hasattr(sys, "ps1") and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

# Parallel training forks worker processes; forking after PyTorch has started a
# multi-threaded pool can deadlock, and every run is single-threaded anyway.
torch.set_num_threads(1)

NOTEBOOK_START = time.time()
QUICK = PROFILE == "quick"
if QUICK:
    print("PROFILE = 'quick': a smoke test. These numbers are NOT results and must not be quoted.")

# %% [markdown]
# ## 1. The input files, verified
#
# The dataset holds only files exactly as retrieved: the parsed weekly report table,
# one report PDF, the Open-Meteo ERA5 responses, the Census and population PDFs, and
# the district graph, boundaries and benchmark array from the benchmark authors'
# repository. `SOURCES.csv` lists each file's origin URL and hash; the cell below
# recomputes every hash and stops if any file differs.

# %%
import csv
import hashlib
import subprocess
from pathlib import Path


def _find_sources() -> Path:
    """The attached dataset: wherever Kaggle mounts it under /kaggle/input, or a local copy."""
    candidates = [Path("/kaggle/input"), Path.cwd() / "dataset", Path.cwd().parent / "dataset",
                  Path.cwd() / "full_paper" / "kaggle" / "dataset"]
    for base in candidates:
        if base.exists():
            hit = next(iter(sorted(base.rglob("SOURCES.csv"))), None)
            if hit is not None:
                return hit.parent
    raise FileNotFoundError("attach the dataset built by full_paper/kaggle/fetch_sources.py")


SRC = _find_sources()
OUT = Path("/kaggle/working/outputs") if Path("/kaggle/working").exists() else Path.cwd() / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

with (SRC / "SOURCES.csv").open(encoding="utf-8") as f:
    SOURCES = list(csv.DictReader(f))
bad = [r["file"] for r in SOURCES
       if hashlib.sha256((SRC / r["file"]).read_bytes()).hexdigest() != r["sha256"]]
if bad:
    raise SystemExit(f"these files do not match their recorded SHA-256: {bad}")
print(f"{len(SOURCES)} source files verified (SHA-256) in {SRC}")
for r in SOURCES:
    if r["file"].startswith(("population/census2012/", "climate/")):
        continue
    print(f"  {r['file']:58s} {int(r['bytes']):>10,d} B  <- {r['source'][:80]}")
print(f"  + {sum(r['file'].startswith('population/census2012/') for r in SOURCES)} Census 2012 district PDFs"
      f" and {sum(r['file'].startswith('climate/') for r in SOURCES)} ERA5 district responses")
print(f"  files whose hash equals the one recorded at first retrieval: "
      f"{sum(r['manifest_match'] == 'yes' for r in SOURCES)}")

try:
    import pymupdf  # noqa: F401
except ImportError:
    wheel = next((SRC / "wheels").glob("*.whl"))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", str(wheel)], check=True)
    import pymupdf  # noqa: F401
print("PyMuPDF", pymupdf.__version__)
