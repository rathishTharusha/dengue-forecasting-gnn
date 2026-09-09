"""Cell sources for 04_kaggle_search.ipynb.

Generated into a notebook by gen_notebooks.py so the source stays diffable --
editing the .ipynb directly is how notebook conflicts become unresolvable (D6).

The notebook runs the Stage-3 hyperparameter search on Kaggle. It is deliberately
thin: all modelling code comes from the repository's `src/dengue_gnn` package, so
Kaggle and local runs execute the *same* code and produce comparable rows. A
notebook that reimplemented the model would be a second implementation to keep in
sync, which is finding F1 waiting to happen.
"""

from gen_notebooks import code as _code
from gen_notebooks import md as _md

CELLS = []


def md(text):
    CELLS.append(_md(text))


def code(text):
    CELLS.append(_code(text))


md(r"""
# Stage-3 search — Kaggle runner

Runs the mechanistic-constraint and hyper-parameter search using the repository's
own `dengue_gnn` package. Nothing is reimplemented here.

**Before running**, attach two things under *Notebook ▸ Add Input*:

1. **Code** — either leave `REPO_URL` set (public repo, cloned at runtime) or
   attach a Kaggle Dataset containing the repo's `src/` directory.
2. **Data** — a Kaggle Dataset containing `sri_lanka_2013-2022_shifted.npy` and
   `sri_lanka_adj_list.json` (both ~1 MB, in `notebooks/baseline/`).

Create the data dataset once with:

```bash
kaggle datasets create -p kaggle/data
```

## What this searches, and why

The Phase-3 results are **under-powered, not under-searched**. Measured on our own
data, the paired standard deviations are:

| Comparison | mean Δ RMSE | SD of Δ | n now | n for 80% power |
|---|---|---|---|---|
| adaptive graph vs its control | −4.59 | 12.01 | 24 | **54** |
| growth-smooth λ=0.3 vs control | −2.05 | 4.82 | 24 | **44** |
| curriculum vs none | −10.15 | 21.21 | 24 | **35** |

Every effect we care about needs roughly **twice** the runs we have. So this search
spends compute on **replication first** (8 seeds instead of 3) and breadth second,
and it selects on validation folds — never on test.
""")

code(r"""
# --- environment -----------------------------------------------------------
import os, sys, subprocess, shutil
from pathlib import Path

REPO_URL = "https://github.com/rathishTharusha/dengue-forecasting-gnn.git"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
SRC = None

# Preferred: an attached Dataset carrying src/. Falls back to cloning.
for cand in Path("/kaggle/input").glob("*/src/dengue_gnn") if Path("/kaggle/input").exists() else []:
    SRC = cand.parent
    break

if SRC is None:
    dest = WORK / "repo"
    if not dest.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(dest)], check=True)
    SRC = dest / "src"

sys.path.insert(0, str(SRC))
import dengue_gnn  # noqa: E402
print("dengue_gnn", dengue_gnn.__version__, "from", SRC)
""")

code(r"""
# --- data ------------------------------------------------------------------
# Kaggle Dataset input, else the cloned repo's copy. Both are the same files.
NPY = NPYADJ = None
for base in list(Path("/kaggle/input").glob("*")) if Path("/kaggle/input").exists() else []:
    hits = list(base.rglob("sri_lanka_2013-2022_shifted.npy"))
    if hits:
        NPY = hits[0]
        NPYADJ = next(base.rglob("sri_lanka_adj_list.json"))
        break
if NPY is None:
    NPY = SRC.parent / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
    NPYADJ = SRC.parent / "notebooks" / "baseline" / "sri_lanka_adj_list.json"

from dengue_gnn.experiment import load_dataset
raw, adj, districts = load_dataset(NPY, NPYADJ)
print("data", raw.shape, "|", len(districts), "districts")
""")

code(r"""
# --- compute budget --------------------------------------------------------
# Kaggle gives 4 CPU cores. This workload is overhead-bound, not compute-bound:
# measured locally, ONE torch thread is ~2.7x faster per epoch than ten, because
# the model is ~7k parameters on a (1, 25, 33) input. So a GPU does not help and
# the pool -- not intra-op threading -- provides the parallelism.
import os, torch, multiprocessing as mp
torch.set_num_threads(1)
WORKERS = max(1, (os.cpu_count() or 4))
print("cores:", os.cpu_count(), "| workers:", WORKERS, "| cuda:", torch.cuda.is_available())
""")

md(r"""
## Search space

Two stages, in this order and for this reason:

**Stage A — replication.** The configurations we already believe in, at 8 seeds
instead of 3, to get n=64 and actually resolve the effects the power analysis says
we cannot currently see.

**Stage B — breadth.** A λ grid centred *below* 0.3, because the Stage-3 sweep
showed λ=0.3 helps and λ=1.5 hurts, so the optimum is at or under the smallest
value tried.

Selection happens on validation RMSE, which `train_fold` already uses for early
stopping. Test rows are recorded but never used to choose anything — choosing on
test is how a search of this size manufactures a false positive.
""")

code(r"""
from dengue_gnn.experiment import Config

ORIGINS = (0.400, 0.475, 0.550, 0.625, 0.700, 0.775, 0.850, 0.925)
TEST_FRAC = 0.075
SEEDS = tuple(range(8))          # n = 8 folds x 8 seeds = 64 per config

common = dict(origins=ORIGINS, test_frac=TEST_FRAC, seeds=SEEDS)

STAGE_A = [
    Config(label="A_dense_fixed",  use_adaptive=False, **common),
    Config(label="A_adaptive",     use_adaptive=True,  **common),
    Config(label="A_mech0.3",      lambda_mech=0.3, mech_mode="smooth", **common),
    Config(label="A_mech0.3_curr", lambda_mech=0.3, mech_mode="smooth",
           curriculum=0.5, **common),
]

STAGE_B = [
    Config(label=f"B_mech{lam}", lambda_mech=lam, mech_mode="smooth", **common)
    for lam in (0.05, 0.10, 0.15, 0.20, 0.45, 0.60)
]

CONFIGS = STAGE_A + STAGE_B
print(f"{len(CONFIGS)} configs x {len(ORIGINS)} folds x {len(SEEDS)} seeds "
      f"= {len(CONFIGS)*len(ORIGINS)*len(SEEDS)} runs")
""")

code(r"""
# --- run -------------------------------------------------------------------
import time, csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dengue_gnn.experiment import run_single, persistence_rows

def job(payload):
    import torch
    torch.set_num_threads(1)
    cfg, fold_idx, seed, kind = payload
    if kind == "persistence":
        return persistence_rows(raw, cfg, fold_idx)
    return run_single(raw, adj, cfg, fold_idx, seed)

jobs = [(c, f, s, "model")
        for c in CONFIGS for f in range(len(ORIGINS)) for s in c.seeds]
jobs += [(CONFIGS[0], f, 0, "persistence") for f in range(len(ORIGINS))]

rows, t0, done = [], time.time(), 0
with ProcessPoolExecutor(max_workers=WORKERS) as pool:
    futs = {pool.submit(job, j): j for j in jobs}
    for fut in as_completed(futs):
        rows.extend(fut.result())
        done += 1
        if done % 25 == 0 or done == len(jobs):
            print(f"  {done}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

out = WORK / "stage3_search.csv"
rows.sort(key=lambda r: (r["label"], r["fold"], r["seed"], r["horizon"]))
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\nwrote {len(rows)} rows to {out} in {time.time()-t0:.0f}s")
""")

md(r"""
## Result

`stage3_search.csv` appears under *Output*. Retrieve it with:

```bash
kaggle kernels output <user>/dengue-stage3-search -p results/
```

Then merge and tabulate locally — the same generator the paper uses, so the
numbers cannot drift:

```bash
python scripts/make_tables.py --runs results/stage3_search.csv --prefix search
```

**Do not read conclusions off this notebook's output pane.** Rule D2: a number in a
Kaggle output pane does not exist until it is in a committed CSV with an
`EXPERIMENT_LOG.md` entry beside it.
""")
