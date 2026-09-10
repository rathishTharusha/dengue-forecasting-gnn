# What can actually be reproduced exactly

Every claim on this page is sourced to the paper, the authors' repository, or a
command whose output is shown. Nothing is inferred.

**Sources**

| Short name | Paper | Code | Data |
|---|---|---|---|
| Weng 2024 | *Graph Representation Learning for Dengue Forecasting*, IEEE BigData 2024 | [MLOpenSourceOpenScience/disease_modeling_MLOS2](https://github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2) @ `45f1c08` | in-repo |
| DengueGNN | *DengueGNN*, Scientific Reports 16:10584 (2026) | none | OpenDengue, subset unspecified |
| SEIR | Gopalakrishnan, *The SEIR model of infectious diseases*, MTH 271 (2020) | printed in the source | none needed |
| SEIR–SEI | Phaijoo & Gurung, GAMS J. Math. Math. Biosci. 6(a) (2018) | none | none needed |

---

## Weng et al. 2024 — Table I has 20 rows; **5 are exactly reproducible**

Table I is 10 models × 2 datasets (Shifted, Unshifted), each with a Cross
Validated and a Full Dataset column.

### ✅ Reproducible — the five GNNs on the shifted dataset

STGAT, A3TGCN, ASTGCN, DCRNN, AAGCN. Everything needed is released:

- **Code** — `Models/evaluation.py` and `Models/gnn_models.py`.
- **Data** — `Data/Datasets/sri_lanka_2013-2022_shifted.npy` and
  `Models/sri_lanka_adj_list.json`.
- **Invocation** — `README.md`: `cd Models && python .\evaluation.py`, whose
  `__main__` calls each `run_*([0.6, 0.7, 0.8, 0.9, 1.0])`.
- **Expected output** — `Models/results.txt` holds the authors' own raw run,
  matching Table I to the decimal. That is the reproduction target.

These five rows are the paper's actual contribution, so the reproducible subset
is the part that matters.

### ❌ Not reproducible — ARIMA, Random Forest, LSTM (both datasets)

`Models/arima.py`, `random_forest.py` and `lstm.py` all begin:

```python
data = pd.read_csv("../Data/Datasets/MLSO2_Final.csv")
```

**`MLSO2_Final.csv` is not in the repository.** `Data/Datasets/` contains only
`sri_lanka_2013-2022_shifted.npy` and `sri_lanka_2013-2022_vertical.csv`, and a
repo-wide search for `*MLSO2*` returns nothing.

`vertical.csv` is not a substitute: the scripts call
`.drop(["Week", "region", "cases", "minTime", ...])`, and `vertical.csv` has
neither a `Week` nor a `minTime` column — it has `index` and `time`. That call
would raise `KeyError`.

### ❌ Not reproducible — XGBoost, ARNN (both datasets)

Table I reports both. **`Models/` contains no XGBoost or ARNN script.** The paper
states ARNN was run "out-of-the-box using NeuralProphet", but no script,
configuration or hyperparameters are released.

### ❌ Not reproducible — the entire Unshifted Dataset half (10 rows)

`evaluation.py` line 31 hardcodes:

```python
data_file = "../Data/Datasets/sri_lanka_2013-2022_shifted.npy"
```

**No unshifted array is in the repository** — a repo-wide `*.npy` search returns
only the shifted file. The paper describes the shift (precipitation −12 weeks,
min NDVI −17 weeks, mean canopy −12 weeks) but the unshifted source array is not
released, and regenerating it would require the raw NASA products and the
authors' preprocessing run, which is not the same thing as reproducing.

### Summary

| Table I section | Rows | Reproducible |
|---|---|---|
| Shifted — 5 GNNs | 5 | ✅ **yes** |
| Shifted — ARIMA, RF, XGBoost, ARNN, LSTM | 5 | ❌ input file / scripts missing |
| Unshifted — all 10 | 10 | ❌ no unshifted array released |

---

## The authors' `requirements.txt` is broken in two independent ways

```
pandas~=2.2.0
torch==2.1.2
torch_geometric==2.5.3
torch_geometric_temporal==0.54.0
```

### 1. It cannot be installed — pandas

`torch_geometric_temporal==0.54.0` declares `pandas<=1.3.5`, contradicting the
file's own `pandas~=2.2.0`:

```
The conflict is caused by:
    The user requested pandas~=2.2.0
    torch-geometric-temporal 0.54.0 depends on pandas<=1.3.5
ERROR: ResolutionImpossible
```

There is additionally no `pandas<=1.3.5` wheel for Python 3.11 (1.3.5 predates
it), so no Python-3.11 environment satisfies PGT's declared constraint at all.

### 2. Even when forced, it cannot import — torch_geometric

Installing the pinned pair anyway and importing gives:

```
File ".../torch_geometric_temporal/nn/attention/tsagcn.py", line 6, in <module>
    from torch_geometric.utils.to_dense_adj import to_dense_adj
ModuleNotFoundError: No module named 'torch_geometric.utils.to_dense_adj'
```

PyG moved that module in 2.4. **PGT 0.54.0 and PyG 2.5.3 — the two versions the
authors pin beside each other — are mutually incompatible.** Tested here:

| torch_geometric | PGT 0.54.0 imports all five architectures? |
|---|---|
| 2.5.3 (authors' pin) | ❌ `ModuleNotFoundError` |
| **2.4.0** | ✅ |
| 2.3.1 | ✅ |
| 2.3.0 | ✅ |

### 3. Their torch predates the runner's Python

Kaggle's notebook image is **Python 3.12**. `torch==2.1.2` publishes no cp312
wheel — PyPI lists `cp38, cp39, cp310, cp311` only, and torch added 3.12 support
in 2.2.0. A first attempt to install the authors' stack on the stock image failed:

```
SystemExit: command failed: ('/usr/bin/python3', '-m', 'pip', 'install', '-q',
'torch==2.1.2', '--index-url', 'https://download.pytorch.org/whl/cpu')
```

The PyG companion wheels have the same ceiling: `data.pyg.org` offers
`torch_scatter`/`torch_sparse` for torch-2.1.2 at cp38–cp311, never cp312.

### The three forced deviations

Some deviation is unavoidable. These are the minimal ones, and all three are
recorded in every notebook:

1. **A standalone Python 3.11**, fetched with `uv` inside the kernel, instead of
   Kaggle's Python 3.12 — so the authors' pinned `torch==2.1.2` can be installed
   at all. Bumping torch instead would change the numerical stack the models run
   on, which is a far larger deviation.
2. **`torch_geometric==2.4.0`** instead of the pinned 2.5.3 — the newest version
   where the authors' own PGT pin imports.
3. **`torch_geometric_temporal` installed with `--no-deps`** so its stale
   `pandas<=1.3.5` does not bind — the authors had already overridden it by
   pinning pandas 2.2.

PGT stays at the authors' `0.54.0` rather than being upgraded, because PGT is
the package that defines the model architectures: changing it risks changing the
models, a far larger deviation than a PyG patch release.

Everything else — `torch==2.1.2`, `numpy~=1.26.2`, `pandas~=2.2.0`,
`scikit_learn==1.4.0`, `statsmodels==0.14.1` — is installed at exactly the
version the authors named.

---

## DengueGNN — not reproducible, per the article itself

From the published article's own statements:

- **Data availability:** "The data is available publicly in this link
  https://opendengue.org/data.html. The data details are given in this article."
- **Code availability:** *no statement appears in the article.*

OpenDengue spans 102 countries at three admin levels from the early 1990s. The
article does not state which countries, which admin level, which date range, or
how many nodes. Nor does it report any hyperparameter value — it describes a grid
search over "learning rate, hidden units, dropout, batch size" without giving the
search space or the chosen values.

**No notebook is provided for this paper**, because there is nothing to run that
would constitute a reproduction. `crosscheck/notebooks/R2_*` already implements
the architecture from Equations (2)–(14) on our Sri Lanka data and states
plainly that its numbers are not comparable to Tables 3–5.

---

## SEIR (Gopalakrishnan) — ✅ reproduces exactly

The source is a lecture notebook that prints all of its code. Already reproduced:
6/6 checks pass (`crosscheck/results/R3_seir_basic_checks.csv`). Pure
NumPy/SciPy; no Kaggle needed, but a Kaggle kernel is provided for completeness.

## SEIR–SEI (Phaijoo & Gurung) — ✅ reproduces exactly

All closed-form. Already reproduced: 7/7 checks, all nine Table 1 sensitivity
indices to six decimals (`crosscheck/results/R4_table1_sensitivity_indices.csv`).
Three internal defects found; see `crosscheck/FINDINGS.md` F4.2–F4.4. Pure
NumPy/SciPy; no Kaggle needed.
