# Cross-check workspace

An independent second implementation, built to answer one question: **are the
papers this project rests on reliable enough to build on, and is our own code
right?**

It is deliberately self-contained — its own virtual environment, its own
library, its own tests, its own notebooks. It shares **no code** with
`src/dengue_gnn`. Every formula here is re-derived from the paper that defines
it, so agreement between the two implementations is evidence and disagreement is
a finding. The single point where they meet is
`tests/test_xcheck.py::test_agrees_with_project_metrics`, which asserts the two
metric implementations produce identical numbers.

Nothing in here feeds the report until it has been re-run at `QUICK_TEST = False`
and logged in `docs/EXPERIMENT_LOG.md`.

---

## The four reproductions

| Notebook | Paper | Reproducible? | Status |
|---|---|---|---|
| [`R1`](notebooks/R1_weng2024_graph_representation.ipynb) | Weng et al., *Graph Representation Learning for Dengue Forecasting*, IEEE BigData 2024 | **Yes** — STGAT reproduces to ~5% | ✅ Full sweep done; 6 findings |
| [`R2`](notebooks/R2_denguegnn_dynamic_stgnn.ipynb) | GulMohamed et al., *DengueGNN*, Scientific Reports 16:10584 (2026) | **No** — dataset undefined | Architecture implemented and running; 2 of 3 ablations testable |
| [`R3`](notebooks/R3_seir_basic_gopalakrishnan.ipynb) | Gopalakrishnan, *The SEIR model of infectious diseases*, MTH 271 (2020) | **Yes, exactly** | ✅ 6/6 checks pass |
| [`R4`](notebooks/R4_seir_sei_sensitivity_phaijoo.ipynb) | Phaijoo & Gurung, *Sensitivity Analysis of SEIR–SEI Model of Dengue Disease*, GAMS 6(a) (2018) | **Yes** | ✅ 7/7 checks pass, 3 defects found |

Findings are collected in [`FINDINGS.md`](FINDINGS.md). The short version:

- **R4 reproduces exactly** and turned up three internal inconsistencies in the
  paper, none of which invalidate its conclusions.
- **R1's Table I reproduces — and naive persistence beats it.** Our STGAT
  reimplementation scores 24.01 MAE / 42.31 RMSE against the published
  25.38 / 44.78. Scored on identical slices with the identical metric
  convention, last-week-carried-forward gives 18.58 / 34.67 — better than all
  ten models in Table I's cross-validated column. (On the Full Dataset column
  persistence wins on MAE but A3TGCN and ASTGCN beat it on RMSE.) The paper
  reports no naive baseline. This reframes our own Phase-1 result: matching the
  persistence floor is what the benchmark achieves too.
- **R2 cannot be reproduced at all**, and the notebook says so rather than
  producing a number that looks like agreement.

---

## Setting up the venv

This workspace has its own environment at `crosscheck/.venv`, separate from
whatever the main project uses.

```bash
cd crosscheck
python -m venv .venv
```

Activate it — `.venv\Scripts\activate` on Windows, `source .venv/bin/activate`
elsewhere — then:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-geometric
```

CPU-only torch is enough: the graph is 25 nodes and 141 edges. For CUDA, take
the matching index URL from <https://pytorch.org/get-started/locally/>.

`torch-geometric-temporal` is **optional**. R1 needs it only to run Weng et al.'s
other four architectures (ASTGCN, A3TGCN, DCRNN, AAGCN) on their original stack;
STGAT — the paper's best model — needs only plain PyTorch Geometric. The notebook
detects its absence and reports those four as unavailable rather than
substituting a lookalike. See `docs/decisions/0002` for why the main project
avoids it.

> **Google Drive.** This folder syncs. A venv is thousands of small files, so
> expect the install to be slow and Drive to churn afterwards. If that becomes a
> problem, create the venv at a local path instead and point your kernel at it —
> nothing here depends on the venv living inside the repo.

### Verify

```bash
python -m pytest
```

---

## Running the notebooks

```bash
cd crosscheck/notebooks
jupyter lab
```

R3 and R4 are pure NumPy/SciPy and run in seconds. R1 and R2 need torch and
carry a `QUICK_TEST` flag, following the same convention as the main project:

- `QUICK_TEST = True` (the default) — few epochs, fewer segments. Confirms the
  pipeline executes. **Its numbers are intentionally degraded and must never be
  cited or logged.**
- `QUICK_TEST = False` — the real run.

`tools/check_notebooks.py` in the main project warns when `QUICK_TEST` is left
enabled in a committed notebook.

---

## Layout

```
crosscheck/
├── README.md              this file
├── FINDINGS.md            what the reproductions turned up
├── requirements.txt       dependencies (torch installed separately)
├── pyproject.toml         ruff + pytest config for this workspace
├── _build/                notebook cell sources — EDIT THESE
│   ├── gen_notebooks.py
│   └── _cells_r{1,2,3,4}.py
├── notebooks/             GENERATED .ipynb files — do not edit by hand
├── lib/xcheck/
│   ├── metrics.py         RMSE/MAE/MAPE variants, Moran's I, CRPS/PICP/MPIW
│   ├── data.py            loading + windowing, reference and corrected
│   ├── protocol.py        segment CV and rolling-origin index plans
│   ├── graph.py           dynamic graph construction (torch-free)
│   ├── models.py          STGAT (R1) and DynamicSTGNN (R2)
│   └── seir.py            SEIR (R3) and SEIR–SEI (R4)
├── results/               committed CSVs, one row per fold/check
└── tests/                 pytest suite, including the agreement test
```

### Notebooks are generated

Same convention as `notebooks/_build` in the main project: `.ipynb` files are
JSON blobs that review and merge badly, so the cell sources are plain Python.

```bash
cd crosscheck/_build
python gen_notebooks.py
```

**Edit `_cells_r*.py`, never the `.ipynb`** — regeneration overwrites them.

---

## Why a separate workspace at all

Three reasons, in order of importance.

1. **A cross-check that shares code with what it checks is not a cross-check.**
   If `xcheck.metrics` imported `dengue_gnn.metrics`, a bug in the SMAPE
   denominator would agree with itself perfectly.
2. **The reproductions need protocols we deliberately rejected.** R1 reproduces
   a training-inclusive evaluation and a global normalization — both things
   `docs/decisions/0001` rules out for our own work. They belong somewhere they
   cannot leak into the ablation table.
3. **It is disposable.** If the reproductions turn out not to be trustworthy,
   this directory can be deleted without touching the project. If they do prove
   reliable, individual pieces can be promoted into `src/dengue_gnn/` under the
   usual rule — stabilized, used twice, tested.

## Promoting anything out of here

Before code moves to `src/dengue_gnn/`:

- [ ] Re-scored under the **frozen protocol** in `docs/ROADMAP.md` — rolling-origin,
      3 origins × 3 seeds. The reproductions use each paper's own protocol, which
      is not ours.
- [ ] Tested in `tests/`, not only in `crosscheck/tests/`.
- [ ] The run logged in `docs/EXPERIMENT_LOG.md` with config and commit SHA.
- [ ] An ADR in `docs/decisions/` if it changes a modelling choice.

The obvious candidate is `xcheck.models.DynamicSTGNN` — a working
adaptive-adjacency ST-GNN with temporal attention, which is most of Phase 2.
Read the caveats at the end of R2 before promoting it.
