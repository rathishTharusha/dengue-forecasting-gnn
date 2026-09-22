# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A research project (CS3631 Group 05, University of Moratuwa) on district-level weekly dengue
forecasting for Sri Lanka — 25 districts, window 3 → horizon 3 weeks. It is **not** a product
codebase: the deliverable is a defensible results table and a paper. Most changes here exist to
make a number citable, so the protocol invariants below outrank convenience.

The headline empirical fact that shapes every design decision: cases at *t−1* explain
**r² = 0.85** of cases at *t*, the best climate covariate explains **r² = 0.02**. Naive
persistence is a hard floor, and several published GNN results do not clear it once reporting
artifacts are removed. Expect negative results and report them as such.

## Commands

```bash
python -m pytest                         # 94 tests: src/, renewal physics, leakage guard
python -m pytest tests/test_renewal.py -q         # one file
python -m pytest tests/test_seir.py::test_growth_rate_sign_tracks_the_threshold -q   # one test
python -m pytest crosscheck/tests -q     # the independent reimplementation's own suite
python -m ruff check src tests tools     # CI lints only these three
python -m ruff format --check src tests tools
python tools/check_notebooks.py          # nbformat validity, execution order, QUICK_TEST left on
```

CI (`.github/workflows/ci.yml`) runs exactly those and **does not install torch** — the shared
library under `src/` must stay torch-free. Anything importing torch belongs in `analysis/lib/`,
`seirgnn2/`, or a notebook.

Torch is available locally (CPU-only: torch 2.13, PyG 2.8). Kaggle credentials are configured in
`~/.kaggle/`; the orchestration scripts are `scripts/sweep_orchestrator.py`,
`scripts/check_kaggle_all.py`, `reproduction/kaggle/setup_kaggle.py`.

### Running experiments

```bash
# Newest harness (seirgnn2) — screening grid, CPU, ~3 min for 135 runs on 6 workers
python seirgnn2/sweep.py screen --workers 6 --epochs 300

# Analysis one-shots: no training, seconds each, each writes analysis/results/<name>.json
python analysis/_build/renewal_feasibility.py
python analysis/_build/r_predictability.py
python analysis/_build/mechanistic_r.py
python analysis/_build/outbreak_signal.py

# Paper stages S4–S9 (these train; minutes to hours)
python analysis/_build/run_s4_seir_lstm.py
python analysis/_build/run_s5_seir_gnn.py        # see the warning under "Known defect" below
python analysis/_build/run_s9_confirmatory.py
```

### Notebooks are generated artifacts

Every `.ipynb` in `notebooks/`, `analysis/notebooks/`, `crosscheck/notebooks/` and
`reproduction/kaggle/kernels/` is built from plain-Python cell sources. **Edit the generator,
never the `.ipynb`.**

```bash
cd analysis/_build && python gen_analysis.py     # same pattern in notebooks/_build, crosscheck/_build, reproduction/_build
```

`full_paper/reproduce_full_paper.ipynb` is the exception — a hand-maintained standalone package
notebook that must keep working from `full_paper/` alone, with no repo-root imports.

## Architecture

### Four workspaces that deliberately do not share code

| Workspace | Question it answers | Independence rule |
|---|---|---|
| `reproduction/` | Do Weng et al.'s published numbers reproduce? | Runs *their* code pinned to a commit; three documented forced deviations |
| `crosscheck/` | Is *our* code right; are the papers' methods sound? | Shares **no code** with `src/` — has its own `pyproject.toml` and requirements |
| `analysis/` | Why does accuracy cap where it does, and what moves it? | Builds on both |
| `seirgnn2/` | The current leakage-controlled SEIR-GNN re-run | Self-contained; imports only `analysis/lib/corrected_data.py` |

`crosscheck/` importing from `src/` would defeat its purpose (a bug that agrees with itself
proves nothing). The single sanctioned point of contact is a test asserting both metric
implementations agree. Do not "DRY up" across these boundaries.

`src/dengue_gnn/` is the torch-free shared library: `metrics.py`, `seir.py` (SEIR–SEI +
`MAX_WEEKLY_LOG_GROWTH` / `CEILING_R0_MAX` growth ceiling), `data.py`.

### Two datasets — never mix them

1. **Legacy** `notebooks/baseline/sri_lanka_2013-2022_shifted.npy` + `sri_lanka_adj_list.json`.
   The benchmark paper's array. Contains reporting backlogs (week 395 is a 19× spike across 18
   districts at once) and `0`-filled weather channels. Used only for reproducing published
   numbers.
2. **Corrected/rebuilt** `data/corrected/rebuilt_*.npy` + `data/external/*.csv`, 559 weeks
   2013–2024, loaded through `analysis/lib/corrected_data.py`. This is what every new result
   uses. Missing weeks stay `NaN`; nothing is interpolated.

Cleaning the artifacts moves the persistence floor from RMSE 44.80 to 29.52 and flips several
models from "beats baseline" to "worse than baseline". A number is meaningless without saying
which dataset produced it.

### The no-future-information rule is enforced in code

`analysis/lib/corrected_data.py` defines `LAGS = {"cases": 1, "climate": 2, "ndvi": 0,
"population": 0}` and `windows()` applies them when slicing.
`tests/test_no_future_leakage.py` perturbs every value at or after each limit and asserts the
output does not move. **Any new input path needs its own leakage test.** Lags are fixed a
priori and are never chosen because they correlate well on test data.

Hindsight-only knowledge (the 2022–23 seroprevalence survey, serotype-switch dates) is
validation-only or lives in an arm explicitly named `oracle_*`, which is never a finalist.

### The frozen evaluation protocol

Pre-registered in `docs/SEIR_GNN_EXPERIMENT_PLAN.md` (rules R1–R8) and implemented in
`seirgnn2/core.py::build_folds`:

- rolling origins **0.55 / 0.70 / 0.85**, seeds **0, 1, 2**
- window 3 → horizon 3, normalisation from training weeks only, no shuffling
- early stopping on validation; **selection on validation RMSE only** — test numbers are written
  to disk but not looked at until a config is frozen and committed
- one row per `(config, origin, seed)`, **never pre-aggregated**, so paired tests are possible
- metrics RMSE / MAE / SMAPE / MAPE(≥1), overall and per horizon

Changing any of this after seeing test numbers is a deviation that must be logged with its
reason. Keeping the protocol identical across baseline and contribution is what makes the
ablation table mean anything.

### Model space (`seirgnn2/models.py`)

`HEADS = ("direct", "residual", "foi", "foi_res")` × `BACKBONES = ("none", "gcn", "gat",
"adaptive", "hybrid")`. `foi` routes the prediction through a force-of-infection physics decoder
against SEIR state reconstructed by `seir_state()`; `none` removes message passing entirely and
is the control that shows whether the graph earns its place (so far: it does not, ~0.5 RMSE,
p ≈ 0.49).

## Working conventions

- **Branch per piece of work**; never commit to `main`. Prefixes `feat/ exp/ fix/ docs/`.
  Current work is on `exp/seir-gnn`. Per `docs/SEIR_GNN_EXPERIMENT_PLAN.md` R8: **commit, never
  push** — the user pushes before a Kaggle kernel needs new code.
- **Every run whose numbers might reach the report** gets an `EXP-NNN` entry in
  `docs/EXPERIMENT_LOG.md` (append-only, newest at top) with commit SHA, full config, seeds,
  origins and the *unrounded* table. The template is at the top of that file.
- **One run is not a result.** Report the seed spread; a difference smaller than it is
  inconclusive, not a finding. Prefer paired tests across matched `(origin, seed)`.
- Never quote `QUICK_TEST = True` numbers — they are intentionally degraded and
  `tools/check_notebooks.py` warns about them.
- Git-ignored and never committed: `papers/`, `datasets/`, `reference_repo/`, `data/raw/`, the
  course handout.

## Current state and a known defect

`full_paper/` is a standalone distribution package (LaTeX source, data, reference PDFs,
`reproduce_full_paper.ipynb`). Its Stage S5 leaderboard
(`full_paper/outputs/csv/stage_s5_leaderboard.csv`) and
`full_paper/overleaf/sections/05_results.tex` report the SEIR-GNN FOI head as a win.

**`seirgnn2/core.py` opens by documenting four defects in the script that produced those S5
numbers** (`analysis/_build/run_s5_seir_gnn.py`): one gradient step per epoch (~60 steps total),
a SMAPE objective scored by RMSE, a force of infection held constant across the horizon, and all
input channels collapsed to a single scalar before the backbone. `seirgnn2/` is the corrected
re-run. Its first screening pass (`seirgnn2/results/screen.json`, 138 rows) puts the `foi` head
**last** (val RMSE 26.89 vs persistence 17.86), which contradicts the S5 claim.

Treat any S5-derived number as provisional until the `seirgnn2` re-run replaces it, and do not
propagate S5 numbers into new text. `seirgnn2/sweep.py` references a `stats.py` for paired tests
that does not exist yet, and `grids.py` currently defines only `screen()`.

One more inconsistency worth knowing: `seirgnn2/core.py::seasonal_features` justifies its
week-of-year features by citing "EDA finding F9", but **F9 was retracted** (see
`analysis/README.md` and `docs/EXPERIMENT_LOG.md`: shape r = −0.065, 0/25 districts significant).
The feature happens to be the strongest arm in the screen, so the fix is the rationale, not the
feature.

## Orientation

- `docs/handbook/00_START_HERE.md` — twelve chapters covering the disease, the maths, the ML,
  and every decision behind the code. Written so the repo could be rebuilt from scratch.
- `docs/EXPERIMENT_LOG.md` — what has actually been tried, with numbers. Read before proposing
  an experiment; most obvious ideas have been run.
- `docs/SEIR_GNN_EXPERIMENT_PLAN.md` — the pre-registered plan, stages S0–S9.
- `crosscheck/FINDINGS.md`, `reproduction/REPRODUCIBILITY_MATRIX.md` — what reproduced and what
  did not.
