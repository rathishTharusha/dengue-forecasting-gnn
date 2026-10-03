# Physics-Informed, GAN-Augmented Graph Neural Networks for Multi-Horizon Dengue Forecasting

District-level weekly dengue incidence forecasting for Sri Lanka (25 districts, 2013–2022).
CS3631 group project — Group 05, Department of Computer Science and Engineering,
University of Moratuwa.

The project began as a spatio-temporal GNN baseline to be extended with three stackable
contributions. It has since done something more valuable first: **reproduced the benchmark
paper exactly**, then established from the data what actually caps accuracy. Both results
changed the plan, and both are documented here with the evidence.

---

## Status

| Phase | Deliverable | State |
|---|---|---|
| 0 | Literature review, project proposal | ✅ Done |
| 1 | GCN/GAT baseline, rolling-origin CV | ✅ Done — matches the persistence floor |
| **R** | **Exact reproduction of the source papers** | ✅ **Done — see below** |
| **E** | **Dataset EDA and cap analysis** | ✅ **Done — 9 findings** |
| 2 | Adaptive graph | ⚠️ **Tested.** Pure adaptive does not help; `hybrid` is ahead at n=9 and a full sweep is built but not yet run |
| 3 | Physics-informed SEIR–SEI loss | ⚠️ **Tested — negative, with a measured mechanism.** See below |
| 4 | GAN augmentation | ☐ Re-aimed: physics as a *generative* constraint, where R only has to be plausible |
| 5 | Full stacked ablation + write-up | ☐ In progress |

---

## What has been established

### 1. The benchmark reproduces — all five GNNs, both Table I columns

Weng et al. (2024) is the paper this project builds on. Running **their code, their data,
their protocol** on Kaggle:

| Model | CV MAE / RMSE | vs their `results.txt` | Full-dataset MAE / RMSE | vs Table I |
|---|---|---|---|---|
| **AAGCN** | 41.93 / 55.83 | **−0.00% / −0.00%** | 40.97 / 50.25 | +4.3% / +2.1% |
| **DCRNN** | 45.81 / 71.32 | −0.35% / −0.40% | 25.46 / 35.75 | −2.8% / −1.5% |
| STGAT | 25.80 / 45.43 | +1.64% / +1.45% | 22.99 / 40.82 | +3.2% / −4.3% |
| ASTGCN | 34.33 / 48.82 | +1.92% / +2.30% | 22.13 / 33.20 | +7.3% / +2.5% |
| A3TGCN | 35.18 / 60.11 | +3.39% / +2.71% | 19.60 / 31.62 | +5.1% / +3.5% |

Worst deviation across all 20 comparisons: **7.3%**. Two models match to four decimals.

**And Table I's two columns are the reverse of what their names suggest.** "Cross Validated"
is measured on the `full` loader — training data included. "Full Dataset" is the genuinely
held-out score. Established by testing four candidate definitions; only one fits all models.

→ [`reproduction/`](reproduction/) · [`reproduction/REPRODUCIBILITY_MATRIX.md`](reproduction/REPRODUCIBILITY_MATRIX.md)

### 2. Naive persistence beats every model in that cross-validated column

Scored on identical slices with the identical metric convention, last-value-carried-forward
gives MAE 18.58 / RMSE 34.67 against the best published 25.38 / 44.78. The paper reports no
naive baseline.

**This reframes our own Phase-1 result.** Matching the persistence floor is not underperformance
against the literature — it is what the literature achieves, once a naive baseline is placed
beside it.

→ [`crosscheck/FINDINGS.md`](crosscheck/FINDINGS.md)

### 3. The ceiling is temporal, not architectural

| Predictor | best \|r\| | r² |
|---|---|---|
| cases at *t−1* | 0.921 | **0.848** |
| cases at *t−3* (the paper's horizon) | 0.817 | 0.667 |
| best covariate (`meanQair`, lag 4) | 0.145 | **0.021** |

→ [`analysis/notebooks/E1_dataset_eda.ipynb`](analysis/notebooks/E1_dataset_eda.ipynb)

### 4. The adaptive graph does not help — and neither does the graph

Paired against a fixed graph on identical folds and seeds: `adaptive` is **+0.46 RMSE worse**
(p = 0.09), `hybrid` +0.19 (p = 0.37), and removing message passing entirely is −0.54
(p = 0.49). Twenty-five independent time series do as well as the graph model.

A genuine negative result, reported as one.

→ [`analysis/notebooks/E2_adaptive_graph.ipynb`](analysis/notebooks/E2_adaptive_graph.ipynb)

---

### 5. The physics cannot help the point forecast — and we can say why

The project's promised contribution was a physics-informed loss from the SEIR–SEI
model. It was built, tested six ways, and does not work. The reason is a number:

| | |
|---|---|
| `R_t` predictability, out of sample | **26%** — from its own past, and nothing else |
| implied error per step | **1.90×** multiplicative, compounding over 3 weeks |
| persistence, for comparison | cases at *t−1* explain **r² = 0.85** directly |

Routing a forecast through `R` re-injects a factor-of-two error where persistence
had none. Climate explains **−0.03** of `log R` even with the hump-shaped thermal
response R₀'s own sensitivity indices imply; susceptible depletion is ~**500×**
too slow to register on a 3-week horizon in an endemic population of 21.9M.

The deeper finding is that the point forecast is at its information limit. The
slope of predicted log-growth on true log-growth is **0.002** — the models are
persistence, and under squared error that is *correct*, because the conditional
mean of a near-unpredictable residual is zero.

### 6. Outbreaks are detectable even though they are not countable

| model | AUC, 3 weeks ahead |
|---|---|
| current level ÷ district threshold | 0.807 |
| **+ `R̂` (the renewal quantity)** | **0.826** |
| + climate | 0.820 (worse) |

`R̂` is the largest single addition — the first time the mechanistic quantity has
measurably helped. Sustained `R > 1` is visible for ~6 weeks before an outbreak
crosses the threshold. This is where the physics belongs.

---

## New to this project? Start with the handbook

[`docs/handbook/`](docs/handbook/00_START_HERE.md) is a complete twelve-chapter
course on this project — the disease, the mathematics, the machine learning, the
code, and every decision behind them. It assumes you can read Python and nothing
else, and builds up epidemiology, graph neural networks and forecasting
evaluation from first principles.

It is written so that you could delete this repository and rebuild it from
scratch using only official library documentation.

---

## Quick start

```bash
git clone https://github.com/rathishTharusha/dengue-forecasting-gnn.git
cd dengue-forecasting-gnn
```

**Everything needed to run is in the clone.** The two data files the notebooks require
(`sri_lanka_2013-2022_shifted.npy`, `sri_lanka_adj_list.json`) are tracked under
`notebooks/baseline/`. Nothing has to be downloaded first.

### Run the EDA (2 minutes, no GPU, no torch)

```bash
python -m venv .venv && .venv\Scripts\activate      # source .venv/bin/activate elsewhere
pip install numpy scipy pandas matplotlib
jupyter lab analysis/notebooks/E1_dataset_eda.ipynb
```

### Run the adaptive-graph experiment (~7 minutes, CPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
jupyter lab analysis/notebooks/E2_adaptive_graph.ipynb
```

Cached results ship with the repo, so it renders instantly; delete
`analysis/results/adaptive_graph_runs.json` to force a re-run.

### Reproduce the papers (Kaggle)

```bash
python reproduction/verify_local.py          # build the pinned stack, smoke-test it
python reproduction/kaggle/setup_kaggle.py --check
```

See [`reproduction/README.md`](reproduction/README.md). Weng et al.'s stack cannot be installed
on Kaggle's stock image (their `torch==2.1.2` has no Python 3.12 wheel), so the kernels
bootstrap a standalone Python 3.11 — one of **three forced deviations**, each documented with
the exact error that forces it.

### Verify our own code

```bash
pytest                       # project tests, incl. 12 tests pinning the renewal physics
cd crosscheck && pytest      # independent reimplementation, incl. cross-agreement test
python tools/check_notebooks.py
python scripts/verify_seir_paper.py   # SEIR-SEI vs Phaijoo & Gurung Table 1
```

### Re-run the analyses behind the recent findings

**Need torch only** (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).
These train nothing and finish in seconds:

```bash
python analysis/_build/renewal_feasibility.py       # renewal equation vs the data (EXP-023)
python analysis/_build/r_predictability.py          # is R predictable at all (EXP-024)
python analysis/_build/mechanistic_r.py             # depletion + thermal structure (EXP-024)
python analysis/_build/outbreak_signal.py           # outbreak detectability (EXP-025)
```

**Need the full pinned stack** — torch 2.1.2 / PyG 2.4.0 / torch-geometric-temporal
0.54.0, built by `python reproduction/verify_local.py --env-only`. These train, so
budget minutes to tens of minutes:

```bash
python analysis/_build/run_reproduced_baseline.py   # the five verified architectures
python analysis/_build/diagnose_errors.py           # where the error lives (EXP-021)
python analysis/_build/response_diagnosis.py        # damping and response (EXP-025)
python analysis/_build/run_physics.py --quick       # the physics arms (EXP-023)
```

Each writes a JSON under `analysis/results/` and prints its table.

---

## Repository layout

```
.
├── notebooks/            Phase-1 baseline + Weng et al. reproduction notebooks
│   └── baseline/         the tracked data files live here
├── src/dengue_gnn/       torch-free shared library: metrics, SEIR, data loader
├── scripts/              verify_seir_paper.py — SEIR reproduction against the paper
├── paper/                the write-up, by section
│
├── reproduction/         run the AUTHORS' code to get the AUTHORS' numbers
├── crosscheck/           an independent reimplementation — is our code right?
├── analysis/             EDA, cap analysis, and contribution experiments
│
├── docs/                 setup, data, roadmap, experiment log, ADRs, handbook
├── results/              index + the SEIR validation table
└── tests/                unit tests for src/ and the renewal physics
```

The Phase-2/3 implementation that backed EXP-001 – EXP-014 was removed once the
work moved to architectures verified against the published papers. It is at tag
**`phase23-archive`**; every one of those log entries keeps its config and
unrounded numbers inline, so they stay citable without it.

```bash
git checkout phase23-archive     # to re-run any of EXP-001..014
```

### Why three separate workspaces

They answer different questions and must not share code:

| Workspace | Question | Independence |
|---|---|---|
| `reproduction/` | Do the authors' numbers reproduce? | Runs *their* code, pinned to a commit |
| `crosscheck/` | Is *our* code right; are the papers sound? | Shares **no code** with `src/` |
| `analysis/` | Why the cap, and what moves it? | Builds on both |

`crosscheck/` sharing code with `src/` would defeat its purpose — a bug that agrees with
itself proves nothing. The single point of contact is a test asserting the two metric
implementations produce identical numbers.

**Local-only (git-ignored):** `papers/` (copyrighted), `reference_repo/` (third-party),
`datasets/` (raw dump), the course handout. None is needed to run anything.

---

## Working on this project

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first. The rules that matter most here:

1. **Branch per piece of work.** Never commit directly to `main`.
2. **Log every real experiment** in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) with the
   config that produced it.
3. **Keep the evaluation protocol identical** across baseline and contributions, or the
   ablation table is meaningless.
4. **One run is not a result.** Report the seed spread; a difference smaller than it is
   inconclusive, not a finding.
5. **Negative results get reported.** Two of the four established findings above are negative,
   and they redirected the project more usefully than a small win would have.

---

## Team — Group 05

Tharusha Perera · Praveen De Silva · Janith Mahanama · Bimsara Udurawana ·
Maleesha Kumarasinghe · Praveen Nawarathna

## Key references

- Weng et al. (2024), *Graph Representation Learning for Dengue Forecasting*, IEEE BigData —
  the benchmark, reproduced in `reproduction/`.
- Phaijoo & Gurung (2018), *Sensitivity Analysis of SEIR–SEI Model of Dengue Disease*, GAMS —
  the compartmental model behind the physics loss; reproduced, with three defects found.
- Wu et al. (2019), *Graph WaveNet*, IJCAI — the adaptive-adjacency construction tested in
  `analysis/`.
- Yoon et al. (2019), *Time-series Generative Adversarial Networks*, NeurIPS — augmentation.

## License

Code is MIT ([`LICENSE`](LICENSE)). Data and third-party papers are **not** covered by it —
see [`docs/DATA.md`](docs/DATA.md).
