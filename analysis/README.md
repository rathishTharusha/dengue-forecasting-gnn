# Analysis workspace

Where the project asks **why** the published numbers stop where they do, and
what to do about it. The two workspaces it depends on answer different
questions:

| Workspace | Question | Status |
|---|---|---|
| `reproduction/` | Do the authors' numbers reproduce? | ✅ all five GNNs, both Table I columns, worst deviation 7.3% |
| `crosscheck/` | Is our own code right; are the papers' methods sound? | ✅ 4 papers, 40 tests |
| `analysis/` | Why do the numbers cap there, and what moves them? | in progress |

Reproduction had to come first. Explaining a number you cannot reproduce is
guesswork.

---

## Notebooks

| Notebook | What it does |
|---|---|
| [`E1_dataset_eda.ipynb`](notebooks/E1_dataset_eda.ipynb) | Nine findings about the dataset, each computed in-notebook |
| [`E2_adaptive_graph.ipynb`](notebooks/E2_adaptive_graph.ipynb) | Tests a learned adjacency by flipping one flag in the authors' own code |

Notebooks are **generated** — same convention as everywhere else in this repo:

```bash
cd analysis/_build && python gen_analysis.py
```

Edit `cells_*.py`, never the `.ipynb`.

---

## E1 — the nine findings

| # | Finding | Why it matters |
|---|---|---|
| F1 | All 11 channels identified by name and lag, 100% exact | `docs/DATA.md` names a land-surface-temperature channel that does not exist |
| F2 | The unshifted array is reconstructible from `vertical.csv` | The 10 unshifted Table I rows may be reproducible after all |
| F3 | 4% of weather values are `0`-filled; usable range compressed 26× | Must be fixed before any covariate-using contribution |
| F4 | Week 395 is a 19× spike across 18 of 25 districts at once | A reporting artifact, not an epidemic |
| F5 | The 2017 outbreak is in **test** for segment 0.6, **train** for the rest | Explains ASTGCN's ±19.85 MAE and the cross-segment spread |
| F6 | cases r²=0.85 at lag 1; best covariate r²=0.02 | Persistence is strong by construction |
| F7 | The adjacency has two one-way edges; Jaffna has degree 1 | A hand-built graph carrying defects a learned one would not |
| F8 | Neighbours r=0.62 vs non-neighbours r=0.55 (p=0.0002) | The graph is real but weak — argues for a learned adjacency |
| F9 | Bimodal seasonality, 2.9× peak/trough, no seasonal feature anywhere | A 3-week window cannot represent a 52-week cycle |

### What caps accuracy

| Predictor | best \|r\| | r² |
|---|---|---|
| cases at *t−1* | 0.921 | **0.848** |
| cases at *t−3* (the paper's horizon) | 0.817 | 0.667 |
| best covariate (`meanQair`, lag 4) | 0.145 | **0.021** |

The target's own history explains ~85% of variance at one week; the best
meteorological covariate explains 2%. The ceiling looks temporal, not
architectural — which is why `crosscheck/` finds naive persistence beating every
model in Table I's cross-validated column.

**Stated caveat:** those are *pooled linear* correlations, and F3 attenuates
them. "The covariates are useless" is not established — it needs re-measuring on
cleaned data.

---

## E2 — the adaptive graph

The authors instantiate the one model that can learn its graph as
`AAGCN(..., adaptive=False)` — the only occurrence of `adaptive` in their
codebase, overriding PGT's default of `True`. **No reproducible paper here tests
a learned adjacency.**

E2 rebinds `evaluation.AdaptiveGCN` to a class identical to theirs except for
two flags, and runs their `run_aagcn` unchanged. Four variants, re-seeded
identically, reported **per segment** so F5's outbreak fold cannot hide the
effect.

AAGCN is the right model to perturb: it reproduced to −0.0014%.

**Prediction recorded before running:** little movement in the pooled average,
with any real effect concentrated in segment 0.6.

---

## Ground rules

Carried over from `CONTRIBUTING.md`, and they matter more here than anywhere
else in the repo, because this is where the temptation to over-claim lives.

- **One run is not a result.** AAGCN's own segment-to-segment spread is ±1.11
  MAE. Anything smaller than that needs multiple seeds before it is reported.
- **Per segment, not pooled.** The five folds are not the same problem. Segment
  0.6 extrapolates to an unseen outbreak; the others interpolate.
- **The held-out column, not the cross-validated one.** Table I's "Cross
  Validated" column is training-inclusive. Improvements measured there may be
  better fitting, not better forecasting.
- **A negative result is a result.** If the adaptive graph does not help, that is
  evidence the ceiling is temporal, and it redirects effort rather than wasting
  it.
