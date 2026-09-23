# seirgnn2 — the leakage-controlled SEIR-GNN harness

The re-run of Stage S5. Everything here exists because
`analysis/_build/run_s5_seir_gnn.py` — the script behind the paper's S5
leaderboard — had four defects that together make its numbers uninformative
about whether physics-informed graph networks work on this series:

| defect in `run_s5_seir_gnn.py` | why it invalidates the number |
|---|---|
| one full-batch gradient step per epoch (~60 total) | nothing converged |
| SMAPE objective, RMSE metric | SMAPE is dominated by districts averaging 2–5 cases/week, RMSE by Colombo (226) and Gampaha (136) — the objective pulled away from the metric |
| force of infection constant across the horizon | one scalar repeated three times; the 3-week structure was never modelled |
| `nn.Linear(in_dim, 1)` before the backbone | every covariate collapsed to a scalar before the graph saw it, which is why its input-level factor moved results by <1% |

**Do not cite any S5-derived number.** Use the tables below.

**Nor any S9-derived number.** `analysis/_build/run_s9_confirmatory.py` is described as a
9-origin paired permutation test and is none of those things: it runs on the 3 origins S5
produced, its "paired" differences subtract hardcoded scalars
(`b_star_test_rmse = 34.837`, `persistence_test_rmse = 36.016`,
`seir_lstm_test_rmse = 62.608`) rather than matching on origin, and the reported early-warning
p-value is the literal `p_auc = 0.04  # Simulated`. See EXP-037. `stats.py` here does the
intended test properly.

---

## Running it

```bash
python seirgnn2/sweep.py <grid> --workers 6 --epochs 300   # writes results/<grid>.json
python seirgnn2/stats.py <grid>                            # paired tests vs persistence
python seirgnn2/stats.py <grid> --ref "LSTM+foi_res season nb" --metric val_RMSE

python seirgnn2/backbones.py      # build + forward-pass every encoder (~10 s)
python seirgnn2/diagnose_foi.py   # why the physics head behaves as it does
python seirgnn2/diagnose_seed.py  # which initial-state seeding is reachable
```

Grids live in `grids.py`, one function each: `screen`, `foi`, `converge`, `real`,
`combo`, `window`, `confirm`, `remedies`. `--keep` also saves every run's forecasts to
`results/<grid>_preds.pkl` (git-ignored), which `ensemble.py` and forecast-level
diagnoses read. A sweep runs every config × 3 origins × 3 seeds and writes **one row per
(config, origin, seed), never pre-aggregated** — that is what makes the paired
tests possible, and the repo's logging convention asks for exactly it.

Everything runs on local CPU. The whole `screen` grid is 135 runs in ~3 minutes;
`real` (162 runs, includes DCRNN at 825k params) took 71 minutes.

### Dependencies

`torch_geometric_temporal` imports `SparseTensor` from `torch_sparse` in exactly
one module (`nn/recurrent/evolvegcno.py`), which has no wheel for torch 2.13 and
needs a source build. EvolveGCN is not one of the five architectures and is never
constructed, so `backbones.install_shim()` supplies the symbol from
`torch_geometric.typing`. This is why the pinned torch 2.1.2 stack in
`reproduction/` is **not** needed here.

```bash
pip install torch-geometric                      # with deps: PyG itself is required
pip install --no-deps torch-geometric-temporal   # without: torch-sparse would build from source
```

Both are needed. Installing only the second one skips `torch_geometric`, and on a
machine that does not already have it every graph arm then dies inside its worker
while the LSTM arms run fine — the grid comes back short with no obvious error.
That cost one Kaggle kernel (EXP-038 note 4), so `backbones.check()` is now a
hard gate in the kernel.

### Kaggle

`python scripts/build_seirgnn2_kernel.py --grid <grid>` generates a kernel that
clones the pushed branch at a pinned SHA, so nothing drifts between local and
remote. **Verified**: the `confirm` grid returns 144 rows on both, with every arm
agreeing to within 0.11 RMSE and persistence identical to the decimal. CPU
workers, not GPU — these architectures are small and the grid is many short runs.

**Pair only within one environment.** Kaggle and local agree on means, not on
individual runs: the same B configuration at origin 0.70, seed 0 scores 11.4680
on Kaggle (identically in two separate kernels) and 11.3414 locally (identically
in two separate local runs). A different torch build and CPU change the last
digits, which compound over training. Every paired comparison in the log keeps
its control inside the same grid, so none is affected -- but never pair a Kaggle
arm with a local control.

---

### Remedy switches (`models.Net`, all off by default)

Added for `docs/REMEDIES_PLAN.md`; with every switch off, runs reproduce earlier
grids **bit for bit** — verified against `screen.json` — provided torch runs
single-threaded, as `sweep.py` and the Kaggle kernel both enforce. Multi-threaded
reductions change the last digits, which compound over 300 epochs.

| option | values | what it does |
|---|---|---|
| `norm` | `fold`, `revin_mean`, `revin` | reversible instance normalisation per district-window (Kim et al. 2022); direct head only |
| `node_emb` | int | learned identity per district, joined at the head (Shao et al. 2022) |
| `backbone="linear"` | — | inputs straight to a linear head; with `dist="nb"` + `node_emb` it is a negative-binomial GLM |
| `aux_phys` | float | SEIR force-of-infection head as an auxiliary loss, forecast stays direct (Rodríguez et al. 2023) |
| `backbone="knn"` | — | analogue forecasting in `knn.py`; no training |

## The protocol

Frozen, from `docs/SEIR_GNN_EXPERIMENT_PLAN.md` R5, implemented in
`core.py::build_folds`: rolling origins 0.55 / 0.70 / 0.85, seeds 0/1/2, window
3 → horizon 3, normalisation from training weeks only, early stopping on
validation, metrics on raw counts pooled over windows.

**Selection is on validation RMSE only (R6).** Test is written to disk and not
compared until a config is frozen. Every table below is ordered by validation.

### Two pairing units, and why `stats.py` prints both

- **`origin`** — seeds averaged, one difference per forecast origin. The origin
  is the only thing resampled from the data, so this is the unit a claim about
  the series rests on. With 3 origins an exact sign-flip test **cannot return a
  two-sided p below 0.25**. `stats.py` prints that floor so a 3-origin grid can
  never be misread as significant.
- **`origin_seed`** — 9 units, floor 0.004, but the three seeds inside an origin
  share their data and are replicates of *initialisation only*. A p-value here
  is a statement about run-to-run stability, not about the series.

Everything in the screening tables is `origin_seed`; nothing there can be
significant at the `origin` unit with only three origins. The **confirmatory**
grid (`grids.confirm`, `core.ORIGINS_9`) runs nine origins with disjoint test
spans and reaches a floor of 0.004 — that is the one to quote. See EXP-038.

---

## What has been tried, and the answer

The point of this table is that nobody repeats these.

### Levers that worked

| lever | effect on val RMSE | evidence |
|---|---|---|
| **Negative-binomial likelihood** (`dist="nb"`) | **−0.41**, 9/9, p_adj = 0.013 | `combo.json`. The largest single controlled gain in the whole study — bigger than any architecture difference. Squared error on `log1p` scored by RMSE is biased low by construction: `expm1` of a log-space mean is the conditional *median*, RMSE is minimised by the *mean*. `error_diagnosis.json` measures that bias at −12.92 on outbreak windows. NB2's `mu` **is** the mean. |
| **Seasonal features** (`use_season`) | −0.88 vs no features | `screen.json`. Largest input-side effect. See the caveat on F9 below. |
| Best stacked arm vs persistence | **−2.20**, 9/9, p_adj = 0.010 | `combo.json`, AAGCN + direct + season + NB: **15.66** vs persistence 17.86 |

### Levers that did not work

| lever | result | evidence |
|---|---|---|
| **The graph itself** | `gcn` beats `none` (no message passing) by **0.07** | `screen.json`. Consistent with the project's earlier finding. 25 independent series do as well as the graph. |
| **Architecture choice** | spread across *working* encoders is **0.15** | `real.json`. AAGCN 16.76, ASTGCN 16.84, LSTM 16.91. STGAT (23.32), A3TGCN (28.63) and DCRNN (34.89) are far worse on the direct head. |
| **λ re-parameterisation** (`lam_param="log"`) | **0.01** | `foi.json`. Measured saturation is real — sigmoid starts 809× above the inverted median and 57% of cells need \|raw\| > 6 — but it is **not binding**; the optimiser traverses the flat region. |
| **More training** | every one of 72 runs stopped early at 3000 epochs / patience 200; best epoch 21–225 | `converge.json`. 10× budget bought the physics head 0.04. Halving the LR made it worse. |
| **Residence-time state seeding** (`state_seed="decon"`) | *worse*: 19.4% of targets unreachable vs 14.9%, λ learnability r² 0.068 vs 0.243 | `diagnose_seed.py`. The dimensionally-correct seeding is the worst of the three. `lagged` stays the default. |
| **Learnable E₀ scale + ρ** (`state_fit=True`) | −2.2 on the physics head, still 8 RMSE behind direct | `foi.json` |
| **Longer input window** (6, 12 vs 3) | **null** — margin over the matched baseline is flat: AAGCN+direct −2.20 / −2.20 / −2.28 | `window.json`, EXP-036. Predicted to be the largest remaining lever; it is not. Absolute RMSE *looks* worse at longer windows, but that is the fold boundaries moving. **Never compare arms across windows on absolute RMSE** — `sweep.run` emits one persistence row per window for this reason. |

| **Literature remedies** (RevIN, district identity, STID, NB-GLM, k-NN, ensembles, SEIR as auxiliary constraint) | **none adopted**; best is the SEIR auxiliary at −0.04, 6/9, n.s. RevIN significantly *worse* (+1.27, 0/9) | `remedies+ens.json`, EXP-040. Even k-NN (no network, no training) has residual correlation 0.92 with the best model — the limit is the data, so ensembles have nothing to average |

| **More real data** (25 → 100% of training windows) | flat: 15.85 → 15.66, nothing past 75% | `curve.json`, EXP-041. The model is neither data-starved nor overfitting |
| **Synthetic training data** (TimeGAN, SEIR-simulated epidemics, jitter, LDS) | **none adopted**; TimeGAN significantly *worse* (+0.53, 1/9) and worsens outbreak bias | `augment.json`, EXP-042. TimeGAN loses the tail (max 882 vs real 2,631). Also tried in EXP-010 on legacy data, same verdict |

| **Climate at longer lags** (2–13, 2–25 weeks) and an 8-week window | **none adopted**; longer lags progressively *worse* (+0.13 → +0.62). Trees gain from climate (−0.49), the network does not | `climate.json`, EXP-044. Exogenous inputs reach only a linear head shared by all districts |
| **Architecture: nonlinear head, district seasonal curves, global context, residual+NB** | **none adopted**; best −0.07 (global context, 5/9). Nonlinear head lets climate help (−0.24) but costs as much itself | `arch.json`, EXP-045. Horizon-3 improves 0.14–0.22 in three arms — a lead only |

### Structural facts about the physics head

Measured by `diagnose_foi.py`, which inverts the simulator by bisection for the
λ that reproduces each true count exactly. These are **backbone-independent** —
they sit downstream of the encoder, so A3TGCN inherits them unchanged.

- **The λ=0 floor overshoots 14–16% of targets.** `E₀ = cases[t−2]/ρ` and half of
  E matures within the week, so with transmission switched *entirely off* the
  simulator still emits more cases than truth. Those cells are unreachable at any
  network output. 39% of cells need λ pinned at 0.
- **λ is much less learnable than the direct target.** log λ* has r² = 0.22–0.26
  from case history; the direct target has r² = 0.81–0.82. This is the project's
  own published R_t result (26% predictable) reappearing inside the head.
- **Susceptible depletion is *not* the problem** — only 2.5% of cells hit the
  S clamp, and holding S at `s0` changes nothing. This was the obvious
  hypothesis and it is wrong.
- Routing a matched 2-parameter rule through SEIR rather than predicting directly
  costs **+28% to +41% RMSE** across the three origins. That is a statement about
  2-parameter capacity, not a ceiling for all models.

---

## Headline results

`combo.json`, 219 rows, selection on validation.

| arm | val | test |
|---|---|---|
| AAGCN + direct, season, NB | **15.66** | 37.55 |
| LSTM + direct, season, NB | 15.80 | 35.70 |
| ASTGCN + foi_res, season, NB | 16.65 | 35.26 |
| LSTM + foi_res, season, NB | 17.27 | 34.97 |
| persistence | 17.86 | 36.02 |

### The two comparisons the paper needs

**SEIR-GNN vs SEIR-LSTM, physics head, matched on everything else:**
ASTGCN+foi_res 16.65 vs LSTM+foi_res 17.27 → **−0.62, 9/9, p_adj = 0.009**.

**Same graph, direct head:** AAGCN 15.66 vs LSTM 15.80 → −0.14, 5/9,
p_adj = 0.315. **Not significant.**

So what the graph adds appears specific to estimating the force of infection —
a transmission quantity where spatial coupling is mechanistically plausible —
and absent when regressing cases directly. The null case is what makes the
positive one worth reporting.

### Confirmatory result (EXP-038) — nine disjoint origins, the origin unit

| comparison | delta | wins | p | p_adj |
|---|---|---|---|---|
| ASTGCN+foi_res vs LSTM+foi_res | **−0.75** | **8/9** | **0.012** | 0.059 |
| AAGCN+direct vs LSTM+direct | −0.95 | 7/9 | 0.531 | 0.885 |
| ASTGCN+foi_res vs persistence | −2.92 | 8/9 | 0.066 | 0.166 |

**The pre-registered endpoint is not met.** The criteria were p_adj < 0.05, ≥6/9
origins, and the same direction as the 3-origin table. SEIR-GNN vs SEIR-LSTM
meets two of three; p_adj = 0.059 against a 0.05 bar (raw p = 0.012, the gap
being multiplicity correction). Report as a consistent directional advantage that
does not clear the bar — **not** as "SEIR-GNN beats SEIR-LSTM".

**Do not read the mean RMSE column on this grid.** Origin 0.40 is an outlier
where every arm fails (140–227 against 13–48 elsewhere) and it dominates every
mean and standard deviation. A variance claim was briefly drawn from it —
ASTGCN+direct sd 25.77 against ASTGCN+foi_res sd 4.27, read as the physics layer
stabilising the encoder — and retracted the same day: the physics head beats the
direct head on **1 of 9 origins**, that one, and is slightly worse on the other
eight. Read win counts and the per-origin column, which is why `stats.py` prints
wins beside every delta. See EXP-038 note 3.

The SEIR-GNN vs SEIR-LSTM result survives that scrutiny: 8/9 origins, winning at
0.40 *and* on seven of the other eight, so it is not outlier-driven.

Absolute levels on the nine origins are much higher (persistence 41.21 vs 17.86)
because they reach back to 0.40. **Never pool or pair across origin sets.**

### Two caveats that travel with the screening numbers

1. **Validation and test disagree at this spread.** LSTM+foi_res is *ahead* of
   ASTGCN+foi_res on test (34.97 vs 35.26), and the best validation arm is
   *worse than persistence* on test (37.55 vs 36.02). A −0.62 validation win
   that reverses on test is not yet a result.
2. **Three origins cannot reach significance at the `origin` unit.**

Both point at the 9-origin confirmatory grid before anything is claimed.

---

## Files

| file | what it is |
|---|---|
| `core.py` | folds, tensor building under `cd.LAGS`, scoring, early stopping |
| `models.py` | `Net`: encoder → head. `HEADS`, `BACKBONES`, `SEEDS`, `DISTS`, `LAM_PARAMS` |
| `backbones.py` | the five published architectures + Liu et al.'s LSTM, reusing `analysis/lib/reproduced.py` |
| `train.py` | one `(fold, seed)` run; losses incl. NB; records `best_epoch` / `epochs_ran` / `stopped_early` |
| `sweep.py` | run a grid across origins × seeds, one row each |
| `stats.py` | paired sign-flip permutation tests, both pairing units, BH-FDR |
| `grids.py` | the grids, one function each |
| `diagnose_foi.py` | why the physics head behaves as it does — reach, seeding, learnability, saturation, ceiling |
| `diagnose_seed.py` | scores the three initial-state seedings without training |
| `diagnose_arch.py` | retrains the remaining architectures keeping every forecast, then measures how they are wrong (EXP-039) |
| `knn.py` | k-nearest-neighbour analogue forecaster (remedy R4b); strict analogue library, leakage-tested |
| `ensemble.py` | equal-weight ensembles from `sweep.py --keep` forecasts (remedy R5) |
| `climate_lags.py` | which climate lags predict growth, on training data only (EXP-043) |
| `gbm.py` | gradient-boosted trees on growth, with or without climate lag blocks (EXP-044) |
| `augment.py` | synthetic training data (EXP-042): TimeGAN, SEIR-simulated epidemics (pre-registered and calibrated), plus LDS weights; training sets only, leakage-tested |

### A note on the seasonal feature's rationale

`core.py::seasonal_features` cites "EDA finding F9" for its week-of-year
features. **F9 was retracted** — `analysis/README.md` and `docs/EXPERIMENT_LOG.md`
record shape r = −0.065, 0/25 districts significant. The feature is nonetheless
the largest input-side effect measured here, so what needs fixing is the cited
rationale, not the feature. Do not repeat the retracted claim in paper text.
