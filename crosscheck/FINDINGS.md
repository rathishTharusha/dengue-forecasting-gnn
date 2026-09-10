# Cross-check findings

What the independent reproductions turned up. Newest section first within each
paper; each finding says how it was established and what, if anything, it
changes for this project.

**Status key:** ✅ verified by an executed notebook · ⏳ demonstrated from source
code, awaiting a full training run · ⚠️ opinion, not measurement.

---

## R1 — Weng et al. (2024), IEEE BigData

The benchmark this project reproduces, and the source of our dataset. The
authors' implementation is at `reference_repo/`, and
`reference_repo/Models/results.txt` holds the raw output behind Table I — so
these findings come from reading the code that produced the published numbers,
not from guessing at it.

### F1.0 — The reproduction succeeds, and persistence beats every model in Table I ✅

Full sweep, `QUICK_TEST = False`: 5 segments, 50 epochs, STGAT reimplemented from
`reference_repo/Models/gnn_models.py`, the reference's own hyperparameters, its
own metric convention (per-window averaging), its own protocol.

| | MAE | RMSE (per-window, as the reference reports) | RMSE (pooled) |
|---|---|---|---|
| Weng et al., Table I — STGAT | 25.38 ± 1.37 | 44.78 ± 2.26 | not reported |
| **This reproduction, published protocol** | **24.01 ± 1.32** | **42.31 ± 2.47** | 63.65 |
| *persistence, identical slices* | *18.58 ± 0.48* | *34.67 ± 0.89* | *55.59* |
| This reproduction, corrected protocol | 35.47 ± 11.79 | 62.10 ± 21.04 | 89.77 |
| *persistence, identical slices* | *21.66 ± 7.08* | *39.08 ± 12.87* | *61.08* |

**Two conclusions.**

**1. Table I reproduces.** 24.01 / 42.31 against the published 25.38 / 44.78 —
within ~5% on both metrics, with matching spread (1.32 vs 1.37; 2.47 vs 2.26).
The published numbers are real and our reimplementation is faithful. Everything
below is therefore about what those numbers *mean*, not about whether they are
what the authors got.

**2. Naive persistence beats them — on the cross-validated column.** Scored on
the identical slices, with the identical metric convention,
last-week-carried-forward gives **MAE 18.58 against STGAT's 24.01, and RMSE 34.67
against 42.31**. Against Table I's *Cross Validated* column, that places first on
both metrics against all ten models: the best reported there are STGAT at 25.38
MAE and 44.78 RMSE.

**Be precise about the second column.** Table I also reports a *Full Dataset*
column, and there the picture is mixed. Persistence evaluated the same way scores
MAE 17.87 / RMSE 33.48, which beats all ten models on MAE — but **A3TGCN (30.55)
and ASTGCN (32.41) beat persistence on RMSE**. So the correct statement is:

- *Cross Validated column* — persistence beats every model on both metrics.
- *Full Dataset column* — persistence beats every model on MAE; two GNNs beat it
  on RMSE.

That is still a serious problem for the paper's claim, because the cross-validated
column is the one it presents as its robustness evidence. But "persistence beats
everything in Table I" would be an overstatement, and this workspace should not
make it.

This needs no corrected protocol to see; it holds under the paper's own. And
persistence is not advantaged by the training-inclusive evaluation of F1.1,
because it does no training: the GNN is the one being scored partly on data it
fitted, and it still loses on the cross-validated column.

**The paper reports no persistence or naive baseline.** Its comparison set is
ARIMA, RF, XGBoost, ARNN and LSTM — all of which, per F1.2, were run on a
different task. Against the one reference that costs nothing to compute, the
central claim that spatio-temporal GNNs are the stronger early-warning system is
not supported by this dataset.

**Caveat, stated precisely:** this is STGAT only. `torch-geometric-temporal` is
not installed here, so ASTGCN, A3TGCN, DCRNN and AAGCN were not re-run — their
Table I figures are all *worse* than STGAT's, so the conclusion follows from the
published numbers for them, but it has not been independently reproduced.

### F1.1 — The "Cross Validated" column is measured on training data ✅

In every one of `run_stgat`, `run_a3tgcn`, `run_astgcn`, `run_dcrnn` and
`run_aagcn` (`reference_repo/Models/evaluation.py`):

```python
y_pred, y_truth, _, _  = infer(model, "cpu", test, m, s, "Test")   # discarded
y_pred, y_truth, ma, rm = infer(model, "cpu", full, m, s, "Full")  # appended
maes += [ma]; rmses += [rm]
```

`full` is a loader over the entire segment. Since each segment is split 70/30,
**70% of every reported evaluation set is data the model was fitted on** —
measured, per segment (R1 §2):

| Segment | Weeks | Windows | Trained on | Reported on | Held out | % of reported set seen in training |
|---|---|---|---|---|---|---|
| 0.6 | 275 | 269 | 188 | 269 | 81 | 69.9% |
| 0.7 | 321 | 315 | 220 | 315 | 95 | 69.8% |
| 0.8 | 367 | 361 | 252 | 361 | 109 | 69.8% |
| 0.9 | 413 | 407 | 284 | 407 | 123 | 69.8% |
| 1.0 | 459 | 453 | 317 | 453 | 136 | 70.0% |

The paper (§VI) describes the protocol as a 70/30 train/test split, so the
reported number is not the one the description implies.

**Why it matters:** Table I's cross-validated column is the one the paper's
headline comparison rests on.

### F1.2 — Classical baselines and GNNs solve different problems ✅

`random_forest.py`, `arima.py` and `lstm.py` all begin:

```python
data = pd.read_csv("../Data/Datasets/MLSO2_Final.csv")
state_data = data[data["region"] == "Kalutara"].reset_index(drop=True)
```

and then fit `cases[t]` from covariates at the **same** week `t`.

| | Classical baselines | GNNs |
|---|---|---|
| Districts | 1 (Kalutara) | 25 |
| Task | regression on same-week covariates | 3-step-ahead forecast |
| Input | current covariates | 3-week window of past cases |

The baselines are not forecasting, and their errors are computed over one
district's case distribution rather than all 25.

**Why it matters:** the paper's central claim — "spatio-temporal GNNs
consistently outperformed their traditional counterparts", average RMSE 38.22 vs
70.65 on the shifted data — compares these two groups directly.

**Measured** (R1 §3), holding the estimator (`RandomForestRegressor`, 100 trees,
seed 42), the data and the segments fixed, and changing *only* the task:

| Framing | MAE | RMSE |
|---|---|---|
| Reference baselines: Kalutara, same-week covariates | 82.52 ± 29.42 | 129.39 ± 54.31 |
| GNN task: 25 districts, 3-step-ahead forecast | 45.29 ± 12.18 | 125.35 ± 35.78 |

The framing alone moves MAE by a factor of 1.8 — and it moves it **against the
baselines**, because Kalutara is a high-incidence district whose errors are large
next to a 25-district average that includes many low-incidence ones. The confound
therefore runs in the direction that flatters the paper's conclusion.

(Absolute values differ from Table I because `MLSO2_Final.csv`, the covariate
file the reference baselines use, is not in this repo; the released array's 11
channels stand in. The *ratio between framings* is the finding, not the levels.)

### F1.3 — Normalization statistics span the test period ✅

`load_data` computes `mean, std` over the whole array before any split, and
`inverse_z_norm` reuses them at scoring time. On a series whose largest outbreak
sits in the later years, the test period's location and scale are known during
training.

**Measured** (R1 §4): whole-series statistics give mean 43.98 / sd 107.94;
training weeks only (first 321) give mean 47.03 / sd 115.01 — a 6.5% shift in
location and 6.1% in scale. Predictions are inverse-transformed with these, so
the reported error is expressed in a scale the test period helped define.

### F1.4 — Stated hyperparameters differ from released ones ✅

Paper §II.D: "a learning rate of 0.00001 and a weight decay of 0.000001".
`evaluation.py` line 26: `lr, decay, dropout = 1e-4, 5e-5, 0.1`. A 10× learning
rate and a 50× weight decay. R1 follows the code, since the code produced Table I.

### F1.5 — RMSE is averaged per window, not pooled ✅

`rmse += RMSE(truth, pred)` inside the loader loop, then `rmse /= n`, with
`batch_size = 1`. That is `mean_w RMSE_w`, which by Jensen's inequality never
exceeds the pooled RMSE, and the gap grows with how uneven the error is across
weeks.

**Measured** (R1 §6), using persistence so no training is involved:

| Metric | Pooled | Per-window mean (reference) | Reference lower by |
|---|---|---|---|
| RMSE | 54.84 | 33.48 | **39.0%** |
| MAE | 17.87 | 17.87 | 0.0% |

This is the largest of the five effects. On this target — median 13, max 2631 —
averaging per-window RMSE reports a number 39% below the pooled RMSE. MAE, a mean
of means, is untouched, which is why the two metrics in Table I are not distorted
equally.

### F1.6 — The released array has 11 features, not the 20 described ✅

The paper refers to "20 features at each time-step"; the released
`sri_lanka_2013-2022_shifted.npy` has 11. This does not affect the GNN
reproduction — the code sets `use_disease_only=True`, so those models see only
the case channel — but the published feature ablation cannot be reproduced from
the released data.

### What R1 means for our project ⚠️

Our Phase-1 baseline is recorded as matching the persistence floor (RMSE 45.3 vs
44.8), which has been read as underperforming against Weng's STGAT at 44.78.
**Those two numbers are not comparable**, and more importantly, the comparison
was never the right one to worry about:

- 44.78 is a per-window-averaged RMSE (F1.5), computed largely on training data
  (F1.1), under a global normalization that saw the test period (F1.3).
- Under the paper's own protocol, persistence scores MAE 18.58 / RMSE 34.67 —
  better than every model in Table I's cross-validated column (F1.0).

**So our Phase-1 result is not a shortfall against the literature; it is the same
result the literature gets, reported honestly.** A spatio-temporal GNN matching
rather than beating the persistence floor on this dataset is exactly what the
benchmark paper's own numbers show, once a naive baseline is put beside them.

Recommended:

- Cite **our own persistence floor** in the ablation table. It is the meaningful
  reference and it is stricter than Table I.
- If Table I appears in the report, cite it as prior work with a stated protocol
  difference — not as a target we failed to beat. `docs/decisions/0001` already
  makes the case for the protocol we use; this is supporting evidence for it.
- The dataset is unaffected and remains the right one to use.
- Consider this an argument *for* the three contributions: the headroom over
  persistence is genuinely open, and no published result on this dataset has
  closed it.

None of this makes the paper's qualitative direction wrong — spatial structure
may well help, and the graph framing is a real contribution. What the numbers do
not support is Table I as a bar to clear.

---

## R2 — GulMohamed et al. (2026), Scientific Reports

### F2.1 — The published results are not reproducible ✅

Not "hard to reproduce" — not reproducible from what is published. Missing:
which countries/regions and which admin level of OpenDengue, the time period,
the number of nodes, the mobility and covariate sources, the graph blend `α`
(Eq. 2), the window length `L` (Eq. 7), and every hyperparameter value (a grid
search is described; neither its space nor its outcome is given). No code is
released.

Two internal tensions compound it. Table 2 reports 56M dengue-case rows against
3.2M temperature and 1.2M mobility rows, so covariates cover at most ~6% of
cases with no stated join or imputation rule. And RMSE 6.1 against a stated mean
incidence of 34.6 (sd 215.2) implies a low-incidence subset or heavy filtering,
neither of which is described.

**Consequence:** R2 does not claim to reproduce Tables 3–5, and any number it
produces is on our Sri Lanka data.

### F2.2 — The architecture *is* specified well enough to implement ✅

Equations (2)–(14) are complete enough to build: hybrid adjacency, gravity
mobility prior, GCN spatial encoder, attention-augmented LSTM, feature fusion,
per-horizon heads, Gaussian uncertainty with 95% intervals. Implemented in
`xcheck/graph.py` and `xcheck/models.py`, and it trains.

### F2.3 — Two of the three Table 6 ablations are testable; one is not ⏳

The ablations are reported as *relative* degradations, so they survive the
missing dataset definition. The paper's ranking is mobility (+8–15%) > dynamic
graph (+6–12%) > attention (+4–8%).

| Paper's ablation | Testable here? |
|---|---|
| No mobility features | **Yes**, with a caveat — features are removed from the Eq. (9) fusion |
| No temporal attention | **Yes** — the attention module is removed from the architecture |
| Static instead of dynamic graph | **No** |

The third is not testable and R2 says so rather than reporting a number against
it. The paper's "dynamic" graph is *time-varying* — `A_t` changes week to week
with mobility. This repo has no time-varying mobility data, so our blended graph
is static too, just denser and distance-weighted. That variant therefore tests
**gravity-blended vs. binary adjacency**, which is a question about graph
density, not about dynamism.

**Caveat on the mobility ablation:** the gravity prior uses hop-distance and
uniform population, because the repo has no district populations or distances
(`docs/DATA.md` records no source). The mobility features it produces — gravity
strength and mobility-weighted neighbour incidence — are a documented proxy. A
null result is evidence about the proxy, not about real mobility flows.

**Also required before reading any delta:** more than one seed. At
`QUICK_TEST = True` there is a single seed, hence no noise estimate, and R2
refuses to draw an ablation conclusion in that mode rather than reading the sign
off a difference that may be smaller than run-to-run variance.

---

## R3 — Gopalakrishnan (2020), MTH 271 course notes

### F3.1 — Reproduces exactly, 6/6 checks ✅

As it should: the source prints all of its code. Verified: the bell-shaped
infection curve, the direction of each parameter's effect, that every equilibrium
of the basic model is disease-free, that traveller influx produces an endemic
equilibrium near 5%, that `R0 = β·s₀/γ` separates outbreak from no outbreak, and
the vaccination threshold `v* = 1 − γ/(β·s₀)`.

Value to the project: a tested SEIR integrator and a calibrated check harness
before either is pointed at R4. Note what does **not** carry over — this is a
host-only SEIR with no vector compartment; dengue needs the SEIR–SEI of R4.

---

## R4 — Phaijoo & Gurung (2018), GAMS J. Math. Math. Biosci.

The model behind Contribution (a), the physics-informed loss.

### F4.1 — Reproduces exactly, 7/7 checks ✅

`R0` agrees between the paper's closed form and the next-generation spectral
radius `ρ(FV⁻¹)` to 1.1e-16. All nine Table 1 sensitivity indices reproduce to
the paper's printed precision, by two independent routes (closed form and
central finite differences). The endemic equilibrium is positive exactly when
`R0 > 1`, as Theorem 3.1 requires.

### F4.2 — Table 1's baseline column contradicts its own indices ✅

The nine indices reproduce **only** from the §4 simulation parameters, not from
the "Baseline Values" column printed beside them in Table 1. Four parameters are
given two different values by the same paper:

| Parameter | Table 1 | §4 | Index from Table 1 | Printed index |
|---|---|---|---|---|
| `μ_v` | 0.02941 | 0.25 | −1.08539 | **−1.31823** |
| `ν_h` | 1.667 | 0.1667 | +0.001346 | **+0.000138** |
| `π_v` | 5000 | 2 500 000 | +0.5 | +0.5 |
| `β_v` | 1 | 0.375 | +0.5 | +0.5 |

`π_v` and `β_v` make no difference — their indices are `+½` for any positive
value, which is likely why this survived review. `μ_v` and `ν_h` do. Both look
typographic (`ν_h` is off by exactly a factor of ten).

**Not fatal:** the paper's conclusions — the ranking, `b` most positive, `μ_v`
most negative — hold under the §4 values that generated the indices. Use those.

### F4.3 — The §4 parameters give `R0 ≈ 0.78 < 1` ✅

By the paper's own Theorems 3.2–3.3 that is the disease-free regime. Integrating
Eq. (2.2) from a small introduced infection with those parameters, the infection
dies out — which sits badly with Figures 2–3, presented as outbreak dynamics.

**Practical consequence:** any simulation reusing the published parameters will
see the epidemic vanish. Raising `b`, raising `β_v`, or lowering `μ_v` restores
an epidemic regime; R4 uses `b = 1.2` (`R0 ≈ 1.88`) wherever an outbreak is
needed.

### F4.4 — The printed endemic equilibrium is wrong in its vector components ✅

Substituting the printed `E₁` into Eq. (2.2) leaves a residual of 9.7e-05. The
host components `s_h*`, `e_h*`, `i_h*` are exact; `e_v*` and `i_v*` are not.

The cause is visible in the formulas. As printed, `e_v*/i_v* = ε`. But
`di_v/dt = ν_v·e_v − ε·i_v = 0` forces `i_v = (ν_v/ε)·e_v`, i.e.
`e_v/i_v = ε/ν_v`. These agree only if `ν_v = 1`, and `ν_v = 0.1428`.

Solving the vector block directly, with `s_v = 1 − e_v − i_v`:

```
e_v = δ·i_h·ε / ((ε + ν_v)(ε + δ·i_h))          i_v = (ν_v/ε)·e_v
```

This gives a zero residual and matches an independent `fsolve` root to 1e-9. It
is available as `seir_sei_equilibrium(p, variant="corrected")`.

**Not fatal:** `R0`, the stability theorems and the sensitivity analysis do not
depend on `E₁`'s vector components. But anyone initializing a simulation at the
published `E₁` will not be at an equilibrium.

### What R4 means for the physics-informed loss ⚠️

The sensitivity ranking is the part to carry forward: `b` (biting rate, +1) and
`μ_v` (vector death rate, −1.32) dominate; `μ_h` and `ν_h` are negligible at
~10⁻⁴. A physics residual that spends capacity constraining `ν_h` is spending it
where `R0` cannot feel it.

The counterpart risk is already in `docs/ROADMAP.md`: `e_h` and `i_v` are
unobserved in district-week case data, so the residual can only ever be evaluated
on `i_h`. F4.3 adds a practical warning — do not adopt the published parameter
set as an initialization without checking `R0` first.

---

## Cross-implementation check

`tests/test_xcheck.py::test_agrees_with_project_metrics` asserts that
`xcheck.metrics` and `dengue_gnn.metrics` return identical RMSE, MAE, SMAPE and
masked MAPE on random heavy-tailed data. They were written independently from
the same definitions, so this passing is evidence both are correct.

**No defect has been found in our own metrics implementation.**
