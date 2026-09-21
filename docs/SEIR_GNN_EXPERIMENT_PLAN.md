# SEIR-GNN — experiment plan (pre-registered)

**Question.** Liu et al. (2025) predict dengue in Sri Lanka with an SEIR model
whose force of infection λ comes from an LSTM. Replace the LSTM with our
spatio-temporal GNNs, predicting λ for every district from its own history and
its neighbours'. Two questions follow. Does it forecast reported cases better
than the best corrected baseline? Does λ warn earlier about the districts heading
into an outbreak?

This file fixes the stages, arms, endpoints and decision rules **before** any
SEIR-GNN result exists. Changing any of them after seeing test numbers must be
logged as a deviation in `docs/EXPERIMENT_LOG.md`, with the reason.

---

## Rules that apply to every stage

**R1 — Data.** Every model input comes from `analysis/lib/corrected_data.py`
(`load()` / `windows()`), built from the sources in `docs/DATA_PROVENANCE.md`.
The original `.npy` climate channels are never used; `features.load_multivariate`
refuses them.

**R2 — No future information.** A forecast at the start of week *i* reads cases up
to *i*−1, climate up to *i*−2, and the NDVI and population rows for week *i*
(already as-of). No lag is ever chosen because it correlates well on test data.
Lag choices for the model are fixed here. **Future lags are not allowed at any
correlation.** `tests/test_no_future_leakage.py` must pass before every run, and
any new input path gets its own leakage test.

**R3 — No generated data.** Nothing is interpolated, imputed, augmented or
synthesised into the training or evaluation data. Weeks with no source report
stay NaN; windows touching them are dropped from inputs, targets and scoring.
The single exception is S3's twin experiment. It is a *code test* on simulated
data, kept in its own folder, never mixed with real data and never reported as a
forecasting result.

**R4 — Hindsight-only knowledge is labelled.** The 2022–23 seroprevalence survey
is used for validation only. Serotype-switch dates may be used only in an arm
named `oracle_*`, which is never a finalist and never compared as a forecaster.

**R5 — Protocol.** The frozen protocol runs on `rebuilt` (559 weeks):
- rolling origins 0.55 / 0.70 / 0.85;
- 3 seeds;
- window 3 → horizon 3;
- normalisation from training data only;
- early stopping on validation;
- metrics RMSE, MAE, SMAPE and MAPE(≥1), overall and per horizon, pooled over windows.

**R6 — Selection on validation only.** Hyper-parameters, formulation choices and
finalists are picked on **validation** RMSE. Test numbers are written to disk but
not compared until an arm's configuration is frozen and committed.

**R7 — Records.** Every run writes a CSV with one row per (origin, seed, horizon, window
set), never pre-aggregated, to `analysis/results/seir_gnn/<stage>/`. It also
gets an `EXPERIMENT_LOG.md` entry with commit SHA, config, seeds, origins and the
unrounded table.

**R8 — Compute and git.** Anything over ~15 min runs on Kaggle. Work on branch
`exp/seir-gnn`. **Commit, never push**; the user pushes before a kernel needs new
code. Never commit `papers/`, `datasets/`, `reference_repo/`, `data/raw/` or
credentials.

---

## S0 — Preconditions (local, ~10 min)

1. `pytest` passes, including `test_no_future_leakage.py` (92 tests at the time of writing).
2. `python analysis/_build/source_manifest.py --check` shows 0 files different.
3. `analysis/results/source_verification/wer_all_verification.json`: every
   report compared, every difference explained.

**Gate G0:** all three hold. Otherwise stop and report.

## S1 — Corrected baseline to beat (Kaggle)

1. Run the existing kernels `corrected-benchmark-{a3tgcn,stgat,astgcn,aagcn,dcrnn}`
   (branch `feat/seir-gnn`, already generated), then
   `python analysis/_build/compare_corrected.py`.
2. Add `--dataset rebuilt` to `run_beat_baseline.py`, reusing
   `run_corrected_benchmark.build_folds_masked`. Run the **9 disjoint origins × 3
   seeds** evaluation for persistence and the five architectures (`base` arm).
   This is the confirmatory protocol used in S9. Three origins give three
   independent observations, and no clustered test can reach significance with
   three.

**Outputs:** baseline table on `rebuilt` (3-origin and 9-origin); `B*` is the
architecture with the lowest **validation** RMSE, together with its test RMSE.
Persistence's `rebuilt` floor is already known: 36.016.

**Gate G1:** `original` reproduces the earlier benchmark numbers (±0.5 RMSE). If it
doesn't, stop; the training loop has changed.

## S2 — Causal covariate check (local, descriptive only)

For each district, cross-correlate log1p cases with each ERA5 channel and NDVI
over **training weeks of the first origin only**, at lags 2–20 (NDVI 0–20).

This check is descriptive. It does **not** pick lags. The model's covariate
window is fixed at `covariate_window = 4` under R2, whatever the correlations
show. The check documents what causal signal exists, so a null climate result
can be interpreted.

## S3 — SEIR core twin experiment (local, CPU, synthetic — code test only)

1. Choose a known λ(t) per district (seasonal sinusoid plus a spatial wave through
   the adjacency). Simulate with `seir_sim.simulate_weeks`, draw reported cases as
   Poisson(ρ·incidence), and save everything to `analysis/results/seir_gnn/s3_twin/`
   tagged `synthetic`.
2. Train each formulation from S4/S5 on the synthetic data.

**Gate G3:** the learned λ correlates with the true λ at r ≥ 0.9, and reported-case
RMSE is within 10% of the Poisson noise floor. If not, λ is not identifiable in
that formulation; fix it or drop it before touching real data.

## S4 — SEIR-LSTM reproduction on corrected data (Kaggle)

Liu et al. (2025), made causal.

| Component | Setting |
|---|---|
| States | fractions S, E, I, R per district; counts = fraction × previous-year population |
| Dynamics | `seir_sim` exponential-flow integration, 7 daily substeps (Liu's weekly Euler step is unstable) |
| ω, γ | ω = 0.7 / week, γ = 1 / week (Liu); sensitivity: ω = 7/5.9, γ = 7/4.5 |
| λ head | λ = λmax · σ(W·h), λmax ∈ {1, 2, 4} / week chosen on validation |
| Reported cases | ρ · incidence, ρ = 1/11 (Liu); sensitivity ρ ∈ {1/2.5, 1/30} |
| S₀ | 1 − 0.682 (suburban Colombo, all ages, 2013–14); sensitivity 1 − 0.514, 1 − 0.908 |
| Serotype reset | **none** in the primary arm; `oracle_reset2017` separate and labelled |
| Loss | SMAPE on observed non-missing weeks (Liu); MSE on log1p as an alternative, chosen on validation |
| Inputs to h | cases window [i−3, i−1], ERA5 [i−5, i−2], NDVI as-of [i−3, i] |

Two formulations are compared on **validation**:

- **F-seq (Liu-faithful).** One rollout through the series. At each week, λ comes
  from the inputs available then, and the SEIR state is carried forward. A
  forecast from origin *i* continues the rollout for 3 weeks, and the head emits
  λ for *i*, *i*+1 and *i*+2 from inputs up to *i*−1. Through a missing week the
  state keeps evolving under the last computed λ; that week gets no loss, no
  score and no assimilation. This is model behaviour, not a data value.
- **F-win (window state).** Each window rebuilds its state from past reported cases
  only:
  - I and E from reported cases in weeks *i*−1 and *i*−2, divided by ρ;
  - S = S₀ − (cumulative reported cases up to *i*−1) / (ρ·N).

  Windows are then independent, which matches the frozen windowed protocol.

**Gate G4:** the formulation with lower validation RMSE, F*, is frozen for S5.
Report SEIR-LSTM test RMSE against persistence and `B*`; it is a reference arm,
with no requirement to win.

## S5 — SEIR-GNN arms (Kaggle)

The SEIR-LSTM with the LSTM replaced by a GNN encoder (`reproduced.build`),
formulation F*.

**Screen**, on A3TGCN only, 3 origins × 3 seeds, selected on validation:

| Factor | Levels |
|---|---|
| Inputs | `cases` · `cases+era5` · `cases+era5+ndvi` |
| Spatial coupling | `implicit` (message passing inside the encoder only) · `explicit` (λᵢ = λᵢ(head) + α·Σⱼ Âᵢⱼ Iⱼ: an import term from neighbours' infectious fraction, Â the row-normalised adjacency without self-loops, α ≥ 0 learned) |
| Head | `foi` (λ → SEIR → cases) · `direct` (same encoder predicts log1p cases; the no-physics control) |

That is 12 arms. The `direct` head is the control that isolates the value of the
SEIR layer from the value of the covariates.

**Expand:** the best 2 SEIR configurations from the screen (on validation), plus
their `direct` controls, across all five encoders (STGAT, ASTGCN, AAGCN, DCRNN,
A3TGCN).

**Gate G5:** freeze the finalist `S*`, the configuration with the lowest validation
RMSE, and commit its config. Only then compare test numbers.

## S6 — Sensitivity and oracle arms on S* (Kaggle)

Each change is made one at a time:
- ρ ∈ {1/2.5, 1/11, 1/30};
- S₀ ∈ {0.318, 0.486, 0.092};
- ω and γ alternatives;
- state assimilation at origin, on or off (rescale E and I so the model's week *i*−1 reported cases equal the observed value);
- `oracle_reset2017`.

These arms are reported as robustness evidence and never replace S*.

## S7 — Early-warning evaluation (local, from saved predictions)

**Outbreak definition** (per district, per origin, from **training weeks only**): a
week is an outbreak when cases exceed the training mean + 2 SD for that district.

| Metric | What |
|---|---|
| POD, FAR, F1 per horizon | alert when a forecast exceeds the threshold |
| Lead time | weeks between the first alert and the first observed outbreak week in each episode |
| AUC | λ (or forecast) at origin *i* as a score for "outbreak in *i*..*i*+2" |
| Precision@5 | top-5 districts by λ vs top-5 by observed incidence per 100k over *i*..*i*+2 |

Compare S*, `B*`, the `direct` control and persistence on the same windows.

## S8 — Seroprevalence validation (local)

1. Take S*'s implied cumulative infected fraction per district at the survey
   midpoint (2022-12), under ρ = 1/11.
2. Compare it with the nine-district IgG seroprevalence (all ages): Spearman
   correlation, and the sign of each district's error.
3. State the mismatch plainly. The survey measures ages 10–20 and lifetime
   exposure; the model tracks the population since 2013.

This stage is descriptive only.

## S9 — Confirmatory test and write-up

1. **Primary endpoint:** S* against `B*`, test RMSE on `rebuilt`, **9 disjoint
   origins × 3 seeds**. Use a paired sign-flip permutation test **clustered by
   origin** (seed-averaged per origin), two-sided.
2. **Family**, with Benjamini–Hochberg at q = 0.05:
   - S* vs `B*`;
   - S* vs persistence;
   - S* vs its `direct` control;
   - S* vs SEIR-LSTM;
   - early-warning AUC of S* vs `B*`.
3. **A "beats the baseline" claim needs all three:**
   - BH-adjusted p < 0.05;
   - a win on ≥ 6/9 origins;
   - the same direction in the frozen 3-origin table.

   Otherwise the result is reported as not shown, with the effect size and CI.
4. Append the ablation table, the per-row CSVs and the log entries. Record in
   plain words which rows use oracle or validation-only knowledge.

---

## Stage outputs and stop points

| Stage | Where it runs | Output | Stop and report after? |
|---|---|---|---|
| S0 | local | checks | yes if any gate fails |
| S1 | Kaggle | `results/seir_gnn/s1_baseline/`, `B*` | **yes** (user pushes, runs kernels) |
| S2 | local | `s2_causal_ccf/` figures | no |
| S3 | local | `s3_twin/` | **yes** |
| S4 | Kaggle | `s4_seir_lstm/`, F* | **yes** |
| S5 | Kaggle | `s5_seir_gnn/`, S* | **yes** |
| S6 | Kaggle | `s6_sensitivity/` | no |
| S7 | local | `s7_early_warning/` | no |
| S8 | local | `s8_seroprevalence/` | no |
| S9 | local + Kaggle | final table, log | **yes** |
