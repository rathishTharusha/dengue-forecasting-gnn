# COVID covariate experiment (exploratory)

Owner: Codex. Keep separate from Claude's spatial experiment.

Question: do lagged national COVID policy and workplace mobility measurements
improve the corrected v2 SEIR-STGAT forecast? This is a forecasting association
test, not evidence that policy caused changes in dengue transmission.

Frozen before full execution: rebuilt data; three-week case window and forecast;
nine existing origins; seeds 0, 1, 2; 400-epoch budget; validation early stopping;
unchanged optimizer and architecture. Three arms: base, policy, policy_mobility.
The additional term multiplies force of infection by exp(w_policy * stringency/100
+ w_mobility * (-workplace_change/100)), with log multiplier bounded to +/-1.5.
Weights start at zero and can have either sign. Inputs use completed week i-2;
no future policy schedule, future mobility, or test-based normalization is used.

Explicit deviation from the no-fill input convention: absent external observations
have a model-level neutral effect; availability counts are recorded. Raw CSV values
remain missing. This preserves identical case windows across arms. This does not
claim measurements of zero restriction/mobility where none exist. Archived source
vintages have not been verified, so describe the evaluation as retrospective with
lagged covariates, not a demonstrated real-time forecasting system.

At early origins there are no COVID observations in training. The zero-initialized
exposure coefficients cannot learn a response there. In particular the first 2020
lockdown cannot be anticipated by fitting later COVID outcomes. Report this limit,
and coverage for every split, even if the overall result is negative.

Compare all arms on identical windows. Report mean split RMSE, MAE and bias,
per-origin differences, and exact sign-flip p-values after averaging seeds within
origins. Adjust the two new-arm versus base comparisons with Holm. This is an
exploratory follow-up after many previous experiments, not a new independent
confirmatory test. Never select a setting by looking at test scores.

Outputs: config.json (source and code hashes), runs.json/runs.csv (one row per
arm/origin/seed), NPZ predictions with exact indices, summary.csv and report.md.
The mean of split RMSEs is not a global pooled RMSE; do not mix the conventions.

Run:    
    C:/rp/v/Scripts/python.exe analysis/_build/run_covid_covariates.py --workers 2

A smoke run must use a different --out folder. Do not combine it with final results.
No GPU dependency changes are needed; each CPU worker uses one torch thread.

Pre-run verification: all 99 project tests passed, including 5 new COVID tests.
The new control equals the original v2 forward pass bit-for-bit. A two-epoch
smoke execution succeeded and is isolated from full results. Source manifest
checking found 4 identical raw files, 1 mismatched GADM boundary file, and 673
unavailable raw downloads. The boundary is not read by this experiment. All
five checked tracked model inputs match HEAD; their checksums are recorded in
input_audit.json. This is not a claim that all raw sources were re-verified.

Concurrent work: Claude has separately added run_stringency_test.py. This
experiment additionally tests workplace mobility, uses lag 2, starts exposure
weights at zero, and records historical-vintage uncertainty. Do not pool rows
from the two runners: their lags and priors differ.

Numerical tolerance for analysis: the matched zero-exposure models differ by
up to 0.000008 RMSE due to floating point training. Treat per-origin differences
smaller than 0.0001 RMSE as ties when counting wins and computing sign flips.
Retain raw scores. This tolerance was recorded after the first no-exposure split,
before outcomes for exposed-training splits were examined.

**Correction (verification, 2026-10-03).** That tolerance was measured on origin
0.50 alone and is about 1000x too small. Origin 0.55 also has zero coverage in
train, validation and test, so its three arms are the same model by construction,
yet they differ by 0.109 RMSE at seed 1: once a 1e-6 difference shifts which epoch
the `v < best - 1e-5` rule selects, the scores separate. The run-to-run floor of
this pipeline is therefore ~0.11 RMSE. This does not weaken the verdict, it
sharpens it -- the two covariate effects (+0.015 and +0.021) are roughly 5x
smaller than the floor, so they are below the resolution of the experiment rather
than merely non-significant. The 0.0001 tie threshold is retained in the saved
summary for reproducibility; read per-origin deltas under 0.11 as noise.

## GPU handoff — 2026-10-03

A separate workspace GPU runtime uses torch 2.1.2+cu121 and matching CUDA graph
extensions. The CPU environment and original result files are preserved.
A 20-epoch single-job timing test took 19.2 seconds on CPU and 10.1 seconds on
CUDA. Initial predictions differed by at most 0.02356 reported cases; the check
uses atol=0.05 cases and rtol=0.0001. CUDA training need not be bit-identical.
The speed test is not a scientific forecasting result.

The final comparison uses 72 CPU rows from origins 0.50 through 0.85 and 9 CUDA
rows for origin 0.90. Every arm and seed within each origin uses the same backend.
The extra partial CPU origin-0.90 results are preserved but are not selected for
the final table. The change was selected on timing before CUDA test results.
Three CUDA workers run the last origin. Hardware variation by origin is a
limitation to disclose, even though the within-origin comparisons remain paired.

GPU runner: analysis/_build/run_covid_gpu.py
GPU progress: analysis/results/covid_covariates_gpu/runs.json and stdout.log
Final combined results/report: analysis/results/covid_covariates_final/
Merger: analysis/_build/merge_covid_devices.py
The finalizer checks 81 unique rows and one backend per origin.

## Completed results — 2026-10-03

All 81 matched evaluations are complete. The final dataset contains 72 CPU rows
and nine CUDA rows, with one backend per origin across every arm and seed.

| Arm | Mean split test RMSE | Mean split test MAE | RMSE difference vs base | Holm p |
|---|---|---|---|---|
| base | 30.048672958656592 | 14.781874462410256 | 0.0 | — |
| policy | 30.063734972918475 | 14.82079142111319 | 0.01506201426188152 | 1.0 |
| policy_mobility | 30.0693471343429 | 14.815282415460658 | 0.020674175686306417 | 1.0 |

Neither covariate arm gives a clear improvement over the matched base model.
Both origin-bootstrap confidence intervals include zero. This conclusion applies
to these inputs, model and evaluation; it does not prove that external data can
never help. The historical-vintage, early-training coverage and mixed-hardware
limitations above remain applicable.

- [Completed report, per-origin scores and limitations](../analysis/results/covid_covariates_final/report.md)
- [Unrounded summary CSV in results/](../results/covid_covariates_summary.csv)
- [Exact configuration and source hashes](../analysis/results/covid_covariates_final/config.json)
- [Per-horizon metrics](../analysis/results/covid_covariates_final/per_horizon.csv)
- [Experiment log entry EXP-056](EXPERIMENT_LOG.md#exp-056--lagged-covid-policy-and-workplace-mobility-completed-exploratory-test)

The final folder also contains all 81 prediction files, run-level scores, input
audits and backend assignments. The report is generated from saved evaluations;
the GPU timing test is excluded from scientific scores. These additions are saved
locally and have not been committed or pushed.

## Independent verification — 2026-10-03 (Claude)

Reproduced from the saved rows rather than from `report.md`, by
`analysis/_build/verify_covid_covariates.py`. Run it to regenerate everything
below; no number here is hand-copied.

**What checks out.** All three arm means re-derive to `|diff| = 0` against the
reported values. The paired sign-flip test agrees on the verdict (policy
p = 0.883, policy + mobility p = 0.820; the small gap from the reported 0.906 and
0.844 is tie handling alone). The five input SHA-256s in `input_audit.json` match
HEAD. The full suite is 99 tests and all pass, including the five COVID tests and
the `atol=0` control proving `CovidModel` is bit-identical to v2 at zero
exposure. The GPU switch was decided on timing before its test scores existed.
The design is careful in a way that matters: forcing the encoder to `in_dim=1`
and initialising the exposure weights at zero means the covariate cannot leak in
through a changed parameter count or a shifted RNG draw.

**Correction 1 — the noise floor is ~0.11 RMSE, not 0.0001.** See the correction
note above. Consequence for the verdict: the effects are +0.015 and +0.021 RMSE,
or 0.14x and 0.19x the floor. The right claim is not "no significant effect at
p = 0.9" but **"the effect is below what this experiment can resolve"**, which is
the stronger statement and the one to put in the paper.

**Correction 2 — the fitted weights explain the ordering of the arms.** The
report gives no reason why adding mobility should be *worse* than policy alone.
The weights do:

| coverage | stringency weight | mobility weight |
|---|---|---|
| origins 0.50–0.65, no lockdown in training | **exactly 0.0000** (all 24 runs) | **exactly 0.0000** |
| origins 0.70–0.90, lockdown in training | **−0.438** mean (correct sign) | **+0.657** mean (wrong sign) |

Two things follow. First, non-identifiability is now *measured*, not argued: in
all 24 runs whose training data contain no lockdown week, both weights stay at
exactly zero, so the arm is incapable of acting on the 2020 collapse at origin
0.60 — which is precisely the window the model gets worst. Second, where the
weights can move, they oppose each other. Stringency and inverted workplace
mobility are both positive under restriction, so a −0.44 term and a +0.66 term
largely cancel. `policy_mobility` is the worst arm because it adds two collinear
covariates with opposing fitted signs: net effect near zero, variance strictly
higher.

**Numbering.** This entry is **EXP-056**. EXP-037 was already taken by the Stage
S8 seroprevalence fix on `fix/s8-seroprevalence`; the log entry and the links
above were renumbered.
