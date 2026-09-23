# Climate lags and the time window — pre-registered plan

Written after the training-only lag analysis (EXP-043) and before any neural
run with longer climate lags. Branch `exp/climate-lags`; runs on Kaggle.

## Why

Every model so far has seen climate only at lags 2–4 weeks: `core.build_tensors`
slices a 3-week climate window ending at the ERA5 release delay. EXP-036 varied
the *case* window and held climate there. Dengue's biology runs longer —
breeding sites, larval development, extrinsic incubation — and EXP-043 found,
inside each fold's training data only:

- a biologically coherent lag pattern: wetter-than-usual conditions ~4 weeks
  back go with rising cases, 8–13 weeks back with falling cases; warmth 10–20
  weeks back with rising cases — all weak, |r| ≤ 0.11;
- **no** stable out-of-sample gain through a linear model (≤ +0.006 R², sign
  flipping between folds), and none from longer case windows;
- a **consistent** out-of-sample gain through gradient-boosted trees: raw
  climate at lags 2–5, 6–9, 10–13 adds **+0.028 R² of growth, positive in 3/3
  folds** (6 blocks to lag 25: +0.027, 3/3).

Raw climate beating anomalies suggests the signal may be district-specific
seasonality — Sri Lanka's two monsoons reach different districts at different
times, which one national seasonal curve cannot represent.

## Arms

B = AAGCN, direct head, NB likelihood, seasonal features, window 3. Frozen three
origins × seeds 0/1/2, 400 epochs, `--keep`. Lag blocks are 4-week means of
the six ERA5 channels, standardised with training statistics; no lag below 2
is ever read.

| id | change against B | tests |
|---|---|---|
| K0 | none | control |
| K1 | climate at lags 2–4 (the existing `use_climate` path) | the current climate input |
| K2 | raw climate blocks 2–5, 6–9, 10–13 | EXP-043's best consistent block |
| K3 | raw climate, six blocks 2–5 … 22–25 | longest range |
| K4 | climate *anomaly* blocks 2–5 … 22–25 (training climatology) | seasonality removed |
| K5 | case window 8 + K2 | the time-window question, combined with longer climate |
| K6 | gradient-boosted trees on the EXP-043 features + K2 blocks (no network) | the model that found the signal, on real validation |
| K7 | K6 without climate | isolates climate's contribution to the trees |

K6/K7 predict log growth per horizon, converted to counts with Duan's smearing
factor estimated on training residuals (so they target the mean, as RMSE
requires). Hyperparameters fixed as in EXP-043.

## Decision rules

Unchanged from `docs/REMEDIES_PLAN.md`: adoption needs validation mean delta
< 0 against K0, ≥ 7/9 (origin, seed) wins, and horizon-3 RMSE not raised. K6/K7
are deterministic, so their seeds are identical; they are adopted on the same
rule. An adopted arm goes to the nine-origin confirmation with the S9 criteria;
the test-set reuse caveat in `REMEDIES_PLAN.md` applies.

## Prediction, stated before the run

K2 is the most likely arm to be adopted; if it is, the gain will be small
(well under 1 RMSE). K1 will match K0, as it did in EXP-035. K5 will not beat
K2, because longer case windows lowered growth R² in EXP-043. K4 will trail K3.
K6 will not beat K0 overall — trees fit the persistence baseline worse than the
network — but K6 will beat K7, reproducing the climate gain on real validation.

## Outcome (added after the run — see EXP-044)

No arm adopted; longer lags make the network progressively worse (K3 +0.30,
K4 +0.62, both significant). The prediction was wrong about K2 and right about
the rest. The trees' climate gain reproduced in direction (−0.49, 2/3 origins).
The likely reason the network cannot use climate is architectural: exogenous
inputs reach only a linear output layer shared by all districts. Followed up in
`docs/ARCH_PLAN.md`.
