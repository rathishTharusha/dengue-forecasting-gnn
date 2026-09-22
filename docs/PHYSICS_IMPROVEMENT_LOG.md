# Physics-informed SEIR-GNN — improvement experiments

**What this is.** An autonomous experiment run against the physics-informed arm of
Stage S5. The original implementation (`analysis/_build/run_s5_seir_gnn.py`) is
untouched and is the reference in every table below. The SEIR simulator stays in
the prediction path in every variant tested; nothing here replaces the physics
with the `direct` head.

## Headline

| | validation RMSE | test RMSE | test MAE |
|---|---|---|---|
| original physics arm (`baseline`) | 28.16 | 61.12 | 26.55 |
| **corrected physics (`v2`)** | **17.40** | **35.36** | **15.71** |
| same physics, encoder frozen (`v2_mech`) | 17.31 | 34.69 | 15.37 |
| persistence (do-nothing) | - | 36.01 | 16.01 |

On the confirmatory protocol (9 disjoint origins, 0.50-0.90, 3 seeds):

| comparison | mean RMSE difference | origins won | p (sign-flip, clustered by origin) |
|---|---|---|---|
| corrected vs original physics arm | **-20.2** | 9/9 | **0.0039** |
| corrected vs persistence | +1.3 | 7/9 | 0.92 |
| corrected vs same physics without the network | +0.1 | 4/9 | 0.97 |

**Read that honestly.** The physics arm improved by a large, significant margin
over its own previous form. It is still not better than persistence overall, and
the graph network cannot be shown to add anything over the same physics with a
constant transmission rate.

## What actually made the difference

1. **Compartments are stocks, not weekly flows** (biggest single effect).
   `I = cases(t-1)/(rho N)` treats a weekly flow as the standing infectious
   population. Scaling by the weekly conversion fraction removed a systematic
   factor-of-two under-prediction: val 28.16 -> 25.28, 9/9 runs better.
2. **No susceptible depletion.** Subtracting ten years of cumulative cases from S
   assumes lifelong immunity; dengue has four serotypes. Removing it:
   val 24.03 -> 19.29, and the prediction bias disappeared (predicted mean 39.6
   against a truth of 40.5, from 17.4 before).
3. **Mass action.** `lambda = beta * I` with the network predicting `beta`,
   instead of an unconstrained bounded `lambda`: val 25.28 -> 24.03.
4. **A calibrated observation model.** Learning a bounded multiplier on the
   reporting rate rho: val 19.29 -> 17.90.
5. **Learned compartment scaling and rates.** Letting the data set the E/I stock
   factors and omega/gamma (softplus-constrained to 0.02-0.45 per day):
   val 17.68 -> 17.29. The data prefers *more* infectious and *fewer* incubating
   than the closed-form factors assume, which is what a reporting lag looks like.
6. **Squared error instead of SMAPE.** Small on its own (val -0.54) but in the
   right direction: the protocol selects and reports RMSE, and SMAPE on a
   zero-heavy district series rewards small predictions.

## What failed, and why it probably failed

| tried | result | likely reason |
|---|---|---|
| `lambda = lam_ref * exp(z)` instead of a sigmoid | no change (val 24.35 vs 24.34) | lambda was never saturated: it already spanned 21x and sat in the right range. The sigmoid was not the constraint. |
| a separate lambda per forecast week | no change once mass action is in | the SEIR rollout already carries the dynamics forward |
| within-week mass-action feedback (`sim_closed`) | worse (val 24.76) | recomputing lambda each substep amplifies the error in the initial I |
| explicit neighbour import term | no change (val 24.00 vs 24.03) | alpha stays near zero; neighbouring districts add nothing beyond a district's own history |
| a yearly harmonic on beta | worse (val 19.19 vs 17.86) | the seasonal signal is already inside the initial state |
| per-district baseline beta | no change | district differences are already carried by I and N |
| wider input projection, ERA5, NDVI | no change or worse | consistent with EXP-024: climate explains almost nothing at this horizon |
| Phaijoo & Gurung's omega, gamma | worse (val 26.98) | faster rates empty the compartments within the forecast window |
| training-only normalisation | ~1 RMSE worse than the leaky scaling | clearer inputs let the encoder overfit 3 origins; kept anyway, because the alternative leaks test weeks |

## The ablation that matters

`v2_mech` is the identical physics with the encoder frozen and a single learned
transmission rate. It scores **17.31 / 34.69**, against `v2`'s 17.40 / 35.36,
and on 9 origins the difference is not distinguishable from zero (p = 0.97).

Everything gained here came from the physics and its calibration. The graph
network is, on this data and this horizon, still not earning its place - which
agrees with the project's earlier findings that the ceiling is temporal rather
than architectural.

## Remaining hypotheses, untested

- **Origin 0.60 is where every variant loses to persistence** (64.6 against 48.7).
  That window contains the 2017 DENV-2 epidemic. A serotype-switch susceptible
  reset is the obvious mechanism, but it is hindsight knowledge and belongs in an
  `oracle_*` arm under R4, never in a forecaster.
- A negative-binomial observation model (overdispersed counts) instead of
  Gaussian or Poisson error.
- Giving the encoder something the SEIR state does not already contain -
  human movement, school terms, vector surveillance - rather than more climate.
- Joint training across districts with partial pooling of beta.

## Every variant tried, ranked by validation RMSE


| variant | family | what changed | val_RMSE | test_RMSE | test_MAE | runs | beats baseline | verdict |
|---|---|---|---|---|---|---|---|---|
| e800_mass_pw | budget | as e400_mass_pw, 800 epochs | 17.13 | 34.66 | 15.24 | 9 | 9/9 | improved |
| e400_mass_pw | budget | mass action + per-week beta + learned rho and rates, 400 epochs | 17.22 | 34.72 | 15.35 | 18 | 18/18 | improved |
| e400_mod | budget | modulated beta, 400 epochs | 17.23 | 34.78 | 15.38 | 18 | 18/18 | improved |
| ls_const |  |  | 17.29 | 34.63 | 15.34 | 9 | 9/9 | improved |
| e800_mass_pw_era5 |  |  | 17.37 | 36.33 | 16.12 | 9 | 9/9 | improved |
| b3_all |  |  | 17.37 | 36.33 | 16.12 | 9 | 9/9 | improved |
| b3_rates_era5 |  |  | 17.44 | 36.28 | 16.11 | 9 | 9/9 | improved |
| ls_mod02 |  |  | 17.65 | 35.56 | 15.8 | 9 | 9/9 | improved |
| e800_const |  |  | 17.67 | 35.69 | 15.87 | 9 | 9/9 | improved |
| mod | F | network modulates a learned baseline beta: beta0 * exp(z) | 17.67 | 35.36 | 15.7 | 9 | 9/9 | improved |
| mod_district | F | as mod, with one baseline beta per district | 17.67 | 35.36 | 15.7 | 9 | 9/9 | improved |
| n400_const |  |  | 17.68 | 35.62 | 15.86 | 9 | 9/9 | improved |
| e400_const | budget | no-network ablation, 400 epochs | 17.68 | 35.62 | 15.86 | 18 | 18/18 | improved |
| n400_mod05_era5 |  |  | 17.68 | 36.09 | 16.06 | 9 | 9/9 | improved |
| b3_rates_pw |  |  | 17.69 | 35.36 | 15.69 | 9 | 9/9 | improved |
| b3_era5 |  |  | 17.71 | 37.35 | 16.63 | 9 | 9/9 | improved |
| n400_mod02 |  |  | 17.74 | 35.63 | 15.87 | 9 | 9/9 | improved |
| mod_tight |  |  | 17.85 | 35.64 | 15.85 | 9 | 9/9 | improved |
| b2_constbeta_pw | ablation | no network, one beta per forecast week | 17.86 | 35.74 | 15.96 | 9 | 9/9 | improved |
| b2_constbeta | ablation | NO NETWORK: one learned beta, mass action | 17.86 | 35.74 | 15.96 | 9 | 9/9 | improved |
| n400_mod10 |  |  | 17.86 | 36.66 | 16.29 | 9 | 9/9 | improved |
| b2_learn_rho |  |  | 17.9 | 35.7 | 15.87 | 9 | 9/9 | improved |
| b3_rates |  |  | 17.91 | 35.73 | 15.87 | 9 | 9/9 | improved |
| b3_per_week |  |  | 17.92 | 35.71 | 15.87 | 9 | 9/9 | improved |
| ls_mass |  |  | 17.93 | 35.94 | 16.13 | 9 | 9/9 | improved |
| n400_mod05 |  |  | 17.95 | 35.81 | 15.98 | 9 | 9/9 | improved |
| e400_mod_norm |  |  | 17.97 | 36.94 | 16.43 | 9 | 9/9 | improved |
| e800_mass_pw_norm |  |  | 18.08 | 36.15 | 16.27 | 9 | 9/9 | improved |
| n400_mass |  |  | 18.08 | 36.15 | 16.27 | 9 | 9/9 | improved |
| b3_rates_pw_norm |  |  | 18.08 | 36.15 | 16.27 | 9 | 9/9 | improved |
| mod_norm |  |  | 18.15 | 37.12 | 16.49 | 9 | 9/9 | improved |
| b2_learn_rates |  |  | 18.22 | 36.27 | 16.13 | 9 | 9/9 | improved |
| b3_norm |  |  | 18.36 | 37.34 | 16.69 | 9 | 9/9 | improved |
| mod_season | D | as mod, plus a learned yearly harmonic on beta | 18.4 | 36.71 | 16.29 | 9 | 9/9 | improved |
| b2_era5 |  |  | 18.4 | 39.22 | 17.38 | 9 | 9/9 | improved |
| mod_district_season |  |  | 18.41 | 36.73 | 16.29 | 9 | 9/9 | improved |
| b2_per_week |  |  | 18.78 | 37.4 | 16.68 | 9 | 9/9 | improved |
| const_district_season |  |  | 19.19 | 38.09 | 16.82 | 9 | 9/9 | improved |
| b2_ndvi |  |  | 19.2 | 38.88 | 17.38 | 9 | 9/9 | improved |
| b2_s0_low |  |  | 19.28 | 38.4 | 17.15 | 9 | 9/9 | improved |
| mass_s_nodep | B/D | no susceptible depletion (dengue has 4 serotypes) | 19.29 | 38.4 | 17.17 | 9 | 9/9 | improved |
| b2_s0_high |  |  | 19.3 | 38.36 | 17.16 | 9 | 9/9 | improved |
| b2_dep10 |  |  | 19.3 | 38.25 | 17.09 | 9 | 9/9 | improved |
| b2_coup |  |  | 19.3 | 38.38 | 17.17 | 9 | 9/9 | improved |
| b2_smape |  |  | 19.31 | 38.53 | 17.25 | 9 | 9/9 | improved |
| b2_poisson |  |  | 19.36 | 38.57 | 17.31 | 9 | 9/9 | improved |
| mass_rho25 | D | reporting rate 1/2.5 | 19.37 | 38.4 | 17.02 | 9 | 9/9 | improved |
| mass_s_partial | B/D | 25% of cumulative infection removes S | 19.39 | 38.49 | 17.02 | 9 | 9/9 | improved |
| b2_dep50 |  |  | 19.85 | 41.07 | 18.13 | 9 | 9/9 | improved |
| b2_closed |  |  | 20.27 | 40.39 | 18.03 | 9 | 9/9 | improved |
| mass_s0_high |  |  | 21.07 | 44.64 | 19.34 | 9 | 9/9 | improved |
| mass_s0_low |  |  | 22.6 | 44.35 | 19.28 | 9 | 9/9 | improved |
| mass_learn_rates | D | learned omega and gamma | 23.76 | 48.13 | 19.83 | 9 | 9/9 | improved |
| mass_era5_mlp |  |  | 23.91 | 48.08 | 19.92 | 9 | 9/9 | improved |
| mass_era5 |  |  | 23.93 | 48.22 | 20.01 | 9 | 9/9 | improved |
| mass_learn_rho | D | learned reporting-rate multiplier | 23.95 | 49.02 | 20.36 | 9 | 9/9 | improved |
| mass_coup |  |  | 24.0 | 48.23 | 20.08 | 9 | 9/9 | improved |
| mass_norm |  |  | 24.0 | 48.16 | 20.0 | 9 | 9/9 | improved |
| mass_per_week |  |  | 24.02 | 48.27 | 20.03 | 9 | 9/9 | improved |
| mass_proj |  |  | 24.02 | 48.27 | 20.03 | 9 | 9/9 | improved |
| head_mass | D/E | mass action: lambda = beta * I, network predicts beta | 24.03 | 48.28 | 20.03 | 9 | 9/9 | improved |
| lam_per_week | C | a separate lambda for each of the 3 forecast weeks | 24.27 | 50.99 | 21.61 | 9 | 9/9 | improved |
| coup_explicit | E | additive neighbour import term on lambda | 24.34 | 51.03 | 21.66 | 9 | 9/9 | improved |
| best_A_B | A+B | state_flow + squared error | 24.34 | 51.03 | 21.66 | 9 | 9/9 | improved |
| head_exp | D | lambda = lam_ref * exp(z) instead of lambda_max * sigmoid(z) | 24.35 | 50.87 | 22.05 | 9 | 9/9 | improved |
| sim_closed | D | mass action recomputed every substep (within-week feedback) | 24.76 | 49.41 | 20.6 | 9 | 9/9 | improved |
| state_flow | B | E,I scaled by the weekly conversion fraction (stock, not flow) | 25.28 | 51.82 | 22.08 | 9 | 9/9 | improved |
| mass_rho30 | D | reporting rate 1/30 | 26.26 | 50.97 | 22.13 | 9 | 9/9 | improved |
| state_residence | B | E,I scaled by mean residence time | 26.4 | 55.96 | 23.8 | 9 | 9/9 | improved |
| mass_rates |  |  | 26.98 | 55.1 | 22.87 | 9 | 7/9 | improved |
| loss_mse | A | SMAPE -> squared error | 27.62 | 60.72 | 26.24 | 9 | 6/9 | improved |
| baseline | reference | exactly run_s5_seir_gnn.py today: sigmoid lambda, state from raw weekly cases, SMAPE loss | 28.1 | 61.09 | 26.55 | 18 | - | reference |
| epochs400_baseline | budget | his baseline with the same 400-epoch budget | 28.16 | 61.12 | 26.55 | 9 | 0/9 | failed |

## Per-origin test RMSE, key variants

| variant | 0.55 | 0.7 | 0.85 |
|---|---|---|---|
| baseline | 84.71 | 26.39 | 72.16 |
| head_mass | 59.5 | 24.59 | 60.76 |
| mass_s_nodep | 44.08 | 22.61 | 48.52 |
| b2_learn_rho | 40.15 | 21.39 | 45.57 |
| e400_const | 40.56 | 21.32 | 44.97 |
| e400_mass_pw | 38.39 | 21.24 | 44.53 |
| e400_mod | 38.36 | 21.17 | 44.81 |
| persistence | 38.95 | 22.4 | 46.69 |

## Confirmatory protocol — 9 disjoint origins (0.50 to 0.90)

| variant | val_RMSE | test_RMSE | test_MAE |
|---|---|---|---|
| baseline@9 | 47.01 | 50.13 | 23.63 |
| e400_const@9 | 26.22 | 29.9 | 14.71 |
| e400_mass_pw@9 | 26.15 | 30.04 | 14.88 |
| ls_const@9 | 25.89 | 29.86 | 14.62 |
| ls_mod02@9 | 26.3 | 29.9 | 14.71 |
