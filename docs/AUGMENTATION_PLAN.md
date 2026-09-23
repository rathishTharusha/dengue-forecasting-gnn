# Synthetic-data augmentation — pre-registered plan

Written before any generator exists. Evidence and reasoning are in
`literature/KNOWLEDGE.md` section 7; the learning curve motivating the design is
EXP-041. Branch `exp/gan-augmentation`; runs on Kaggle.

## The question

After EXP-040 found every model family making the same errors, the question
was whether a GAN could generate better training data. EXP-041's flat learning
curve already says more samples from the *same* distribution will not help. The
literature says the synthetic data that has helped epidemic forecasting came
from mechanistic simulators and helped by covering regimes absent from the real
history. So this plan tests both: the GAN the question asked about, and the
simulator the literature supports.

## Deviation from the SEIR-GNN plan, stated up front

Plan rule R3 forbids generated data. This plan deviates **for training sets
only**, at the user's request. **Validation and test windows stay real, always.**
Every generator is fitted to, or seeded from, only the fold's analogue-safe
training windows — those whose targets end before the first validation origin
(`knn.library_idx`) — so no validation or test week can reach a synthetic
sample. A leakage test enforces this.

## Arms

All arms are B (AAGCN, direct head, NB likelihood, seasonal features, window 3)
with one change. Frozen three origins x seeds 0/1/2, 400 epochs, `--keep`.

| id | change | what it tests |
|---|---|---|
| G0 | none | control (B) |
| G1 | jitter (N(0, 0.05) on log1p inputs) and magnitude scaling (a common N(0, 0.1) shift of the whole window in log1p space), resampled every batch | cheap transformation augmentation |
| G2 | **TimeGAN** (Yoon et al. 2019) fitted per fold and seed on 6-week x 25-district log1p sequences plus seasonal features; as many synthetic windows as real ones, appended to training | the GAN that was asked about |
| G2t | B trained on TimeGAN windows **only** (train on synthetic, test on real) | generator fidelity — diagnostic, never a finalist |
| G3a | **SEIR-simulated epidemics**, as many as real windows, appended to training | the simulator the literature supports |
| G3b | pretrain on the SEIR windows for 50 epochs, then train on real data as usual | the DEFSI / Osthus usage |
| G4 | label-distribution-smoothed reweighting of the NB loss (Yang et al. 2021), weights ∝ density^-0.5 | emphasise the tail without generating anything |

### Fixed a priori

- **TimeGAN:** GRU networks, 2 layers, hidden 32; 1,500 iterations for each of
  the embedding, supervised and joint phases; batch 64; the paper's loss
  weights (γ = 1, η = 10).
- **SEIR simulator** (`analysis/lib/seir_sim.py`, the harness's own SEIR): each
  synthetic window starts from the SEIR state of a random analogue-safe
  training window (its season and population too) and simulates 6 weeks.
  log λ per district and week = the training median of the inverted λ (from
  `diagnose_foi.invert`) + a national random walk + a district random walk,
  each with step sd equal to the training sd of week-to-week change in
  inverted log λ divided by √2. Reported cases are drawn negative-binomial
  around ρ·incidence·population with dispersion α = 0.2. The first 3 weeks are
  the input, the last 3 the target.
- **LDS:** histogram of training target log1p values in 50 bins, Gaussian kernel
  sd 2 bins, weights normalised to mean 1.

## Decision rules

- **Adoption:** as in `docs/REMEDIES_PLAN.md` — against G0, validation mean delta
  < 0, **and** ≥ 7 of 9 (origin, seed) wins, **and** validation horizon-3 RMSE not
  raised.
- **Confirmation:** an adopted arm runs on the nine disjoint origins against B,
  SEIR-LSTM and persistence, criteria exactly as SEIR-GNN plan S9 (test RMSE,
  BH across the family, p_adj < 0.05, ≥ 6/9 origins, same direction). The reuse
  caveat in `REMEDIES_PLAN.md` applies.
- **Fidelity is reported whatever the outcome:** G2t against G0 (TSTR), and the
  99th percentile of synthetic against real target values, so a failure of G2
  can be attributed to the generator or to the idea.

## Prediction, stated before the run

G1 and G2 will not be adopted: the learning curve is flat and GAN output loses
tails. G2t will be markedly worse than G0. G3a/G3b are the only arms with a
mechanism, and the prediction is still that they will not reach 7/9, because
EXP-040 found the remaining error unpredictable from the inputs whatever the
training set. G4 will reduce outbreak under-prediction and raise RMSE. If G3
is adopted, the prediction was wrong and that is the more interesting outcome.
