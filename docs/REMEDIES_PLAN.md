# Remedies from the literature — pre-registered plan

Written before any of these arms has been run. Changing an arm, a metric or a
decision rule after seeing results is a deviation and goes in
`docs/EXPERIMENT_LOG.md` with the reason. Evidence for each hypothesis is in
`literature/KNOWLEDGE.md`; the diagnosis motivating them is EXP-039.

Branch: `exp/literature-remedies`. Runs on Kaggle via
`scripts/build_seirgnn2_kernel.py --grid remedies`.

## Starting point

The reference arm **B** is the best configuration found so far: AAGCN, direct
head, negative-binomial likelihood, seasonal features, window 3
(`combo.json`: validation RMSE 15.66 on the frozen three origins).

EXP-039 says the three working encoders (ASTGCN, AAGCN, LSTM) have residual
correlations of 0.98–0.99 and leave only ~5% of their residual linearly
predictable. The plan follows from that: architecture changes on the same
inputs are expected to do little, so every remedy below changes something
*other* than the encoder.

## Hypotheses

| id | change against B | hypothesis | literature |
|---|---|---|---|
| R1a | `norm="revin_mean"`: subtract each district-window's mean log level, add it back to the output | reduces error under regime shift | RevIN (Kim 2022) |
| R1b | `norm="revin"`: also divide by the window's spread, with learned affine | as R1a; may be noisy at window 3 | RevIN |
| R2 | `node_emb=16`: a learned 16-d identity per district, joined at the head | lets one model fit 25 different districts | STID (Shao 2022) |
| R3 | STID: MLP (`backbone="none"`) + district identity + seasonal identity | a simple identity model matches or beats the graph models | STID, BasicTS |
| R4a | NB-GLM: linear head on the same inputs + district identity, NB loss | a statistical model is competitive | dengue NMA 2026, DLinear |
| R4b | k-nearest-neighbour analogue forecast (no training) | top-ranked dengue method in the NMA; errors unlike the neural models' | dengue NMA 2026 |
| R5 | ensemble of B with the best non-neural member (post hoc, equal weights) | beats both members, *because* their errors differ | Johansson 2019, Cramer 2022 |
| R6 | SEIR-GNN as a regulariser: B plus an auxiliary force-of-infection head whose SEIR incidence is fitted jointly (weight w ∈ {0.1, 0.3}); the forecast stays on the direct head | physics as a constraint helps where physics as a decoder (the `foi` head) did not | EINN (Rodríguez 2023) |

R6 is the proposal's own contribution restated in the form the literature
supports. It is the arm that answers "does SEIR structure help a GNN".

## Protocol

- **Screening.** Frozen three origins × seeds 0/1/2, window 3 → horizon 3, the
  `seirgnn2` harness unchanged otherwise. Every arm keeps its forecasts
  (`--keep`) so R5 can be formed post hoc.
- **Selection** on validation RMSE only (plan R6 of the SEIR-GNN plan).
- **Adoption rule.** A remedy is adopted into the finalist if, against its
  matched control, it improves validation RMSE with mean delta < 0 **and** wins
  on ≥ 7 of 9 (origin, seed) units, **and** does not raise horizon-3 RMSE.
  Several adopted remedies are combined once, not searched over.
- **Confirmation.** The frozen finalist then runs on the nine disjoint origins
  (`core.ORIGINS_9`) against B, SEIR-LSTM (`LSTM+foi_res`) and persistence.
  Endpoint, family and criteria exactly as SEIR-GNN plan S9: **test RMSE**,
  paired sign-flip clustered by origin, BH across the family, a win needs
  p_adj < 0.05 **and** ≥ 6/9 origins **and** the same direction as screening.

## Known limitation, stated in advance

The nine-origin test spans were already used to evaluate five arms in EXP-038,
so they are no longer untouched. They have not been used to *select* anything,
and the arms here have never been evaluated on them, but a reviewer is entitled
to discount a win on reused data. **The only fully clean confirmation is data
after the series ends (February 2024).** Weekly district counts for 2024–2026
from the Epidemiology Unit's weekly reports would give that, and would be
worth more than any further modelling.
