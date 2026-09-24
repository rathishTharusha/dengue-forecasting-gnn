# Physics-informed GNN — pre-registered plan

Written before any of these arms exists in code. Branch `exp/physics-gnn`;
runs on Kaggle. This is the research's stated goal: a physics-informed way to
improve GNN-based dengue forecasting, through a structural change to the model.

## What the evidence already says about physics here

| role of physics | result | where |
|---|---|---|
| SEIR as the **decoder** (forecast routed through λ) | −7 RMSE vs direct; structural (λ=0 floor, λ only r²≈0.25 learnable) | EXP-032, EXP-034 |
| SEIR **gated** onto persistence (`foi_res`) | graph beats LSTM only in this formulation: −0.62, 9/9 units (3 origins); −0.75, 8/9 origins, raw p 0.012, p_adj 0.059 (9 origins) | EXP-035, EXP-038 |
| SEIR as an **auxiliary loss** | null (−0.04) | EXP-040 |
| **spatial physics penalty** (relative incidence smooth across borders) | **−0.044, 57/60 runs, p = 3×10⁻⁶** on STGAT — the one significant physics gain in the project | manuscript §5, `physics_final.json` |
| R̂ for **outbreak detection** | AUC 0.807 → 0.826 | manuscript §5 |
| gated SEIR models **out of sample** | best on the nine-origin test (31.47–31.70 vs direct 33.93–36.79, persistence 31.76) — observed, never selected on | EXP-038, EXP-045 note 4 |

Two structural gaps remain untested. The spatial penalty was shown on STGAT,
not on the current best model. And every SEIR head so far couples districts
only through an ad hoc "import" term on the initial state: force of infection
is held fixed within each week and never responds to infections elsewhere.
The metapopulation SEIR of MepoGNN (Cao et al. 2023) and CausalGNN (Wang et al.
2022) does that coupling in the dynamics themselves, and the harness's own
simulator (`seir_sim.simulate_closed_loop`) already implements it.

## Arms

B = AAGCN, direct head, NB likelihood, seasonal features, window 3. Frozen
three origins × seeds 0/1/2, 400 epochs, `--keep`.

| id | model | tests |
|---|---|---|
| P0 | B | control |
| P1 | B + spatial penalty, ratio form, weight 0.001, district scales = training-mean cases, binary border adjacency — exactly the manuscript's configuration | does the significant physics constraint transfer to the best model? |
| P2 | B + spatial penalty, log form, weight 0.05 (`physics_net` default) | the bounded variant |
| P3 | gated SEIR-GNN: AAGCN encoder, `foi_res` head, NB, season | the existing physics formulation on B's encoder |
| P4 | **metapopulation SEIR-GNN**: AAGCN predicts each district's weekly transmission rate β_i(t); the SEIR simulator recomputes force of infection daily as β_i · Σ_j C_ij I_j with C a learned row-stochastic coupling initialised to the border graph; incidence gated onto persistence as in P3 | the structural physics-informed change |
| P5 | P4 + P1's penalty + district seasonal transmission (the zero-initialised district × Fourier table, here modulating β) | the physics-GNN combination, declared once |
| P6 | SEIR-LSTM: Liu et al.'s LSTM encoder, `foi_res`, NB, season | the comparator the research is measured against |

## Decision rules

- **Adoption** (P1–P5 against P0): validation mean delta < 0, ≥ 7/9 wins,
  horizon-3 not raised — unchanged from every earlier plan.
- **Physics comparisons**, reported whatever the adoption outcome: P4 vs P3
  (does coupling inside the dynamics beat the import term?) and P3, P4, P5 vs
  P6 (SEIR-GNN vs SEIR-LSTM), paired on validation at both units.
- **Nine-origin confirmation**, run in parallel because the SEIR-GNN vs
  SEIR-LSTM question is the research's own and already has nine-origin
  evidence (EXP-038): P0, P3, P4 and P6 on `core.ORIGINS_9`. Endpoint as plan
  S9: **test RMSE**, family {P4 vs P6, P4 vs P0, P4 vs persistence, P3 vs P6},
  BH, p_adj < 0.05 and ≥ 6/9 origins. Validation reported alongside. The
  nine-origin test spans were used in EXP-038; that caveat stands.

## Prediction, stated before the run

P1 will be adopted if the manuscript's finding transfers; the effect will be
small (around −0.05), as it was on STGAT, and may fall short of 7/9 on nine
units. P2 will be null. P3 and P4 will trail P0 on validation by 0.5–1.0, as
gated heads did in EXP-035. P4 will not beat P3 by much: Moran's I of R is
0.005, so coupling transmission across borders has little to work with. P3/P4
will beat P6 on validation, repeating EXP-038, and on the nine-origin test
the gated SEIR arms will again sit near or below persistence while P0 does not.

## Outcome (added after the run — see EXP-046, EXP-047)

No physics arm is adopted against B. The spatial penalty keeps its sign (−0.06,
horizon 3 −0.20) but wins 6/9, short of 7/9. The metapopulation SEIR-GNN — the
structural change this plan was written for — is significantly *worse* than the
existing gated head on validation (+0.20, p_adj 0.022) and ties SEIR-LSTM on the
nine-origin test (−0.01, 4/9). Coupling districts inside the dynamics has
nothing to work with when transmission is not spatially autocorrelated
(Moran's I of R 0.005).

The **gated SEIR-GNN against SEIR-LSTM** wins 9/9 (p_adj 0.007) on the
three-origin validation screen and −0.41, 6/9, raw p 0.031, p_adj 0.125 on the
nine-origin test. The S9 endpoint is not met. It is also not consistent with
EXP-038 on a matched metric: there the graph won on validation (−0.75, 8/9) and
lost on test (+0.23, 4/9); here it ties on validation (−0.01) and wins on test.
EXP-038 used validation as its endpoint and this plan used test — a mismatch
between the two pre-registrations, recorded in EXP-047 note 2. On test the gated SEIR-GNN is the best arm of the grid
(31.06, persistence 31.76, B 33.21) while validation ranks B first — the fifth
instance of the disagreement recorded in EXP-045 note 4.
