# Architectural changes — pre-registered plan

Written before any of these arms exists in code. Branch `exp/arch-improvements`;
runs on Kaggle. Asked for: changes to loss functions, encoder structure and
embeddings, informed by what is now known about these architectures.

## What has already been tested — not repeated here

| category | tried | result | where |
|---|---|---|---|
| loss | squared error on log1p / Huber / squared error on counts / SMAPE / **negative binomial** | NB best (−0.41, 9/9) — kept | EXP-035 |
| loss | label-distribution-smoothed reweighting | worse (+0.26) | EXP-042 |
| loss | SEIR force-of-infection as an auxiliary loss | null (−0.04) | EXP-040 |
| head | direct / residual / SEIR decoder / gated SEIR | decoder −7 RMSE; others within 0.1 | EXP-032..035 |
| head | per-horizon heads, temporal attention, Gaussian head | null | manuscript §5 |
| encoder | STGAT, A3TGCN, ASTGCN, AAGCN, DCRNN, LSTM | spread 0.15 among the working three | EXP-034 |
| encoder | no graph / fixed / learned / hybrid graph | graph worth 0.07 | EXP-008..009, screen |
| encoder | STID-style MLP, linear GLM, k-NN, trees | all behind AAGCN | EXP-040, EXP-044 |
| embedding | seasonal sin/cos | **−0.88** — kept | screen |
| embedding | additive district identity | +0.06, n.s. | EXP-040 |
| normalisation | reversible instance normalisation | worse (+1.27) | EXP-040 |

## What the evidence points at instead

Two bottlenecks that none of the above touched:

1. **The fusion of exogenous inputs.** In B, seasonal features and climate skip
   the encoder and meet it at a single linear layer shared by all 25 districts.
   So climate can act only linearly — EXP-043's ridge found nothing that way,
   while trees found a consistent gain — and every district is forced onto one
   national seasonal curve, although Sri Lanka's two monsoons reach districts
   months apart. An additive district identity (tried) shifts levels; it cannot
   change the curve's shape.
2. **A national common component.** EXP-039 found residuals correlated +0.12
   between *any* two districts, not only neighbours. Graph convolution passes
   information between neighbours; no layer gives a district the national picture.

## Arms

B = AAGCN, direct head, NB likelihood, seasonal features, window 3. Frozen
three origins × seeds 0/1/2, 400 epochs, `--keep`.

| id | change against B | bottleneck | reference |
|---|---|---|---|
| A0 | none | control | — |
| A1 | residual head (forecast = last week + correction) under the NB likelihood | the one head × loss pairing not yet run | EXP-035 ran residual only under squared error |
| A2 | **district-specific seasonal curves**: a learned (district × 4 Fourier terms × horizon) coefficient table added to the output, initialised to zero | 1 | two monsoons |
| A3 | **nonlinear head**: 2-layer MLP (hidden 64) in place of the shared linear output | 1 | EXP-043 trees vs ridge |
| A4 | nonlinear head + climate blocks at lags 2–5, 6–9, 10–13 | 1 | EXP-043/044 |
| A5 | **global context**: each district's encoding is concatenated with the mean encoding over all districts (a virtual node) | 2 | virtual nodes, Gilmer et al. 2017 |
| A6 | A3 + A2 + A5 + climate blocks + 16-d district identity — declared now as the single combination | 1 + 2 | — |

## Decision rules

Unchanged: adoption needs validation mean delta < 0 against A0, ≥ 7/9
(origin, seed) wins, and validation horizon-3 RMSE not raised. An adopted arm
goes to the nine-origin confirmation under the S9 criteria against B,
SEIR-LSTM and persistence; the test-set reuse caveat applies.

## Prediction, stated before the run

A1 will match A0 (residual and direct tied under squared error). A2 is the arm
most likely to be adopted, with a gain under 0.3. A3 alone will not help; A4
will beat A3 if the trees' gain came from nonlinearity. A5 will be small either
way, since the national component is a contemporaneous shock that past inputs
do not reveal. A6 has the most capacity and the highest risk of overfitting a
three-week window.
