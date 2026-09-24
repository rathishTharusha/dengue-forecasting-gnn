# Is the SEIR head's rescue physics or the anchor? — pre-registered plan

Written before the grid runs. Branch `exp/rescue-control`; runs on Kaggle.

## The observation being tested

In `real.json` (EXP-034) three published encoders fail on the direct head and
recover completely behind the gated SEIR head (`foi_res`), paired over 9
(origin, seed) units:

| encoder | direct val | foi_res val | delta val | delta test |
|---|---|---|---|---|
| STGAT | 23.32 | 17.62 | −5.70, 9/9 | −24.99, 9/9 |
| A3TGCN | 28.63 | 17.68 | −10.95, 9/9 | −24.90, 9/9 |
| DCRNN | 34.89 | 17.67 | −17.22, 9/9 | −38.23, 9/9 |

`foi_res` differs from `direct` in three ways at once: it forecasts a departure
from **last week's value** (the anchor), it shrinks that departure through a
**learned gate** initialised at sigmoid(−2) = 0.12, and the departure is
produced by the **SEIR simulator**. Only the third is physics. This grid
separates them.

## Arms

All six encoders × four heads, configuration identical to `grids.real`
(squared error on the scaled target, no seasonal features, `lam_param=log`,
`state_fit=True`, 300 epochs), frozen three origins × seeds 0/1/2. Every arm is
rerun here, because Kaggle and local runs must not be paired.

| head | anchor | gate | SEIR |
|---|---|---|---|
| `direct` | – | – | – |
| `residual` | ✓ | – | – |
| `gated` (new; `foi_res` with the simulator removed) | ✓ | ✓ | – |
| `foi_res` | ✓ | ✓ | ✓ |

## Decision rules

- **Primary**, per failing encoder (STGAT, A3TGCN, DCRNN): `foi_res` vs `gated`
  on validation RMSE, origin_seed unit, BH across the three. The rescue is
  credited to physics only if mean delta < 0, ≥ 7/9 wins and p_adj < 0.05.
- **Share of the rescue** carried by anchor + gate:
  (direct − gated) / (direct − foi_res), reported per encoder.
- `residual` vs `gated` shows what the gate adds on top of the anchor.
- The working encoders (AAGCN, ASTGCN, LSTM) are reported under the same rules
  as context; nothing is being rescued there.
- Test RMSE is reported beside every validation number and decides nothing.
  With three origins nothing here can reach significance at the origin unit;
  this is a mechanism check, not a confirmation.

## Prediction, stated before the run

Anchor + gate carry most of the rescue (share > 0.8), and `foi_res` vs `gated`
is within ±0.3 on every failing encoder — the SEIR simulator is not the reason
those encoders recover. `residual` rescues less than `gated`, because without
the gate a failing encoder's correction reaches the forecast unshrunk. If this
is wrong and `foi_res` beats `gated` 7/9 or better, the gated SEIR head is a
physics-informed fix for unstable graph encoders, the cleanest physics result
in the project.

## Outcome (added after the run — see EXP-048)

The rescue is the anchor, not the physics. Anchor + gate carry 96–112% of the
validation repair on all three failing encoders, and the SEIR simulator earns
credit on none: STGAT is significantly *worse* with it (+0.67, 2/9,
p_adj 0.047), A3TGCN ties (−0.00) and DCRNN's −0.68 wins only 3/9. On the working
encoders the simulator costs 0.74–0.88 (0/9). Prediction check: the share was
right; the ±0.3 band and the residual-vs-gated ordering were wrong. On test the
SEIR head does beat its twin on the failing encoders (A3TGCN −1.65, DCRNN −3.33,
9/9) — the recurring validation/test disagreement, recorded and not acted on.
