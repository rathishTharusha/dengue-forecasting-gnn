# The frozen protocol

**Fingerprint `e9afbdfb0528`.** One command checks whether a run belongs to it:

```
python analysis/_build/protocol_check.py          # the frozen protocol
python analysis/_build/protocol_check.py --all    # find which setup a number came from
```

Anyone reporting a number for this project runs that first and pastes the output.

## The settings

| | |
|---|---|
| dataset | `rebuilt` |
| protocol | `9origin` |
| origins | 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90 |
| window → horizon | 3 → 3 |
| seeds | 0, 1, 2 |
| normalisation | train split only |
| metric | mean of per-origin split RMSE, raw counts |
| seed aggregation | **average seeds within an origin, before any test** |
| cluster unit | **origin** (9 clusters) |
| test | exact paired sign-flip, two-sided |
| multiplicity | Holm across arms sharing a baseline |

## The check value

Persistence — repeat the last observed week — has no parameters and no
randomness. Two people on the same protocol get **exactly** the same number, so
it settles a disagreement before any model is discussed.

```
origin 0.50  20.2707     origin 0.70  15.7118     origin 0.85  30.9637
origin 0.55  45.4928     origin 0.75  29.5488     origin 0.90  32.8591
origin 0.60  48.7141     origin 0.80  21.9677
origin 0.65  11.3405                              MEAN        28.5410
```

**If your persistence is not 28.5410, you are not on this protocol.** What the
other values mean:

| persistence | that run is |
|---|---|
| **28.5410** | the frozen protocol |
| 36.0157 | `rebuilt`, **3 origins**, H=3 |
| 38.4763 / 39.2928 | `original` / `reordered`, 9 origins, H=3 |
| 44.7953 / 48.2652 | `original` / `reordered`, 3 origins, H=3 |
| 36.0361 | `rebuilt`, 9 origins, **H=6** |

## Clustering: the rule that is easiest to get wrong

Three seeds at one origin share one test window. They differ only in random
initialisation, so they are **one** piece of evidence, not three. Average them
within the origin, then test across the nine origins.

Counting origin x seed as the cluster unit inflates significance, and the
arithmetic shows how much — the smallest two-sided p an exact sign-flip test can
return:

| clusters | smallest attainable p |
|---|---|
| 3 origins | **0.25000** |
| 6 | 0.03125 |
| 9 origins | 0.00391 |

So a run on 3 origins **cannot** produce p < 0.25, whatever it reports. A table
showing "9/9 wins, p = .007" on a 3-origin protocol has counted 3 origins x 3
seeds as 9 independent clusters, and its p-value is not valid.

## Recording a run

Every run writes its own CSV with one row per (arm, origin, seed), and an entry
in `docs/EXPERIMENT_LOG.md` giving the script, the commit, the config and the
verdict. A number without a reproducible config does not go in the report.
