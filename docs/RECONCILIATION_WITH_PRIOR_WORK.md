# Reconciling our results with Weng et al. (2024) and DengueGNN (2026)

**Date:** 2026-09-03 · **Prompted by:** an apparent contradiction — published work
reports GNNs outperforming baselines on this dataset, while we find they lose to a
naive persistence forecast.

**Conclusion: there is no contradiction.** The two sets of numbers are computed
with different aggregations, and once put on the same footing the published models
also lose to persistence — a baseline none of them tested.

---

## 1. Our harness is not the problem

First check, because it had to be ruled out before anything else. Persistence was
recomputed directly from `sri_lanka_2013-2022_shifted.npy` with no pipeline
involvement — no windowing code, no normalisation, no inverse transform:

```
independent computation, our 8 folds : 55.12
harness reports                      : 55.12
```

Exact agreement. Our evaluation is sound.

---

## 2. DengueGNN is not comparable

`DengueGNN Graph-based deep.pdf` reports RMSE 6.1 for one-week and 10.9 for
four-week prediction. Those are on **OpenDengue**, a different dataset at a
different scale, not the Sri Lanka series. It is a useful methodological reference
— dynamic adjacency, temporal attention, Moran's I — but its numbers cannot be
compared to ours and do not bear on this question.

`09_SEIR_model.pdf` is a Portland State teaching notebook on the single-population
SEIR model for influenza, not a dengue paper. See `docs/STAGE3_EXPERIMENTS.md` §1.1.

---

## 3. Weng et al. is the real comparison, and the aggregation differs

Their protocol (paper §VI): rolling cross-validation over 5 nested segments —
60%, 70%, 80%, 90%, 100% of the data — each split 70/30 train/test.

We reimplemented that protocol exactly and computed persistence on it. The answer
depends entirely on how RMSE is aggregated:

| Aggregation | Persistence RMSE |
|---|---|
| Pooled over all district-weeks | **58.79** |
| Mean of per-district RMSEs | 44.59 |
| **Mean of per-window RMSEs** (theirs) | **38.46** |

The spread is a factor of **1.53×** between the first and last. On a target with
median 13 and maximum 2,631, this is not a rounding detail: a handful of outbreak
windows dominate a pooled RMSE, while averaging per-window RMSEs gives each window
equal weight regardless of magnitude.

### Which one do they use? Their code settles it

`reference_repo/Models/evaluation.py` is their released implementation:

```python
batch_size = 1                      # line 23
...
for i, batch in enumerate(dataloader):
    ...
    rmse += RMSE(truth, pred)       # RMSE of this one window
    n += 1
rmse /= n                           # mean of per-window RMSEs
```

with `RMSE = sqrt(mean((pred - truth)**2))` over a single window of 25 districts
× H horizons. So their reported figures are **means of per-window RMSEs**, and
because `sqrt` is concave that is systematically below the pooled RMSE of the same
predictions.

---

## 4. On their own footing, none of their models beats persistence

Persistence, computed by their code's aggregation on their own CV segments, is
**38.46**. Against their Table I:

| Model (Weng et al., shifted, CV) | Reported RMSE | vs persistence 38.46 |
|---|---|---|
| STGAT | 44.78 | loses |
| ASTGCN | 47.72 | loses |
| AAGCN | 55.83 | loses |
| A3TGCN | 58.52 | loses |
| DCRNN | 71.61 | loses |
| Random Forest | 84.66 | loses |
| LSTM | 131.36 | loses |
| ARIMA | 189.22 | loses |

**Every model in their table loses to a naive persistence forecast.** They do not
report persistence, so this was not visible in the original paper.

This is the strongest form of our result: it does not depend on our architecture,
our implementation, or our protocol. It follows from *their* published numbers,
*their* protocol and *their* evaluation code.

---

## 5. What this changes

**Our central claim is confirmed and strengthened.** The apparent contradiction was
an artefact of comparing a pooled RMSE (ours) against a mean-of-per-window RMSE
(theirs). Corrected, both point the same way.

**It also sharpens what we may claim.** Our own experiments test *our* GCN
variants, not STGAT or A3TGCN. We have not run those architectures, so we cannot
say from our own runs that published spatio-temporal GNNs lose to persistence. We
can say it from *their* numbers, which is a different and better-supported
argument. The paper must make that distinction explicitly.

**A caveat that cuts against us.** Absolute RMSE values are not comparable across
the two papers. Our 55.12 (pooled) and their 44.78 (per-window mean) are different
quantities. Any table placing them side by side would repeat exactly the error this
document is about. Comparisons must be within one aggregation.

---

## 6. Actions

1. Report the aggregation-sensitivity table (§3) in the paper. It is a
   methodological contribution in its own right: a 1.53× swing from a reporting
   choice that papers rarely state.
2. Add the recomputed persistence-vs-Weng comparison (§4) as a table. It is the
   single strongest result available to us.
3. Narrow the claim in §5 of the paper: our runs speak to our architectures; the
   published architectures are addressed via their own reported numbers.
4. State our aggregation explicitly wherever RMSE appears, and say why it matters.
5. **Still outstanding:** port STGAT/A3TGCN/DCRNN into our harness so the published
   architectures are also tested under our protocol. `notebooks/02_baselines_gnn.ipynb`
   already reproduces them; they were never moved into `src/`. Until that is done,
   the distinction in item 3 must be honoured strictly.

## Provenance

Every figure recomputed from `notebooks/baseline/sri_lanka_2013-2022_shifted.npy`.
Their protocol from `papers/GraphRepresentation_Dengue_IEEEbigData2024.pdf` §VI;
their aggregation from `reference_repo/Models/evaluation.py` lines 23 and 175–200.
