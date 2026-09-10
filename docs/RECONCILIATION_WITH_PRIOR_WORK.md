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
SEIR model for influenza, not a dengue paper. See `docs/STAGE3_EXPERIMENTS.md` §1.1 (tag `phase23-archive`).

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

> **Note added 2026-09-10.** These three figures vary the *aggregation* while
> holding the evaluation set fixed. §7.2 later established that Table I varies the
> **evaluation set** too, and that the effect is larger. Recomputed directly from
> `evaluation.py`'s own index arithmetic, persistence under their aggregation is
> **34.67** on the `full` loader (what the Cross Validated column reports) and
> **39.08** on the held-out `test` slices (what the Full Dataset column reports).
> The 38.46 above is close to the latter. Quote whichever matches the column being
> compared, and say which.

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

> **Corrected 2026-09-10.** This section originally cited persistence as
> **38.46**, computed on the *held-out test* slices. Table I's "Cross Validated"
> column is not computed there — see §7 — so that was a basis mismatch. The
> matching figure is **34.67**. The conclusion is unchanged and slightly
> strengthened: the correct comparator is *lower*, so every model loses by more.

Persistence under their own aggregation (mean of per-window RMSEs), recomputed on
each of the two bases their table actually uses:

| Basis | MAE | RMSE (per-window) | RMSE (pooled) |
|---|---|---|---|
| `full` loader — what the **Cross Validated** column reports | 18.58 | **34.67** | 55.59 |
| held-out `test` slices — what the **Full Dataset** column reports | 21.66 | **39.08** | 61.08 |

Against their Table I, Cross Validated column, on its matching basis:

| Model (Weng et al., shifted, CV) | Reported RMSE | vs persistence 34.67 |
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
the two papers. Our 55.12 (pooled) and their 44.78 (per-window mean, on the `full`
loader) are different quantities in two ways at once — aggregation *and*
evaluation set. Any table placing them side by side would repeat exactly the error
this document is about. Comparisons must be within one aggregation **and** one
evaluation set.

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

---

## 7. Addendum, 2026-09-10: their code has now been run

§6 item 5 recorded "port STGAT/A3TGCN/DCRNN into our harness" as outstanding. It
was overtaken by something better: **their own code was run, unmodified, at a
pinned commit**, on Kaggle. See `reproduction/`.

### 7.1 Table I reproduces

| Model | our CV MAE / RMSE | vs their `results.txt` | our Full-Dataset MAE / RMSE | vs Table I |
|---|---|---|---|---|
| AAGCN | 41.93 / 55.83 | **−0.00% / −0.00%** | 40.97 / 50.25 | +4.3% / +2.1% |
| DCRNN | 45.81 / 71.32 | −0.35% / −0.40% | 25.46 / 35.75 | −2.8% / −1.5% |
| STGAT | 25.80 / 45.43 | +1.64% / +1.45% | 22.99 / 40.82 | +3.2% / −4.3% |
| ASTGCN | 34.33 / 48.82 | +1.92% / +2.30% | 22.13 / 33.20 | +7.3% / +2.5% |
| A3TGCN | 35.18 / 60.11 | +3.39% / +2.71% | 19.60 / 31.62 | +5.1% / +3.5% |

Worst deviation over all 20 comparisons: 7.3%. Two models match to four decimals.
Their published numbers are real, and this document's premise — that the
disagreement was aggregation, not error — is confirmed from their side too.

### 7.2 The two columns are the reverse of what their names suggest

This is new, and it changes how §4 must be read. Testing four candidate
definitions against three models, only one fits:

- **"Cross Validated"** = the `full` loader, i.e. **training-inclusive**, averaged
  over the five segments. In `evaluation.py` every `run_*` computes the held-out
  score and then discards it:

  ```python
  y_pred, y_truth, _, _  = infer(model, "cpu", test, m, s, "Test")   # discarded
  y_pred, y_truth, ma, rm = infer(model, "cpu", full, m, s, "Full")  # appended
  ```

  Measured per segment, **70% of every reported evaluation set is training data**.

- **"Full Dataset"** = the `test` loader, a genuinely held-out score at
  subset=1.0. The name means "no CV subsetting was used", not "evaluated on
  everything".

Every alternative reading fails badly — A3TGCN's `full` value is 32.74 against the
paper's printed 18.65, while its `test` value is 19.60.

### 7.3 Three further defects in the released implementation

Each measured, in `crosscheck/FINDINGS.md` F1.1–F1.6:

| Finding | Effect |
|---|---|
| RMSE accumulated per batch with `batch_size=1`, then divided | reports RMSE **39% below** pooled (33.48 vs 54.84) |
| z-score statistics computed over the whole array before splitting | test-period location and scale known during training |
| ARIMA/RF/LSTM run on **Kalutara alone**, predicting same-week cases from same-week covariates | a different task from the GNNs' 3-step-ahead, 25-district forecast |

The third is the most consequential for the paper's headline claim, since that
claim is a comparison between those two groups.

### 7.4 What is and is not reproducible

Of Table I's 20 rows, **10** can be reproduced: the five GNNs on the shifted
dataset, and — newly — the unshifted half, since the unshifted array turns out to
be reconstructible from `sri_lanka_2013-2022_vertical.csv` in their repo
(EDA F2). The ARIMA/RF/LSTM rows cannot: they read `MLSO2_Final.csv`, which is not
in the repository. XGBoost and ARNN have no script at all.

`reproduction/REPRODUCIBILITY_MATRIX.md` records this row by row.

### 7.5 DengueGNN: §2's judgement is confirmed by the paper itself

§2 set DengueGNN aside as not comparable. The published article confirms it goes
further than that — it is not reproducible at all. Its data-availability statement
points at OpenDengue without naming a country, admin level, date range or node
count, and **there is no code-availability statement**. No hyperparameter value is
reported despite a grid search being described.

---

## 8. Revised actions

1. ✅ Aggregation-sensitivity table — still a methodological contribution worth reporting.
2. ✅ Persistence-vs-Weng comparison — now on the correct basis (34.67, §4).
3. ✅ Narrow the claim — and it can now be widened slightly: their architectures
   *have* been run, as their own code, so "published spatio-temporal GNNs lose to
   persistence on this dataset" is supportable from a reproduction as well as from
   their printed numbers.
4. ✅ State the aggregation wherever RMSE appears — **and the evaluation set**, which
   §7.2 shows is the larger of the two effects.
5. ✅ Superseded by `reproduction/`: their code runs directly, which is stronger
   evidence than porting their architectures into our harness would have been.
6. **New:** the paper must not cite `results/table_main.md` or its siblings. Those
   are the n=24 tables, superseded by `final_table_*.md` at n=64, and they carry an
   adaptive-graph claim that did not survive replication (EXP-009, EXP-018). They
   now carry a supersession banner.
