# Chapter 5 — Evaluation

*Prerequisites: Chapter 4.*

This is the most important chapter in the handbook.

Nearly every error found in this project — ours and the published work's — has
been in **measurement**, not in modelling. The models mostly do what they claim.
The numbers describing them frequently do not mean what they appear to.

By the end you will be able to look at a forecasting result table and identify the
five specific ways it can mislead you, because you will have seen all five in a
published paper.

---

## 5.1 Metrics

### RMSE — root mean squared error

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i}(\hat{y}_i - y_i)^2}$$

```python
def rmse(pred, truth):
    p, y = _pair(pred, truth)
    return float(np.sqrt(np.mean((p - y) ** 2)))
```

In the units of the target (cases). Squaring means large errors dominate: a single
error of 100 outweighs a hundred errors of 10.

**On this dataset RMSE is largely a measurement of outbreak weeks.** That is not
necessarily wrong — outbreak weeks are what you care about operationally — but you
must know that is what you are measuring.

### MAE — mean absolute error

$$\text{MAE} = \frac{1}{N}\sum_{i}|\hat{y}_i - y_i|$$

Also in cases, but each error counts once. MAE describes typical performance; RMSE
describes worst-case performance. **Report both.** A large RMSE/MAE ratio tells you
errors are concentrated — and on this dataset that ratio runs about 2.5.

### MAPE — and why it is a trap here

$$\text{MAPE} = \frac{100}{N}\sum_i \frac{|\hat{y}_i - y_i|}{y_i}$$

Scale-free, which is why people like it. **But 9.7% of this dataset's targets are
zero, so the denominator is zero.**

Here is what the reference implementation does about that:

```python
WENG_MAPE_EPS = 1e-15

def mape_weng(pred, truth):
    """MAPE exactly as the Weng et al. reference implementation computes it.

    Percentage. Explodes to ~1e17 on any zero-truth element -- that behaviour is
    the point of transcribing it, so keep it.
    """
    p, y = _pair(pred, truth)
    return float(np.mean(np.abs(p - y) / (y + WENG_MAPE_EPS)) * 100)
```

Adding `1e-15` looks like a routine numerical stabiliser. It is not. On a zero
week with a prediction of 1, the term becomes $1/10^{-15} = 10^{15}$, and after
the ×100 it is $10^{17}$. **Any batch containing one zero week produces a MAPE
dominated entirely by that week.**

The function is transcribed faithfully, with a docstring insisting the explosion
be preserved, because the goal is to reproduce what produced the published
numbers — not to produce a defensible metric. Those are different jobs, and
conflating them is how reproductions quietly fail.

Two honest alternatives:

```python
def mape_masked(pred, truth, floor=1.0):
    """MAPE over elements with truth >= floor. nan if none qualify."""
    m = y >= floor
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(p[m] - y[m]) / y[m]) * 100)


def smape(pred, truth, eps=1e-6):
    """Symmetric MAPE, percentage, bounded at 200."""
    return float(np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + eps)) * 100)
```

**Masked MAPE** simply excludes zero weeks and says so — you are reporting on a
subset, which is fine as long as it is stated.

**SMAPE** puts both values in the denominator:

$$\text{SMAPE} = \frac{100}{N}\sum_i \frac{2|\hat{y}_i - y_i|}{|\hat{y}_i| + |y_i|}$$

It is bounded at 200%, so no single point can dominate. It has its own asymmetry
(over-prediction and under-prediction are not penalised equally), but it is
finite, which MAPE is not.

**This project reports SMAPE plus masked MAPE, and never bare MAPE.**

### Probabilistic metrics

A point forecast of 150 cases tells you nothing about confidence. Three metrics
score distributions instead.

**CRPS** — Continuous Ranked Probability Score — generalises MAE to
distributions. For a Gaussian forecast there is a closed form:

```python
z = (y - mu) / sigma
crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
```

CRPS rewards being both **accurate and appropriately confident**. Declaring a huge
variance to hedge is penalised; so is a confident wrong answer. It reduces to MAE
when the forecast is a point mass, which makes it directly comparable.

**PICP** — fraction of true values falling inside the stated interval. For a 95%
interval, PICP should be ≈ 0.95. Below that, you are overconfident.

**MPIW** — mean interval width. PICP alone is trivially gamed: predict
$[-\infty, \infty]$ and score 1.0. **PICP and MPIW must always be reported
together** — coverage is only meaningful alongside the width that achieved it.

### Moran's I — is spatial structure being reproduced?

$$I = \frac{n}{S_0}\cdot\frac{\sum_i\sum_j w_{ij}(x_i-\bar{x})(x_j-\bar{x})}{\sum_i (x_i-\bar{x})^2}$$

Roughly a correlation coefficient for space: I ≈ +1 means neighbours resemble each
other, 0 means no spatial pattern, negative means neighbours differ.

Compute it on the observed and predicted maps. If the truth has I = 0.4 and your
forecast has I = 0.05, your model produces spatially flat forecasts — it may score
well on RMSE while getting the *geography* wrong, which is precisely what a
graph-based method is supposed to get right.

---

## 5.2 Baselines

**A metric value alone is meaningless.** RMSE 45 is neither good nor bad until
something else is scored on identical data.

### Persistence — the one that matters

$$\hat{y}_{t+h} = y_{t-1} \quad \text{for all } h$$

"Next week will be like last week." No parameters, no training.

```python
def persistence_scores(fold, cases, horizon=3, artifact=None):
    """Last-value-carried-forward on the same test windows -- the floor to beat."""
    pred  = np.stack([np.repeat(cases[i - 1][:, None], horizon, axis=1)
                      for i in fold.test_index])
    truth = np.stack([cases[i : i + horizon].T for i in fold.test_index])
    return pooled_scores(pred, truth, artifact)
```

Note `fold.test_index` — persistence is scored on the **identical windows** as the
model. Any other comparison is meaningless.

**Why persistence is strong here:** lag-1 autocorrelation ≈ 0.92 pooled (Chapter
3). Persistence captures ~85% of the variance for free.

**The result that reframed this project:** under the benchmark paper's own
protocol, on its own slices, with its own metric convention —

| | MAE | RMSE (per-window) |
|---|---|---|
| Weng et al., Table I — STGAT (best of 10) | 25.38 ± 1.37 | 44.78 ± 2.26 |
| Our reproduction of STGAT | 24.01 ± 1.32 | 42.31 ± 2.47 |
| **Persistence, identical slices** | **18.58 ± 0.48** | **34.67 ± 0.89** |

Persistence beats all ten published models on both metrics in that column. **The
paper never tested it.**

Be precise about scope, though — the paper has a second column, and there the
picture is mixed. Persistence scores 17.87 MAE / 33.48 RMSE, beating all ten on
MAE, but **A3TGCN (30.55) and ASTGCN (32.41) beat it on RMSE**. So:

- *Cross Validated column* — persistence beats every model on both metrics.
- *Full Dataset column* — persistence beats every model on MAE; two GNNs beat it
  on RMSE.

"Persistence beats everything in Table I" would be an overstatement. Getting this
right matters more than making the point forcefully.

### Other baselines worth having

- **Seasonal naive** — $\hat{y}_t = y_{t-52}$. Useless here; Chapter 3 D6 showed
  there is no usable annual cycle.
- **Historical mean** — the district's average. Weak, but catches a model that has
  learned nothing.
- **Linear/ARIMA** — classical statistical forecasting.

**Rule: if your neural network does not beat persistence, you do not have a
result.** You may have a useful negative finding, but it has to be reported as
one.

---

## 5.3 Cross-validation for time series

Standard k-fold CV shuffles data into k groups. **On a time series it is
invalid** — it trains on the future to predict the past.

### Rolling-origin (expanding-window) CV

```
Fold 1:  [=== train ===][val][test]
Fold 2:  [====== train ======][val][test]
Fold 3:  [========= train =========][val][test]
                                          time ->
```

Training always precedes validation, which always precedes test. Each fold moves
the origin later and keeps everything before it.

```python
for origin in (0.55, 0.70, 0.85):
    cut = int(origin * len(ids))
    end = int(min(origin + test_frac, 1.0) * len(ids))
    train_ids = ids[: cut - val_weeks]
    val_ids   = ids[cut - val_weeks : cut]
    test_ids  = ids[cut:end]
```

Three details:

- **`cut - val_weeks`** — validation is carved out of the *end* of training, so it
  is the most recent data before test. Validating on old data would select
  hyperparameters for a regime that has passed.
- **`min(origin + test_frac, 1.0)`** — the last fold's test window is clipped at
  the end of the series.
- **Statistics are recomputed inside this loop** (Chapter 4, Leak 2). Each fold has
  a different training period and therefore a different mean and standard
  deviation.

### The frozen protocol

```
3 origins × 3 seeds = 9 runs per configuration
window W = 3, horizon H = 3
metrics per horizon and overall: RMSE, MAE, SMAPE, masked MAPE
```

"Frozen" means: **every row of the results table uses this, identically.** Change
any part of it and you must re-run every row, including the baseline.

This sounds bureaucratic. It is the difference between a table you can defend and
a table where a difference between two rows might be a protocol difference rather
than a model difference. That failure is undetectable after the fact.

---

## 5.4 The six ways a results table misleads you

All five were found in one published paper by running its code. None is visible
from reading it.

### Trap 1 — evaluating on training data

The paper's Table I has a column labelled **"Cross Validated"**. The natural
reading is "held-out performance".

What the code does:

| origin | total windows | in the reported evaluation | of which trained on |
|---|---|---|---|
| 1.0 | 453 | 453 | **317 (70.0%)** |

The evaluation loader is the `full` loader — every window, including every
training window. **70% of the reported "cross validated" evaluation is data the
model was fitted on.**

The paper's §VI describes a 70/30 train/test split. The code does not implement
what the prose describes.

**How to catch this class of error:** the only reliable way is to read the
evaluation loop and ask *which indices does this iterate over?* No amount of
reading the prose reveals it.

**The genuinely counter-intuitive part:** the column labelled "Full Dataset" is
the one computed on **held-out** data. The two column names are, in effect,
inverted relative to their meanings.

### Trap 2 — per-window averaging instead of pooling

Two ways to compute RMSE over many windows:

```python
# Pooled: one RMSE over every prediction
rmse = np.sqrt(np.mean((all_preds - all_truths) ** 2))

# Per-window mean: RMSE of each window, then averaged
rmse = np.mean([np.sqrt(np.mean((p - t) ** 2)) for p, t in windows])
```

These are **not** the same number. By **Jensen's inequality** — the square root is
concave, so the mean of roots is at most the root of the mean — the per-window
average is never larger, and the gap widens as error becomes more uneven across
windows.

Measured on this dataset, using persistence so no training is involved:

| Metric | Pooled | Per-window mean | Reference lower by |
|---|---|---|---|
| RMSE | 54.84 | 33.48 | **39.0%** |
| MAE | 17.87 | 17.87 | **0.0%** |

**A 39% reduction from a choice most readers would never think to check.**

MAE is untouched because it is already a mean of means, and averaging means of
equal-sized groups gives the same answer either way. This is why the two metrics
in the published table are not distorted equally — a fact that looks like noise
until you know the cause.

Neither convention is *wrong*. But a paper reporting per-window RMSE and a paper
reporting pooled RMSE are reporting different quantities, and comparing their
numbers is meaningless.

### Trap 3 — normalisation spanning the test period

```python
mean, std = data.mean(), data.std()     # over the WHOLE array, before any split
```

Measured impact here:

```
whole-series statistics:  mean 43.98   sd 107.94
training weeks only:      mean 47.03   sd 115.01
                          6.5% shift in location, 6.1% in scale
```

Predictions are inverse-transformed using these, so the reported error is
expressed in a scale the test period helped define. Small, but it is leakage and
it flows in the flattering direction.

### Trap 4 — comparing models that solve different problems

The paper's central claim is that GNNs beat classical baselines: average RMSE
38.22 vs 70.65.

What the baseline scripts actually do:

```python
data = pd.read_csv("../Data/Datasets/MLSO2_Final.csv")
state_data = data[data["region"] == "Kalutara"].reset_index(drop=True)
```

| | Classical baselines | GNNs |
|---|---|---|
| Districts | **1 (Kalutara)** | 25 |
| Task | regression on **same-week** covariates | **3-step-ahead forecast** |
| Input | current covariates | 3-week window of past cases |

The baselines are not forecasting at all — they fit this week's cases from this
week's weather. And they are scored on one high-incidence district rather than a
25-district average.

Holding the estimator, data and segments fixed and changing **only the task
framing**:

| Framing | MAE | RMSE |
|---|---|---|
| Reference framing: Kalutara, same-week covariates | 82.52 ± 29.42 | 129.39 ± 54.31 |
| GNN framing: 25 districts, 3-step-ahead | 45.29 ± 12.18 | 125.35 ± 35.78 |

**Framing alone moves MAE by a factor of 1.8**, in the direction that flatters the
paper's conclusion. Kalutara is a high-incidence district, so its errors are large
next to an average that includes many quiet districts.

### Trap 5 — one contaminated window dominating

From Chapter 3 D2: on the origin-0.85 fold, **6 of 68 test windows carry 90% of
the squared error**. Persistence RMSE there is 68.62 with them, 22.80 without.

The response is to report both:

```python
def pooled_scores(pred, truth, artifact=None):
    out = {"RMSE": rmse(pred, truth), "MAE": mae(pred, truth)}
    if artifact is None:
        return out
    keep = ~np.asarray(artifact, dtype=bool)
    out["n_windows"]  = int(keep.size)
    out["n_artifact"] = int(keep.size - keep.sum())
    out["RMSE_clean"] = rmse(pred[keep], truth[keep]) if keep.any() else float("nan")
    out["MAE_clean"]  = mae(pred[keep], truth[keep]) if keep.any() else float("nan")
    return out
```

The docstring states the principle: *"Both numbers are reported because neither
alone is honest."* Including the artifact measures a reporting backlog; excluding
it silently flatters every arm equally and hides that the fold's difficulty is one
week of bad data.

And the baseline must be split the same way — **"the floor has to be split the
same way the models are, or the comparison is between two different test sets."**


---

### Trap 6 — the model that *is* the baseline

The nastiest one, because it produces the **best-looking row in the table**.

When a model predicts a residual over persistence, outputting zero *is*
persistence — and it inherits persistence's score. On a target where the previous
week explains r² = 0.85, that is a safe, competitive local minimum. A model that
settles there will have a low RMSE, possibly the lowest in your table, and will be
forecasting nothing at all.

It was first caught in the architecture study, where STGAT came top:

| arm | RMSE mean | vs persistence |
|---|---|---|
| persistence | 55.12 | — |
| **STGAT** | **54.95** | 15/24, p = 0.307 |
| A3TGCN | 56.19 | 11/24, p = 0.839 |
| GCN + gated TCN | 56.39 | 12/24, p = 1.000 |
| GAT | 59.99 | 10/24, p = 0.541 |
| GCN (control) | 60.08 | 9/24, p = 0.307 |

The lowest mean, and the only arm under persistence. It looks like the headline.

**The tell was not in the mean — it was in the spread of the paired differences:**

| arm | mean paired difference vs persistence | SD |
|---|---|---|
| GCN | +4.96 | 11.97 |
| GCN + gated TCN | +1.27 | 9.50 |
| GAT | +4.87 | 16.97 |
| **STGAT** | **−0.16** | **0.70** |
| A3TGCN | +1.08 | 9.57 |

Every other arm deviates from persistence with an SD of 9.5–17. STGAT deviates by
**0.70** — it tracks persistence to within one RMSE point on *every single fold*.
That is not a model that agrees with persistence on average. It is persistence.

**How to test for it.** Measure what the model adds against what it would need to
add:

```
move   = mean |prediction − persistence|      what the model actually contributes
needed = mean |truth      − persistence|      what it would have to contribute
ratio  = move / needed
```

A ratio near zero means collapse, whatever the RMSE says.

**The sharper version of the same test** (EXP-025) regresses the model's predicted
log-growth on the true log-growth and reads off the slope:

| | A3TGCN, current protocol |
|---|---|
| slope of predicted log-growth on **true** log-growth | **0.002** |
| slope of predicted log-growth on `log R̂` | −0.007 |

A slope of 1.0 would be an undamped response; 0.0 is a flat forecast. At 0.002 the
model's growth predictions carry **no information about growth**. The verified
architectures in this project are sophisticated persistence, and the collapse is
not a bug in any one of them.

**And here is the part that matters for anyone trying to fix it:** under squared
error, collapse is *correct*. The residual is nearly unpredictable, and the
conditional mean of an unpredictable quantity is zero. Flatness is what MSE is
asking for. That is why an asymmetric loss, a physics penalty and a mechanistic
decoder each bought responsiveness and paid for it in RMSE, in the same direction,
in three separate experiments (EXP-021, EXP-023). A model that reacts is MSE-worse
by construction — so if you want reaction, you have to change the objective, not
add a term to it.

---

## 5.5 Reading a results table, in practice

Given any forecasting table, ask:

1. **Is there a naive baseline?** If not, the numbers are uninterpretable.
2. **What data is the reported number computed on?** Read the evaluation loop, not
   the prose.
3. **Pooled or per-window?** Worth 39% here.
4. **Where do the normalisation statistics come from?**
5. **Do the compared models solve the same problem?** Same districts, same
   horizon, same inputs.
6. **How many seeds, and what is the variance?** A difference smaller than the
   seed-to-seed spread is not a difference.
7. **Is any single window dominating?**

If you cannot answer these from the paper, you cannot use its numbers as a target.
That is the honest conclusion this project reached, and it is why the reproduction
workspace exists.

---

## 5.6 What this means for our own results

Our Phase-1 baseline was recorded as matching the persistence floor: RMSE 45.3 vs
44.8. This was read, for a long time, as underperforming against the published
STGAT at 44.78.

**Those two numbers were never comparable.** 44.78 is a per-window-averaged RMSE
(Trap 2), computed largely on training data (Trap 1), under a normalisation that
saw the test period (Trap 3). Under the paper's own protocol, persistence scores
34.67 — better than every model in that column.

**So the Phase-1 result is not a shortfall against the literature. It is the same
result the literature gets, reported honestly.**

That sentence is the payoff of this entire chapter, and it was obtainable only by
running the code.

---

*Next: [Chapter 6 — Graph Neural Networks from Scratch](06_gnn_theory.md)*
