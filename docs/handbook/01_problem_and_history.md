# Chapter 1 — The Problem and the History

*Prerequisites: none.*

---

## 1.1 What dengue is, and why anyone forecasts it

Dengue is a viral infection carried between humans by *Aedes* mosquitoes,
principally *Aedes aegypti*. The transmission cycle has a shape that matters for
everything in this project:

```
infected human  --mosquito bites human-->  infected mosquito
      ^                                            |
      |                                            v
infected human  <--mosquito bites human--   infected mosquito
```

There is no human-to-human transmission. The virus must pass through a mosquito,
and inside the mosquito it must incubate — the *extrinsic incubation period*,
roughly 8–12 days — before that mosquito can infect anyone. Inside the human
there is a further *intrinsic incubation period* of 4–7 days before symptoms
appear, and reporting adds more delay still.

**Consequence:** the causal chain from "conditions became favourable for
mosquitoes" to "cases appear in a health-ministry report" is weeks long. This is
why the dataset in this project lags its rainfall channels by 12 weeks. It is
also why forecasting three weeks ahead is considered useful rather than trivial:
an accurate three-week warning is enough time to spray breeding sites.

Sri Lanka is a natural place to study this. It has 25 administrative districts,
a tropical climate with two monsoons, and a national surveillance system that
publishes weekly case counts per district. In 2017 it experienced the largest
recorded dengue outbreak in its history — over 186,000 cases. That outbreak
appears in our dataset and, as Chapter 3 will show, dominates the evaluation.

---

## 1.2 The forecasting problem, stated precisely

We have, for each of 25 districts and each of 459 consecutive weeks:

- a **case count** — the number of dengue cases reported that week in that
  district;
- ten **environmental covariates** — temperature, humidity, soil moisture,
  precipitation, vegetation index, canopy interception.

Write the case count for district *i* in week *t* as $y_t^{(i)}$.

**The task.** Given the last $W$ weeks of data, predict the next $H$ weeks of
case counts for every district simultaneously:

$$
\big(y_{t-W+1}, \ldots, y_t\big) \;\longmapsto\; \big(\hat{y}_{t+1}, \ldots, \hat{y}_{t+H}\big)
$$

where each $y_t$ is a vector of 25 numbers, one per district.

Throughout this project $W = 3$ and $H = 3$. Those values are not arbitrary. The
benchmark paper chose them after consulting a Sri Lankan epidemiologist, on the
grounds that the average interval between dengue infection and diagnosis is about
three weeks. We inherit them so our numbers stay comparable, and Chapter 9
records what happened when someone tested wider windows.

### Why this is a *multivariate* problem, not 25 separate ones

You could treat each district as its own time series and fit 25 independent
models. The hypothesis motivating this entire line of work is that you should
not — that dengue spreads *between* districts as infected humans travel, so
knowing what happened in Colombo last week should help predict Gampaha this week.

That hypothesis is testable, and Chapter 9 tests it. Hold onto the fact that it
is a hypothesis and not an assumption.

---

## 1.3 Why this problem is hard

Four properties of the data make it difficult, and each one recurs throughout the
handbook.

### The target is extremely heavy-tailed

| statistic | value |
|---|---|
| median weekly cases per district | 13 |
| maximum | 2,631 |
| zero-case district-weeks | 9.7% |
| skewness | 8.6 |
| share of all cases in the top 1% of district-weeks | 18.3% |

A distribution where 1% of observations carry 18% of the mass breaks the
intuitions behind most standard tools. Squared-error loss becomes a measurement
of a handful of outbreak weeks. Percentage error explodes on the 9.7% of weeks
that are zero. Averages of any kind become unstable.

### The signal is dominated by its own past

This is the single most important fact in the project.

| predictor | correlation \|r\| | r² (variance explained) |
|---|---|---|
| cases at *t−1* | 0.921 | **0.848** |
| cases at *t−3* | 0.817 | 0.667 |
| best environmental covariate | 0.145 | **0.021** |

Last week's case count explains ~85% of this week's variance. The best
meteorological variable — specific humidity at a 4-week lag — explains 2%.

Whatever you build must beat a model that simply says "next week will look like
this week". Chapter 5 formalises this as the **persistence baseline**, and it is
the bar that matters.

### The epidemic regime is not stationary

Annual national totals over the eight complete years in the data:

```
60,372   31,234   22,711   48,075   154,000   48,008   96,705   24,560
```

The largest year is **6.8×** the smallest. A model trained on quiet years and
tested on 2017 is being asked to extrapolate, not interpolate — a fundamentally
harder problem, and one that no amount of architecture tuning solves.

### The data has defects

Chapter 3 documents six. Two matter enormously: 4% of the weather values are the
number zero where they should be "missing", and one week is a reporting artifact
with 19× the surrounding case volume.

---

## 1.4 The history of this project, and why it changed direction

Understanding *why* the project is shaped the way it is requires knowing what was
tried and what happened.

### Phase 0 — literature review and proposal

Three research threads were identified as promising, and a gap between them:

1. **Spatio-temporal GNNs** for disease forecasting.
2. **Physics-informed neural networks** — adding an epidemiological differential
   equation to the loss so the model is penalised for producing dynamics that are
   epidemiologically impossible.
3. **GAN-based augmentation** — synthesising extra outbreak-like sequences,
   because real outbreaks are rare.

The gap: no published work combined all three for dengue. Three "contributions"
were proposed: (a) a physics-informed SEIR–SEI loss, (b) GAN augmentation, and
(c) a learned adaptive graph.

### Phase 1 — the baseline

A GCN and a GAT were trained under a rolling-origin protocol. The result:

| model | RMSE | MAE |
|---|---|---|
| Persistence (naive) | 44.8 | 15.7 |
| GCN | 45.3 | 15.9 |
| GAT | 45.5 | 15.9 |

**The baseline matched the persistence floor rather than beating it.** This was
recorded honestly rather than buried, which turned out to be the most consequential
decision in the project — because it framed everything that followed as "why does
this stop here?" rather than "how do we get a better number?"

Two refinements were what made it competitive at all, and both are now standard
throughout the codebase:

- **Residual over persistence.** The network predicts a *correction* to last
  week's value rather than the absolute count. This moved RMSE from ~66 to ~45.
- **log1p target space.** Train on $\log(1 + y)$ so squared error is not hijacked
  by the tail.

### Phases 2 and 3 — the contributions, and a series of negative results

The adaptive graph, spatial regularisation, mechanistic constraints and GAN
augmentation were each implemented and tested. The experiment log records what
happened, and it is unusually candid:

- **EXP-009** — the adaptive-graph effect that looked promising at 24 runs **did
  not survive** replication at 64 runs.
- **EXP-012** — an interim claim that wider windows helped was **retracted**; it
  did not replicate across folds.
- **EXP-014** — the physics-informed constraint's apparent benefit came from a
  constant being **wrong by 3.4×**; with the constant corrected, the term became
  inert.

This is a research record of a team repeatedly disproving its own promising
results. That is what good practice looks like, and it is why the later
reproduction work treats those findings as evidence rather than noise.

### The reproduction phase — the current work

A question had been left unanswered: *the benchmark paper reports GNNs beating
classical baselines. We find they lose to persistence. Which is right?*

Answering it required running the benchmark's own code. That produced the results
in Chapter 9, and three discoveries that reframed the project:

1. The published numbers **do reproduce** — the paper is honest about what it ran.
2. The reported column is **training-inclusive**, so it does not mean what its
   name suggests.
3. Naive persistence **beats every model** in that column — a baseline the paper
   never tested.

There is no contradiction between the two sets of results. There never was. The
apparent conflict was two papers measuring different quantities and comparing the
numbers.

---

## 1.5 The three workspaces, and why they are separate

The repository has three parallel workspaces. Understanding why they do not share
code is important.

| workspace | question it answers | independence rule |
|---|---|---|
| `reproduction/` | Do the authors' numbers reproduce? | Runs **their** code at a pinned commit. We contribute nothing but the comparison. |
| `crosscheck/` | Is **our** code right? Are the papers sound? | Shares **no code** with `src/`. Every formula re-derived independently. |
| `analysis/` | Why does accuracy cap where it does? | Builds on both. |

The `crosscheck/` independence rule is the important one. If `crosscheck`'s
metric implementation imported `src`'s, then a bug in the SMAPE denominator would
agree with itself perfectly and prove nothing. The two implementations were
written separately from the same published definitions, and a single test asserts
they produce identical numbers. **Agreement is then evidence; disagreement is a
bug in one of them.**

---

## 1.6 What "good" would look like

Given everything above, what would count as success?

**Not** a lower RMSE on the pooled average. Every model measured on this dataset —
the benchmark's five architectures, this project's own GCN variants, and the
reproduction-derived baselines — lands within about ±2 RMSE of the persistence
floor. Moving that number by 1% is noise.

Success would be one of:

- **Beating persistence on the extrapolation fold specifically** — the fold whose
  test window contains an outbreak larger than anything in its training data.
  That is the only place these models genuinely fail, and the only place a real
  improvement would show.
- **Calibrated uncertainty.** A forecast that says "150 cases, 90% interval
  [40, 400]" is operationally more useful than a point estimate of 150, even at
  the same RMSE. Nothing in the benchmark provides this.
- **A defensible negative result.** "Spatio-temporal GNNs do not beat persistence
  on district-level dengue data, and here is the measured reason" is a genuine
  contribution, and it is the one this project is currently best positioned to
  make.

---

*Next: [Chapter 2 — Epidemiological Modelling](02_epidemiology_theory.md)*
