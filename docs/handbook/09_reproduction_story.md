# Chapter 9 — The Reproduction Story

*Prerequisites: Chapters 5 and 7.*

This chapter is the narrative of what was actually done: the question, the
method, the results, and every mistake made along the way — including ours.

Read it as a case study in how to check a published result.

---

## 9.1 The question

A contradiction sat unresolved in the project for months.

> The benchmark paper reports spatio-temporal GNNs beating classical baselines by
> a wide margin (average RMSE 38.22 vs 70.65). Our own baseline, built on the same
> data, **matches rather than beats** naive persistence.
>
> Either our implementation is wrong, or something about theirs is not what it
> appears.

You cannot resolve this by reading. Both papers describe reasonable protocols.
The only way is to **run their code and measure**.

That is what `reproduction/` is for.

---

## 9.2 The method

### Principle: change nothing

The rule was stated up front and enforced throughout:

> Never carry any assumptions — everything must come from the paper itself or
> their repository.

Concretely: their code, their pinned dependencies, their hyperparameters, their
data, their metric conventions, their evaluation loop. No improvements. No
"obvious" fixes.

The moment you fix something, you are no longer reproducing.

### What is reproducible, and what is not

Table I has 20 rows (10 models × 2 datasets). **Five are exactly reproducible.**

| | status | why |
|---|---|---|
| 5 GNNs, shifted data | ✅ | code, data, invocation and expected output all released |
| ARIMA / RF / LSTM | ❌ | depend on `MLSO2_Final.csv`, not in the repo |
| 10 unshifted rows | ⚠️ | the unshifted array is *reconstructible* (EDA F2), so potentially reopened |

The reproducible five are the paper's actual contribution, so the reproducible
subset is the part that matters.

**The reproduction target is not the paper's table.** The repo contains
`Models/results.txt` — the authors' own raw run output, matching Table I to the
decimal. Reproducing against the raw output rather than the rounded table is
strictly stronger.

### The three forced deviations

Some deviation was unavoidable. All three are minimal, and all three are recorded
in every notebook:

1. **Standalone Python 3.11 via `uv`** instead of Kaggle's 3.12, so the authors'
   `torch==2.1.2` installs at all.
2. **`torch_geometric==2.4.0`** instead of the pinned 2.5.3 — the newest version
   where the authors' own PGT pin imports.
3. **`torch_geometric_temporal` with `--no-deps`**, so its stale `pandas<=1.3.5`
   does not bind — the authors had already overridden it by pinning pandas 2.2.

The reasoning for each is explicit:

> Bumping torch instead would change the numerical stack the models run on, which
> is a far larger deviation.

> PGT stays at the authors' `0.54.0` rather than being upgraded, because PGT is
> the package that defines the model architectures: changing it risks changing the
> models, a far larger deviation than a PyG patch release.

**Choose the deviation that touches the least, and write down why.** A
reproduction with three documented deviations is credible. One with three
undocumented ones is not a reproduction.

---

## 9.3 The results

All five architectures reproduce. Deviation from published values, worst case
7.3%:

| Model | CV column (vs `results.txt`) MAE / RMSE | Full Dataset (vs Table I) MAE / RMSE |
|---|---|---|
| **AAGCN** | **−0.00% / −0.00%** | +4.3% / +2.1% |
| DCRNN | −0.35% / −0.40% | −2.8% / −1.5% |
| STGAT | +1.64% / +1.45% | +3.2% / −4.3% |
| ASTGCN | +1.92% / +2.30% | +7.3% / +2.5% |
| A3TGCN | +3.39% / +2.71% | +5.1% / +3.5% |

AAGCN matches to four decimal places. The residual variation across the others is
what you expect from GPU non-determinism and a patch-level PyG difference.

**Conclusion 1: the published numbers are real.** The paper reports what its code
produces. This needed establishing before anything else could be said, and it
disposes of the "maybe they made it up" hypothesis entirely.

Everything that follows is therefore about **what those numbers measure**, not
whether they are genuine.

---

## 9.4 Discovery 1 — the columns mean the opposite of their names

Table I has two columns: **"Cross Validated"** and **"Full Dataset"**.

Reading the evaluation code:

| column label | data loader used | what it actually is |
|---|---|---|
| "Cross Validated" | `full` | **every** window, training included |
| "Full Dataset" | `test` | genuinely **held-out** windows |

At origin 1.0: 453 windows evaluated, **317 of them (70.0%) trained on**.

The paper's §VI describes the protocol as a 70/30 train/test split. The code does
not implement what the prose describes.

**So the column the paper presents as its robustness evidence is the
training-inclusive one**, and the column that sounds like it includes everything
is the honest held-out measurement.

This is not detectable from the paper. It is only detectable by asking of the
evaluation loop: *which indices does this iterate over?*

---

## 9.5 Discovery 2 — persistence beats every model

Once you know the CV column is training-inclusive, the obvious question is: what
does a trivial baseline score on the same slices?

Scored on identical windows, with the identical metric convention, under the
identical protocol:

| | MAE | RMSE (per-window, as reported) |
|---|---|---|
| Table I — STGAT (best of 10) | 25.38 ± 1.37 | 44.78 ± 2.26 |
| Our reproduction of STGAT | 24.01 ± 1.32 | 42.31 ± 2.47 |
| **Persistence** | **18.58 ± 0.48** | **34.67 ± 0.89** |

**Last-value-carried-forward beats all ten published models on both metrics.**

Note the standard deviations too: persistence is not only better on average but
**dramatically more stable** (±0.48 vs ±1.37 on MAE). It has no parameters to
vary.

### Be precise about scope

The paper's *Full Dataset* column tells a more mixed story. Persistence there
scores 17.87 MAE / 33.48 RMSE:

- beats all ten models on **MAE**
- but **A3TGCN (30.55) and ASTGCN (32.41) beat it on RMSE**

So the accurate statement is:

- *Cross Validated column* — persistence beats every model on both metrics.
- *Full Dataset column* — persistence beats every model on MAE; two GNNs beat it
  on RMSE.

"Persistence beats everything in Table I" would be an overstatement, and the
findings document says so explicitly. **Getting this right matters more than
making the point forcefully** — an overstated finding is refutable, and its
refutation discredits the correct part along with it.

---

## 9.6 Discovery 3 — the 39% metric convention

From `evaluation.py`:

```python
rmse += RMSE(truth, pred)     # inside the loader loop, batch_size = 1
...
rmse /= n
```

That is $\text{mean}_w \text{RMSE}_w$, not pooled RMSE. By Jensen's inequality
(the square root is concave) the per-window mean never exceeds the pooled value.

Measured, using persistence so no training is involved:

| Metric | Pooled | Per-window mean | Reference lower by |
|---|---|---|---|
| RMSE | 54.84 | 33.48 | **39.0%** |
| MAE | 17.87 | 17.87 | 0.0% |

**A 39% reduction from a choice most readers would never think to check.** MAE is
untouched, being a mean of means — which is exactly why the two metrics in the
published table are not distorted equally.

---

## 9.7 Discovery 4 — the baselines solve a different problem

The paper's headline claim compares GNNs (RMSE 38.22) against classical methods
(70.65). But:

```python
data = pd.read_csv("../Data/Datasets/MLSO2_Final.csv")
state_data = data[data["region"] == "Kalutara"].reset_index(drop=True)
```

| | Classical baselines | GNNs |
|---|---|---|
| Districts | **1 (Kalutara)** | 25 |
| Task | regression on **same-week** covariates | **3-step-ahead forecast** |

The baselines are not forecasting. They fit this week's cases from this week's
weather — a fundamentally easier task measured on a harder subset.

Holding the estimator (`RandomForestRegressor`, 100 trees, seed 42), the data and
the segments fixed, and changing **only the framing**:

| Framing | MAE | RMSE |
|---|---|---|
| Reference: Kalutara, same-week covariates | 82.52 ± 29.42 | 129.39 ± 54.31 |
| GNN framing: 25 districts, 3-step-ahead | 45.29 ± 12.18 | 125.35 ± 35.78 |

**Framing alone moves MAE by 1.8×**, in the direction that flatters the paper's
conclusion — because Kalutara is a high-incidence district whose errors are large
next to a 25-district average that includes many quiet ones.

Absolute values differ from Table I because `MLSO2_Final.csv` is not in the repo.
**The ratio between framings is the finding, not the levels** — and the finding
survives the missing file, which is why it is stated as a ratio.

---

## 9.8 Two further findings

**Stated hyperparameters differ from released ones.** Paper §II.D says "a learning
rate of 0.00001 and a weight decay of 0.000001". `evaluation.py` line 26 says
`lr, decay, dropout = 1e-4, 5e-5, 0.1` — a **10× learning rate and a 50× weight
decay**. The reproduction follows the code, since the code produced Table I.

**The feature count.** The paper refers to "20 features at each time-step"; the
released array has 11. This does not affect the GNN reproduction — the models run
with `use_disease_only=True` and see only the case channel — but **the published
feature ablation cannot be reproduced from the released data.**

---

## 9.9 What this means for our own results

The reconciliation, stated plainly:

> Our Phase-1 baseline is recorded as matching the persistence floor (RMSE 45.3 vs
> 44.8), which has been read as underperforming against Weng's STGAT at 44.78.
> **Those two numbers are not comparable**, and more importantly, the comparison
> was never the right one to worry about:
>
> - 44.78 is a per-window-averaged RMSE, computed largely on training data, under
>   a global normalisation that saw the test period.
> - Under the paper's own protocol, persistence scores MAE 18.58 / RMSE 34.67 —
>   better than every model in Table I's cross-validated column.
>
> **So our Phase-1 result is not a shortfall against the literature; it is the
> same result the literature gets, reported honestly.**

There was never a contradiction. There were two papers measuring different
quantities, and someone comparing the numbers.

---

## 9.10 The adaptive graph experiment (EXP-018)

With a trustworthy baseline established, Contribution (c) could finally be tested
properly.

**Design:** four graph modes — `none`, `fixed`, `adaptive`, `hybrid` — sharing one
encoder, one training loop, one normalisation and one protocol.

```python
"""The point of this module is a controlled comparison. Four graph modes share one
encoder, one training loop, one normalisation and one protocol, so a difference
between them is attributable to the graph and nothing else.
"""
```

**Results.** Paired against `fixed` on identical folds and seeds, in RMSE where
positive means worse:

| arm | Δ RMSE | p |
|---|---|---|
| `adaptive` | +0.46 | 0.09 |
| `hybrid` | +0.19 | 0.37 |
| `none` | −0.54 | 0.49 |

**Null result.** The learned graph does not help. Removing the graph entirely does
not hurt.

The decisive evidence is not the p-values, though. It is that **the sign of the
adaptive effect flips between two implementations** on identical folds and seeds:
−0.47 in the repo's model, +0.45 in ours.

**An effect whose sign depends on implementation details is a null effect.** That
is a cleaner argument than any p-value, because it does not depend on a
significance threshold.

It is also consistent with EDA D5 (the graph adds +0.07 above a national common
mode) and with EXP-009, where an adaptive-graph effect that looked promising at 24
runs did not survive replication at 64.

**Three independent lines of evidence pointing the same way is a finding.**

### The head-to-head that disproved a premise

A proposal was made to delete the repo's Phase-2 work on the grounds that its
baseline was "very poor" — RMSE 64.67 against reproduction-quality numbers around
45.

The instruction was to **verify before deleting**, by running the experiments
rather than trusting the notes. Measured head-to-head on identical folds:

```
ours:fixed         43.36
ours:adaptive      43.81
persistence        44.80
theirs:adaptive    45.00
their DenseGCN     45.47
```

**Their baseline is at the floor, not far below it.** The 64.67 figure came from a
*different protocol* — 8 origins rather than 3 — not a worse model.

The deletion premise was wrong, and the deletion did not happen. This is Chapter
5's Trap 5 (protocol differences masquerading as model differences) encountered in
our own records rather than someone else's paper.

**"Verify before deleting" is the instruction that saved this.** Notes decay;
measurements do not.

---

## 9.11 Our own mistakes

The reproduction found errors in published work. It also found several of ours.
Both belong here.

### The retracted seasonality finding

Claimed: bimodal seasonality, 2.9× peak-to-trough, week-of-year feature as
priority #1.

Testing: de-meaned log yearly shapes correlate at **r = −0.065** (chance), **0 of
25** districts show a significant week-of-year effect, R² = **0.03** against
**0.86** from last week's value.

**Cause:** the seasonal profile averaged raw counts across years whose totals span
6.8×. That measures the timing of 2017, not a recurring cycle.

**EXP-012 in the old repo had reached the correct conclusion first and was not
consulted.** Retracted in the notebook, the README, the experiment log, and
`improved.py`.

### The benchmark filter bug

A head-to-head script filtered rows with `!= "persistence"`, but the other
harness writes `"Persistence"`. Persistence rows were being averaged **into**
their model's score.

Fixed with `.lower()` plus an assertion that the filter matched something:

```python
model   = [r for r in pooled if str(r.get("model", "")).lower() != "persistence"]
persist = [r for r in pooled if str(r.get("model", "")).lower() == "persistence"]
assert model and persist, f"unexpected model labels: {set(r.get('model') for r in pooled)}"
```

The comment above it names the stakes: *"A case-sensitive mismatch here silently
averages the persistence rows into the model's score, which is exactly the kind of
error this script exists to avoid making about someone else's work."*

**The fix is the assertion, not the `.lower()`.** A filter that silently matches
nothing is the failure mode; making it loud is the repair. The corrected
`theirs:adaptive` figure — 45.00 in `analysis/results/baseline_benchmark.json` —
is the one quoted in §9.10.

### The inert increment

`mask_artifact` was designed to drop week-395 windows from *training*. Measurement
showed week 395 falls in the **test** set of origin 0.85 and in **no** training
set under this protocol. The increment did nothing.

Removed, with the docstring left behind explaining why — because the idea is
obvious enough that someone will have it again.

### The R2 ablation that was a no-op

`use_mobility=False` did nothing, because no mobility tensor was ever passed. "No
mobility" was byte-identical to "static graph", and the reported *"ranking
reproduced: True"* was an artifact of tied values.

**A passing check on identical inputs is not a passing check.** Rewritten with
real environmental and mobility features.

### The STGAT batching bug

Covered in Chapter 7 §7.6. A 25-node edge index passed with batched input left
every graph after the first with no edges. Silent. The fix was verified
bit-identical against the looped path, and pinned with a test.

### The recurring heredoc bug

Generated notebook cell sources are Python strings containing Python code. `\n`
inside them collapsed to a real newline, producing unterminated string literals.
This happened roughly four times.

The fix that finally worked was not care. It was extending
`tools/check_notebooks.py` to `compile()` every code cell, making a non-parsing
generated notebook an error.

**When a mistake recurs, the problem is the process, not the person.**

---

## 9.12 The physics arc (EXP-021 – EXP-025)

The reproduction settled what the literature's numbers meant. The next question
was whether the project's own promised contribution — a physics-informed loss from
the SEIR–SEI model — could beat them. It cannot, and the reason is worth more than
the attempt.

### Where the error actually lives (EXP-021)

Before designing a constraint, measure the failure it is supposed to fix.

| | A3TGCN | STGAT |
|---|---|---|
| predicted \|weekly log growth\|, max | 0.336 | 0.242 |
| **observed** max | **3.638** | **3.638** |
| outbreak windows | 12.6% of data | — |
| …carrying | **61.7% of squared error** | 60.7% |
| bias on outbreak windows | **−12.92** | −7.32 |

Outbreak weeks are an eighth of the data and nearly two-thirds of the error, and
the models under-forecast them every time. Forecasting them as well as quiet weeks
would take RMSE from ~29 to ~20 — about 30% below the persistence floor. That is
the headroom, and it is much larger than the ablation table suggests.

### Why the obvious fix fails

An asymmetric loss charging under-prediction 3× more *does* remove the bias
(−10.58 → +1.21 for A3TGCN) and *loses* accuracy (28.98 → 31.82). The model cannot
tell which windows are outbreaks, so it inflates everywhere. **The problem is
outbreak detection, not loss asymmetry.**

### The renewal equation, and why it was the right physics to try (EXP-023)

EXP-014 had already retired a growth *ceiling*: an upper bound is slack against a
model that under-reacts, so it contributed exactly zero gradient. The renewal
equation is the mechanistic identity on the observable itself,

```
cases_t = R_t · Σ_s w_s · cases_{t−s}
```

with `w` the generation-interval distribution implied by Phaijoo & Gurung's stage
durations — mean **3.30 weeks**, and validated in `dengue_gnn.seir`. It describes
this data well *in hindsight*: a 3-week replay with oracle `R_t` scores RMSE
**13.44** where persistence scores 55.03.

It still lost, both as a soft penalty and as a decoder that reconstructs cases
through the mechanism. The decoder did exactly what it was built to do on the
growth axis — max log-growth **4.14** against the baseline's 0.07, so the flatness
was gone — and was 6 RMSE worse anyway.

### The number that explains all of it (EXP-024)

Every causally available physics anchor is worse than persistence, artifact-free:

| anchor | mean RMSE |
|---|---|
| persistence | **29.52** |
| `force` (R = 1) | 35.45 |
| `R̂ · force` | 50.01 |
| ratio form, damped | 33.05 |

Because the predictive content is entirely in `R_t`, and `R_t` is only **26%
predictable** out of sample — from its own past, and from nothing else:

| predictor of `log R_t` | out-of-sample r² |
|---|---|
| `log R_{t−1}` | **0.263** |
| climate, linear | −0.025 |
| climate + the hump-shaped thermal response R₀'s sensitivity indices imply | −0.028 |
| susceptible depletion (8 / 26 / 52-week) | −0.011 / −0.029 / −0.021 |
| own past + depletion | 0.263 |

`sd(log R) = 0.749`, so 26% predictability leaves a **1.90× multiplicative error
per step**, compounding over three weeks. Persistence carries no such factor —
cases at *t−1* explain r² = 0.85 directly. **Reconstructing a forecast through R
re-injects a factor-of-two error where persistence had none.** That single fact
explains the anchor table, the decoder, and retrospectively EXP-004, EXP-005,
EXP-008 and EXP-014.

Two mechanisms were tested properly and are inert for reasons worth stating:

- **Susceptible depletion** (`R_eff = R₀·s`) is real but ~500× too slow. Even at
  1-in-20 under-ascertainment it drifts ≈ 0.0013 per week against `sd(log R)` of
  0.749. Sri Lankan dengue is endemic in a population of 21.9M that never runs out
  of susceptibles on a 3-week horizon.
- **Climate driving R₀** through the sensitivity structure — `b` +1.000,
  `mu_v` −0.818, `m` +0.500, `nu_v` +0.318 — predicts a hump-shaped thermal
  response that a linear fit must miss. Allowing the curvature changes r² from
  −0.025 to −0.028.

No entomological temperature-response curves were invented to rescue this. Neither
source paper supplies them, and guessing one is precisely the EXP-014 error.

### And the direction that survived (EXP-025)

`R̂` is **backward**-looking: it correlates **+0.722** with the past three weeks of
growth and **−0.319** with the next three. Pushing a forecast toward `R̂·force`
pushes it the wrong way, which is the monotonic damage EXP-023 measured.

But a backward-looking state estimate is exactly what *detection* needs. Outbreaks
are rankable three weeks out even though the same models cannot count them:

| model | AUC | ΔAUC |
|---|---|---|
| current level ÷ threshold | 0.807 | — |
| level + growth | 0.811 | +0.004 |
| **level + `R̂`** | **0.826** | **+0.019** |
| level + neighbours | 0.813 | +0.006 |
| everything incl. climate | 0.820 | +0.013 |

`R̂` is the largest single addition — the first time in this project that the
mechanistic quantity measurably added anything. Around onset, mean `R̂` runs
1.13–1.24 for six weeks before the first week above threshold, against a global
median of 0.911: sustained `R > 1` is the textbook signature of a growing epidemic,
and it is visible before the outbreak is.

**The conclusion the project now carries:** the SEIR–SEI mechanism is useful for
outbreak *state estimation* and as a generative constraint, and demonstrably not
useful for point forecasting on endemic dengue — with six independent measurements
behind the negative. That is a stronger claim than a physics term that helps by
0.3 RMSE, and much stronger than one that does not help at all.

---

## 9.13 What the reproduction established

1. The published numbers **are real** — the code produces them.
2. The reported column is **training-inclusive**; the column names are effectively
   inverted relative to their meanings.
3. **Persistence beats every model** in that column, on both metrics, and was
   never tested.
4. RMSE is **per-window averaged**, worth 39% on this target.
5. The classical baselines **solve a different problem** on a different subset,
   worth 1.8× on MAE.
6. Our own baseline was **never underperforming**; it was the only one being
   measured honestly.
7. The adaptive graph is a **null result**, by three independent lines of
   evidence.
8. Accuracy is capped by **the data**: autocorrelation dominance, a +0.07 graph,
   corrupted covariates, and one week of bad reporting.
9. The **physics cannot close that cap**, because `R_t` is 26% predictable and
   routing a forecast through it costs more than it gains (§9.12). It can,
   however, rank outbreaks.

Point 6 is the one that changed the project. It converted "why can't we beat the
literature?" into "what does the literature's number actually mean?" — and that
question had an answer.

---

*Next: [Chapter 10 — Build It From Scratch](10_build_from_scratch.md)*
