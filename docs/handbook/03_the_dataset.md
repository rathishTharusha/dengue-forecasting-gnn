# Chapter 3 — The Dataset

*Prerequisites: Chapter 1.*

Everything in this project is downstream of one file. If you misunderstand it,
every number you produce afterwards is wrong in a way no amount of careful
modelling will reveal. This chapter establishes exactly what is in it.

**Rule for this chapter:** every claim below is computed, not quoted. Where a
project document disagrees with the array, the array wins.

---

## 3.1 The file

```
notebooks/baseline/sri_lanka_2013-2022_shifted.npy      shape (459, 25, 11)
notebooks/baseline/sri_lanka_adj_list.json              25 districts
```

A NumPy array with three axes:

| axis | size | meaning |
|---|---|---|
| 0 | 459 | consecutive weeks, 2013–2022 |
| 1 | 25 | Sri Lankan districts |
| 2 | 11 | feature channels |

```python
import numpy as np
N = np.nan_to_num(np.load(path, allow_pickle=True)).astype(float)
cases = N[..., 5]          # (459, 25) — the forecast target
```

Two details in that load line matter enormously and are covered in §3.4:
`nan_to_num` and the choice of index 5.

### District ordering — a subtle trap

Axis 1 has no labels. The mapping from position to district name comes from
sorting the adjacency JSON's keys:

```python
DISTRICTS = sorted(json.loads(adj_path.read_text()))
```

If you build the adjacency matrix with one ordering and index the array with
another, **every model still trains and every metric still computes**. You get a
graph that connects the wrong districts and no error message. This class of bug —
silent, plausible-looking, undetectable from the loss curve — is the reason this
project pins the ordering in one place and derives everything from it.

---

## 3.2 The eleven channels

These were **not** taken from the paper's prose. Earlier revisions of
`docs/DATA.md` did that and got it wrong — it described a land-surface-temperature
channel that does not exist.

They were identified empirically: each of the 11 channels was matched against
every named column of the authors' `sri_lanka_2013-2022_vertical.csv` at every
candidate lag. All eleven matched **exactly** — not approximately, but
bit-for-bit.

| index | channel | lag (weeks) | what it is |
|---|---|---|---|
| 0 | `meanTair_F_Inst` | 0 | mean 2 m air temperature (**Kelvin**) |
| 1 | `minTair_F_Inst` | 0 | minimum air temperature |
| 2 | `maxTair_F_Inst` | 0 | maximum air temperature |
| 3 | `meanQair_F_Inst` | 0 | specific humidity |
| 4 | `meanSoilmoi0_10Cm_Inst` | 0 | soil moisture, 0–10 cm |
| **5** | **`cases`** | 0 | **weekly dengue cases — the target** |
| 6 | `meanCanopint_Inst` | 12 | canopy interception (water held on leaves) |
| 7 | `meanPrecipitationcal` | 12 | mean precipitation |
| 8 | `minPrecipitationcal` | 12 | minimum precipitation |
| 9 | `maxPrecipitationcal` | 12 | maximum precipitation |
| 10 | `minNdvi` | 17 | minimum vegetation index |

### What "lag-shifted" means, and the mistake it invites

The filename ends in `_shifted`. Channel 7's value at week *t* is **not** the
rainfall in week *t* — it is the rainfall from week *t−12*, already moved forward.

The reason is the causal chain from Chapter 1. Rain fills containers, larvae
mature, mosquitoes bite, humans incubate, cases get reported. Roughly 12 weeks
elapse. Pre-shifting the array means a model reading week *t* sees the rainfall
that plausibly *caused* week *t*'s cases.

**The mistake:** applying your own lag on top. You would then be feeding the model
rainfall from 24 weeks ago and wondering why the covariates are useless.

Three different lags exist because three biological processes have different
timescales: precipitation and canopy at 12 weeks (breeding-site formation),
minimum NDVI at 17 weeks (vegetation responds to rain, then supports mosquitoes).
These match the paper's §IV values, which independently confirms this is the array
it describes.

### What was left out

`meanNdvi`, `maxNdvi` and `meanPsurf_F_Inst` exist in the source CSV and were
**excluded**. Only `minNdvi` survives. The authors do not explain why. Recording
"we do not know why" is more useful than inventing a justification.

---

## 3.3 The target's distribution

```
median 13    max 2631    zeros 9.74%    skewness 8.6
```

Nearly a tenth of all district-weeks are exactly zero, and the largest is 200×
the median. Three consequences drive every modelling decision in this project.

**1. Squared error becomes a measurement of outliers.** An error of 500 on an
outbreak week contributes 250,000 to the sum; an error of 5 on a typical week
contributes 25. Ten thousand typical weeks weigh the same as one outbreak week.

**2. Percentage error is undefined.** MAPE divides by the true value. On 9.7% of
observations that is a division by zero. Chapter 5 shows what different papers do
about this and why it changes their headline numbers.

**3. Training is unstable.** Gradients from outbreak weeks dwarf everything else,
so the optimiser spends its capacity on the tail.

The response is `log1p`:

$$z = \log(1 + y), \qquad y = e^{z} - 1$$

```python
z = np.log1p(cases)         # log(1+y): defined at y=0, unlike log(y)
y = np.expm1(z)             # exact inverse
```

Use `log1p`/`expm1` rather than `log(1+x)`/`exp(x)-1`: the library versions stay
numerically accurate for small *x*, where the naive forms lose precision to
cancellation.

After the transform the median sits at 2.6 and the max at 7.9 — a range of 3×
rather than 200×. Squared error now weights weeks comparably.

**Crucially, scoring still happens in raw counts.** Predictions are inverted out
of log space before any metric is computed. An RMSE in log space is a different
quantity with a different meaning, and reporting one as if it were the other is a
common and serious error.

---

## 3.4 Six defects

This is the substance of the chapter. Every one was found by measurement.

### D1 — 4% of weather values are `0`, not missing

The array contains no NaNs. That is not because the data is complete; it is
because whoever produced it called `np.nan_to_num`, which replaces every NaN with
`0.0`.

For a temperature in Kelvin, this is catastrophic. Real values sit around 298 K.
The fill value is 0 K — 290 K away from every genuine observation, and physically
impossible.

```
meanTair std including zeros:  58.72
meanTair std excluding zeros:   2.23
real 13.8 K spread = 0.23 sd   (should be 6.16 sd)
usable dynamic range compressed 26x
```

**Why this destroys the channel.** Standardisation computes $(x - \mu)/\sigma$.
With $\sigma$ inflated 26× by the fill values, every genuine temperature
difference is divided by a number 26× too large. The entire real signal is
squeezed into 3.8% of the standardised range, while the fill values sit five
standard deviations away. A neural network sees a binary "missing or not" feature
with a trace of temperature riding on it.

**Does it invalidate the reproduction?** No — and this is exactly why measuring
beats assuming. The five reproduced GNNs run with `use_disease_only=True` and
**never read these channels**. The defect is invisible to every published result.

It is fatal to anything that *does* read them: a physics-informed loss reading
temperature, or a GAN conditioned on meteorology.

**The fix** is masking or interpolation, never zero-filling — the gaps are
scattered rather than one contiguous block, so interpolation is well-posed — and
normalising over observed values only.

### D2 — week 395 is a reporting artifact

```
week 395 national total:            7,165 cases
median of weeks 390-400:              380 cases
ratio:                                 19x
districts simultaneously above 100:   18 of 25
```

Real epidemics have spatial structure. They start somewhere, spread along travel
routes, and take weeks to peak. Eighteen districts spiking in the same week and
collapsing the next is not epidemiology; it is a **reporting backlog** — a batch
of accumulated cases entered at once.

Compare the genuine 2017 outbreak, which peaks at week 221 with 10,535 national
cases: comparable magnitude, but reached over months with a coherent spatial
progression.

**Why this is the most consequential fact in the project.** Under the
rolling-origin protocol, week 395 lands in the **test** split of the last fold. In
that fold, **6 test windows out of 68 — 9% — carry 90% of the total squared
error.** Persistence RMSE there is 68.62 with those windows and 22.80 without.

A headline RMSE that does not say which side of that split it is on is largely a
measurement of a data-entry event.

The project's response is to **flag, never drop**:

```python
def artifact_windows(window_index, window, horizon):
    """True where a window's input or target overlaps week 395."""
    idx = np.asarray(window_index)
    return (idx - window <= ARTIFACT_WEEK) & (ARTIFACT_WEEK < idx + horizon)
```

Dropping them flatters every model equally and measures nothing. Reporting both
numbers tells the reader which regime they are looking at.

Note also what this rules out. An earlier plan to **mask the artifact from
training** was implemented and then removed, because measurement showed week 395
falls in *no* training split under this protocol. The increment was a no-op. Its
removal is documented in `analysis/lib/improved.py` rather than quietly reverted.

### D3 — the 2017 outbreak is in test for exactly one fold

Annual national totals, 2013–2020:

```
60,372   31,234   22,711   48,075   154,000   48,008   96,705   24,560
```

The largest year is **6.8×** the smallest. 2017 alone is 2.6× the next largest.

Under rolling-origin CV this outbreak falls in the **test** split for the earliest
origin and in the **training** split for every later one. Those folds are not the
same task:

- Early fold: predict an outbreak **larger than anything seen in training** —
  extrapolation.
- Later folds: predict quiet years having *already seen* an outbreak —
  interpolation.

This single fact explains the cross-fold variance in the published results.
ASTGCN's ±19.85 MAE spread across segments is not instability; it is two different
problems being averaged together.

It also identifies where genuine improvement would show. Every model handles the
interpolation folds comfortably and fails on the extrapolation fold. That fold is
the whole problem.

### D4 — the adjacency graph is not symmetric

```
Kandy    lists Ampara     |  Ampara   does not list Kandy
Kegalle  lists Kalutara   |  Kalutara does not list Kegalle
```

Adjacency is a symmetric relation — if A borders B, then B borders A. Two pairs
violate it, which means the released list contains a transcription error. (Neither
pair is plausibly adjacent geographically, so the likelier reading is a spurious
edge than a missing one; but either way it is a defect, not a modelling choice.)

```
116 directed edges  ->  141 with self-loops
mean degree 4.72
Jaffna: degree 1  (its only neighbour is Kilinochchi)
```

Jaffna at degree 1 is a real structural fact, not an error — it sits at the tip of
the northern peninsula. But it means Jaffna's representation is a function of one
other district, so message passing gives it almost nothing.

The project symmetrises, and says why in the code:

```python
# Symmetrise: the released list has two one-way edges (EDA F7), and an
# adjacency relation that is not symmetric is a data defect, not a modelling
# choice. Both arms get the same corrected graph.
a = np.maximum(a, a.T)
```

The last sentence is the important one. Both experimental arms get the *same*
corrected graph, so the correction cannot advantage either.

### D5 — the graph adds only +0.07 of correlation

Does adjacency carry signal at all? Test it directly: correlate every pair of
districts' case series, then split the pairs by whether they are neighbours.

```
neighbouring pairs:      r = 0.62
non-neighbouring pairs:  r = 0.55
difference:            +0.07   (p = 0.0002)
```

**The effect is real** — p = 0.0002 is not chance. **The effect is small** — 0.07
above a floor of 0.55.

The right thing to look at is that 0.55 baseline: *any* two districts in Sri Lanka
correlate at 0.55, because they share a national epidemic trend and a national
climate. Adjacency adds a further 0.07 on top of that common mode.

This is the quantitative reason the graph contributes so little in Chapter 9's
experiments. It is not that the graph is wrong. It is that most of what it could
transmit is already present in every node's own history.

### D6 — the seasonality finding, and its retraction

**This one is included because it was our own error.**

An earlier analysis reported bimodal seasonality with a 2.9× peak-to-trough ratio
and recommended a week-of-year feature as the top-priority improvement.

That finding was **wrong and has been retracted.** The sharper test:

| test | result |
|---|---|
| correlation between de-meaned log yearly shapes | r = **−0.065** (chance) |
| districts with a significant week-of-year effect | **0 of 25** |
| R² of week-of-year alone | **0.03** |
| R² of last week's value alone | **0.86** |

**The cause of the error.** The seasonal profile was computed by averaging each
week-of-year across all years. But annual totals span 6.8× (D3). Averaging raw
counts across such unequal years lets 2017 dominate completely — so the "seasonal
cycle" was a picture of *2017's specific timing*, presented as a recurring
pattern. Normalising each year's amplitude before averaging makes it vanish.

An earlier experiment (EXP-012) had reached the correct conclusion first and was
not consulted.

The retraction is left visible: in the notebook, in `README.md`, in the experiment
log, and in `analysis/lib/improved.py` under a heading that reads *Deliberately
excluded*. It is more instructive than the finding would have been.

---

## 3.5 Two autocorrelation numbers, and why both are right

`docs/DATA.md` says lag-1 autocorrelation is ≈ 0.68. Chapter 1 says 0.92. Both
were computed from this array. Neither is a typo.

```python
# pooled: stack every district-week into one long vector
np.corrcoef(cases[:-1].ravel(), cases[1:].ravel())                          # 0.9210

# per-district: correlate within each district, then average
np.mean([np.corrcoef(cases[:-1, i], cases[1:, i])[0, 1] for i in range(25)])  # 0.6785
```

The pooled figure is **inflated by between-district scale differences**. Colombo
consistently reports hundreds of cases and Mannar consistently reports single
digits, so a pooled correlation partly measures "big districts stay big" rather
than "this week resembles last week". The per-district average removes that and is
the honest measure of temporal persistence.

Per-district values range from **0.014 to 0.944** — some districts are almost
perfectly persistent, others nearly noise.

**Neither number is wrong; they answer different questions.** Chapter 1 quotes the
pooled figure because the models are pooled across districts, so it describes what
persistence actually achieves in this setup. Any analysis of temporal structure
should quote 0.68.

If you take one habit from this chapter, take this one: **when you report a
correlation on panel data, state whether it is pooled or within-unit.** They can
differ by 0.24 and both be correct.

---

## 3.6 What this chapter licenses you to conclude

1. Channel 5 is the target; the other ten are covariates lagged 0, 12 or 17 weeks.
2. The covariate channels are **corrupted by zero-filling** and unusable as-is.
3. One week is a reporting artifact that dominates one fold's error.
4. One year is 6.8× a typical year and lands in test for exactly one fold.
5. The graph carries real but weak signal (+0.07 above a 0.55 common mode).
6. There is **no usable annual cycle**.
7. Correlations on panel data must state pooled versus within-unit.

Every one is reproducible in `analysis/notebooks/E1_dataset_eda.ipynb`, where the
number is computed by the cell above the sentence that states it.

---

*Next: [Chapter 4 — Machine Learning Foundations](04_ml_foundations.md)*
