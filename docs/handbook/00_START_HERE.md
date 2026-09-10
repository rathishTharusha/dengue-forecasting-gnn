# The Dengue Forecasting Handbook

**A complete course in this project — the disease, the mathematics, the machine
learning, the code, and every decision behind them.**

This handbook assumes you can read Python and nothing else. It does not assume
you know epidemiology, graph theory, neural networks, or time-series
forecasting. Everything is built up from first principles.

The goal is that after reading it you could **delete this repository and rebuild
it from scratch**, using only official documentation for NumPy, PyTorch and
PyTorch Geometric — no tutorials, no AI assistance, no guessing.

> **Status.** Current as of EXP-025. This handbook supersedes the earlier
> single-file `docs/COMPLETE_GUIDE.md`, which covered the same ground and was
> removed in the Phase-2/3 cleanup — its one unique section, the residual
> collapse, is now Chapter 5 §5.4 Trap 6. The Phase-2/3 implementation the older
> chapters described is at tag `phase23-archive`.

---

## What this project actually is

One sentence: **we forecast weekly dengue cases in Sri Lanka's 25 districts,
three weeks ahead, using a neural network that treats the districts as a graph.**

But that sentence describes the *starting* ambition. What the project has
actually produced is more interesting, and more useful:

1. **An exact reproduction** of the benchmark paper's results — running the
   authors' own code and recovering their published numbers to within 7.3%,
   two of them to four decimal places.
2. **A demonstration that those numbers do not mean what they appear to mean** —
   the column the paper reports as "cross validated" is measured on data the
   model trained on, and a naive baseline the paper never tested beats every
   model in it.
3. **A measured explanation of why accuracy stops where it does** — the ceiling
   is a property of the data, not of the architecture.
4. **A series of negative results**, each one killing a plausible idea with
   evidence.

If that sounds like a disappointment, read Chapter 1. It is not. A project that
knows exactly what it cannot do, and why, is worth more than one that reports a
number it cannot defend.

---

## Reading order

Read straight through. Each chapter depends on the ones before it.

| # | Chapter | What you learn | Prereqs |
|---|---|---|---|
| 1 | [The Problem and the History](01_problem_and_history.md) | Dengue, why forecasting is hard, how this project evolved and why | none |
| 2 | [Epidemiological Modelling](02_epidemiology_theory.md) | Compartmental models, SEIR, SEIR–SEI, R₀, the next-generation matrix, sensitivity analysis | high-school algebra |
| 3 | [The Dataset](03_the_dataset.md) | Where every number came from, all 11 channels, and six defects we found in it | 1 |
| 4 | [Machine Learning Foundations](04_ml_foundations.md) | Supervised learning, gradient descent, overfitting, time-series windowing, normalisation, leakage | 1 |
| 5 | [Evaluation — the part everyone gets wrong](05_evaluation.md) | Metrics, baselines, cross-validation for time series, aggregation traps | 4 |
| 6 | [Graph Neural Networks from Scratch](06_gnn_theory.md) | Graphs, adjacency, message passing, GCN derived from first principles, attention, diffusion | 4 |
| 7 | [The Five Architectures](07_architectures.md) | STGAT, A3TGCN, ASTGCN, DCRNN, AAGCN — what each one actually computes | 6 |
| 8 | [Code Walkthrough](08_code_walkthrough.md) | Every module, every function, line by line | 2–7 |
| 9 | [The Reproduction Story](09_reproduction_story.md) | What we ran, what we found, and every mistake made along the way | 5, 7 |
| 10 | [Build It From Scratch](10_build_from_scratch.md) | A staged implementation plan, with checkpoints | all |
| 11 | [Glossary and References](11_glossary.md) | Every term and symbol, and the papers | — |

**If you have one hour**, read Chapter 1, then §5.4 (why persistence is hard to
beat), then Chapter 9's summary table.

**If you are here to implement**, read 4 → 5 → 6 → 7 → 10, and use 8 as a
reference while you write.

---

## The five results you should be able to explain when you finish

If you can explain these five things to someone else, you have understood the
project.

**1. Why persistence is hard to beat.** Weekly dengue counts have a lag-1
autocorrelation of r = 0.92, meaning last week's count explains 85% of this
week's variance. The best meteorological covariate explains 2%. A forecaster
that simply repeats last week's number is therefore an extremely strong
baseline, and most published models on this dataset do not beat it.

**2. Why the benchmark paper's headline number is not a bar to clear.** Its
"Cross Validated" column is computed on a data loader that includes the training
windows — about 70% of each reported evaluation set is data the model was fitted
on. Its RMSE is also averaged per-window rather than pooled, which reports a
figure roughly 39% lower on this heavy-tailed target.

**3. Why the graph might not be doing any work.** Neighbouring districts
correlate at r = 0.62; non-neighbouring districts correlate at r = 0.55. The
graph structure adds only +0.07 above a national common trend. When we removed
message passing entirely, performance was statistically indistinguishable.

**4. Why one fold dominates every result.** Six windows out of 68 in the hardest
fold — the ones overlapping a single week of corrupted data — carry 90% of that
fold's squared error. That fold's RMSE is largely a measurement of a reporting
backlog.

**5. What would actually be needed to improve.** Not a better architecture.
Either better data (the covariates are 4% zero-filled and one week is a
19× artifact), or a genuinely different approach to the one failure mode that
matters: extrapolating to an outbreak larger than anything in the training set.

---

## How to verify anything in this handbook

Every quantitative claim here is reproducible. The commands:

```bash
# The dataset findings (Chapter 3)
jupyter lab analysis/notebooks/E1_dataset_eda.ipynb

# The graph experiment (Chapter 9)
jupyter lab analysis/notebooks/E2_adaptive_graph.ipynb

# Our independent reimplementation and its agreement tests (Chapter 8)
cd crosscheck && pytest

# The exact reproduction of the benchmark (Chapter 9)
python reproduction/verify_local.py
```

If a number in this handbook disagrees with what those produce, **the notebook is
right and this document is stale**. Report it.

---

## A note on how this project treats being wrong

You will find retractions in this handbook. A finding about seasonality was
published, then withdrawn when a sharper test contradicted it. An earlier claim
about which persistence figure to compare against was corrected. A benchmark
script had a bug that flattered someone else's work.

These are left in, visibly, with the reasoning. Two reasons. First, a research
record that only shows successes teaches you nothing about how research actually
proceeds. Second, and more practically: **the errors in this project have almost
all been in the evaluation, not the model.** Learning to distrust your own
measurement is most of the skill here.

---

*Next: [Chapter 1 — The Problem and the History](01_problem_and_history.md)*
