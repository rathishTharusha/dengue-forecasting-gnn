# Handover — beating the persistence floor

**Branch:** `exp/beat-baseline` · **Goal:** get an arm below the artifact-free
persistence floor of **29.52** with a p-value that survives clustering *and*
multiple-comparison correction.

Everything below is built, smoke-tested and ready to run. You run it, download
the JSON, and one script turns it into the decision table. Nothing here needs a
judgement call from you mid-run.

---

## 0. What we are actually fixing

Two things, and it is worth being clear which is which, because they need
different remedies.

### The result is not stuck at 0.47 — it is stuck at n=3

A3TGCN sits at 29.05 against a floor of 29.52, reported as `p=0.10` at n=9. That
p-value counts training runs, not evidence. Within an origin, seed variance on
this task is **0.02–0.05**; between origins it is **22.07 / 37.74 / 29.35**.

| origin | A3TGCN | floor | delta |
|---|---:|---:|---:|
| 0.55 | 27.35 | 26.88 | **+0.47** |
| 0.70 | 37.74 | 38.89 | −1.15 |
| 0.85 | 22.07 | 22.80 | −0.72 |

Nine rows, three observations. Clustered by origin: **−0.467, p=0.436**. Adding
seeds cannot fix this. Only adding origins can, and every run below uses **9
disjoint origin blocks** instead of 3 overlapping ones.

### There is a real, named bias to remove

The pipeline **trains on `log1p` and scores on counts**. For `log(1+y) = m + e`,
`expm1(m)` is the conditional *median*; RMSE is minimised by the conditional
*mean*, `E[y] = exp(m)·E[exp(e)] − 1`. So it under-predicts **by construction**,
by an amount that grows with residual variance.

That is exactly what `error_diagnosis.json` already measured and nobody read as a
bias: A3TGCN is **−12.92 on outbreak windows and +0.38 on quiet ones**. High
variance → large bias; low variance → none.

Rough headroom: removing the outbreak bias cuts outbreak MSE ~3.8%, and outbreak
windows carry 61.7% of squared error → **≈0.35 RMSE**, comparable to the entire
margin in play.

---

## 1. The full experiment matrix

Four independent levers. Each is measured against the same floor on the same
windows, and none of them changes what the network learns except where stated.

| # | Lever | Arms | Where |
|---|---|---|---|
| **A** | Statistical power | 9 disjoint origins (was 3 overlapping) | every run |
| **B** | Retransformation bias | `raw`, `smear`, `lognorm`, `calib` | every run |
| **C** | Combination | `blend_*` (with persistence), `ens_*` (seeds), `multi_*` (architectures) | every run |
| **D** | Objective | heads `det`, `gauss`, `nb` | beat-floor kernels |
| **E** | Information | features `cases`, `causal`, `climate` | covariate kernels |
| **F** | Physics | `spatial` constraint on A3TGCN | existing physics kernels |

### B — the four estimators

All fitted **on validation only**, applied per horizon, capped at 3×.

- `raw` — the existing pipeline, `expm1(m)`. The control.
- `smear` — Duan's smearing, `S = mean(exp(e))` on validation residuals.
  Nonparametric.
- `lognorm` — the Gaussian case, `S = exp(v/2)`. Under a `gauss` head, `v` is
  predicted per window, so the correction is **heteroscedastic** and largest
  exactly where the measured bias is largest.
- `calib` — one factor per horizon fit directly against validation RMSE.
  Subsumes the others as an estimator, and doubles as a diagnostic.

### C — combination

- `blend_*` — `w·model + (1−w)·persistence`, `w` per horizon by least squares on
  validation, clipped to `[0,1]`. Fifty years of forecast-combination literature
  says this is the single most reliable thing you can do with a strong naive
  baseline, and persistence here has lag-1 `r² = 0.85`.
- `ens_*` — mean of the seeds' count-space predictions. Pure variance reduction,
  needs no calibration.
- `multi_*` — mean across A3TGCN/STGAT/ASTGCN. Only exists in the combination
  kernel, because it needs every member in one session.

### D — heads

- `det` — Huber on `log1p`. The current pipeline.
- `gauss` — Gaussian NLL; predicts a per-window variance.
- `nb` — **negative binomial on raw counts**, `Var = μ + αμ²`, emitting the
  conditional *mean* directly. Where B repairs the bias afterwards, this never
  creates it. NB2 rather than Poisson because the target is heavily overdispersed
  (median 13, max 2631, 9.7% zeros). For this head `calib` is a **test**: a
  factor near 1.0 means the count likelihood genuinely removed the bias.

### E — covariates

Ten of eleven channels are currently thrown away (`cases = raw[..., 5]`).

| set | channels | what it is |
|---|---|---|
| `cases` | 1 | univariate control — every covariate arm must beat this |
| `causal` | 8 | humidity, soil moisture, mean temp, mean precip (+ indicators) |
| `climate` | 17 | everything (+ indicators) |

**Two data problems were found and fixed before this could run.** `np.nan_to_num`
turned missing GLDAS values into zeros, and a 0 K temperature is 290 K from every
real value. Measured, the missingness is almost entirely **one district**:
`Jaffna` is zero across all five GLDAS channels for all 459 weeks — 1 in 25,
which is where `DATA.md`'s "~4%" comes from. Interpolating along time cannot
reach it, so it is filled from its **graph neighbours**, with an indicator
channel marking every synthetic value. `tests/test_features.py` also pins that
covariate normalisation does not leak across the split.

---

## 2. Step by step

### Step 0 — push the branch (required; kernels clone from GitHub)

```bash
git push -u origin exp/beat-baseline
```

### Step 1 — preflight (run this FIRST, ~10 minutes)

Smokes all ten architecture × head × feature combinations at `--quick`. Its
numbers are degraded and **must never be reported** — the only output that
matters is the PASS/FAIL table. A wide multivariate input changes the input width
every architecture is built with (3 → 51), which is exactly what fails at
construction time after a kernel has already burned four hours.

```bash
cd reproduction/kaggle/kernels/beat-floor-preflight && kaggle kernels push
```

Wait for it, open it, confirm **10/10 configurations run**. If anything fails,
stop and send me the failing line — do not launch step 2.

### Step 2 — launch the sweeps (parallel, ~2–5 h each)

```bash
for k in beat-floor-a3tgcn beat-floor-stgat beat-floor-astgcn beat-floor-aagcn beat-floor-dcrnn covariates-a3tgcn covariates-stgat beat-floor-combination; do (cd reproduction/kaggle/kernels/$k && kaggle kernels push); done
```

Kaggle runs these concurrently. Priority if you'd rather stagger them:

1. `beat-floor-a3tgcn` — the best architecture, all three heads
2. `covariates-a3tgcn` — the largest untested information lever
3. `beat-floor-combination` — the cross-architecture ensemble
4. everything else

### Step 3 — also launch the physics kernels

These were already built and are still gated on nothing now that the branch is
pushed. They test the spatial constraint on **A3TGCN**, which has never been run.

```bash
for k in physics-sweep-a3tgcn physics-sweep-stgat physics-sweep-astgcn physics-sweep-aagcn physics-sweep-dcrnn; do (cd reproduction/kaggle/kernels/$k && kaggle kernels push); done
```

### Step 4 — check status

```bash
kaggle kernels list --mine --page-size 30
```

### Step 5 — download outputs

```bash
mkdir -p analysis/results/beat_baseline
for k in beat-floor-a3tgcn beat-floor-stgat beat-floor-astgcn beat-floor-aagcn beat-floor-dcrnn covariates-a3tgcn covariates-stgat beat-floor-combination; do kaggle kernels output tharushaperera16/$k -p analysis/results/beat_baseline; done
```

### Step 6 — build the decision table

```bash
python analysis/_build/merge_beat.py --dir analysis/results/beat_baseline --out analysis/results/beat_summary.md
```

This clusters by origin (one observation per independent fold, seeds averaged
first) and applies **Benjamini–Hochberg** across the whole family. That second
part is not optional: this sweep evaluates on the order of a hundred arms, so at
α=0.05 roughly five spurious "wins" are expected by chance, and an uncorrected
minimum over the table is not a result.

The script prints one of two conclusions, and both are publishable:

- *"N arm(s) beat the floor and survive Benjamini-Hochberg"* — we have a result.
- *"No arm beats the floor after correcting for multiple comparisons"* — then the
  informational-ceiling argument is now backed by a genuinely exhaustive search
  at 3× the statistical power, which is a considerably stronger version of the
  current paper's claim than it can make today.

---

## 3. What to send me

1. The preflight PASS/FAIL table.
2. `analysis/results/beat_summary.md`.
3. The console output of step 6.
4. Anything that errored, with the failing line.

I will then log it as **EXP-026** in `docs/EXPERIMENT_LOG.md` with the exact
config, commit SHA, seeds, folds and unrounded table, and write the surviving
findings into the paper.

---

## 4. Reading the output before I see it

`vs_floor` is negative when the arm beats persistence. Two columns decide
everything:

- **`wins`** — how many of the 9 origins it won. 9/9 or 8/9 is a real effect;
  5/9 is noise regardless of the p-value.
- **`BH`** — survives multiple-comparison correction. Treat a nominally
  significant row that fails BH as a candidate to re-run with more origins, not
  as a finding.

Expected ordering if the bias diagnosis is right: `calib` and `smear` beat `raw`;
`blend_*` beats its unblended twin; `ens_*` beats its single-seed twin; and the
`nb` head's `calib` factor sits near 1.0 while `det`'s sits above it.

If `raw` ends up beating the corrected arms, the retransformation argument is
wrong and I want to know — that is a clean negative result and it goes in the
log too.

---

## 5. Local commands (no Kaggle needed)

```bash
python -m pytest -q
```

```bash
python analysis/_build/run_beat_baseline.py --arch A3TGCN --quick --features causal
```

The second needs `torch_geometric_temporal`, which is not installed locally — it
is the Kaggle-pinned stack. It will run once you are in an environment that has
it; otherwise the preflight kernel is the smoke test.

---

## 6. Known caveats, stated up front

- **Multivariate inputs are flattened**, `(nodes, features × window)`, so every
  architecture takes them without rewiring. A3TGCN could take
  `(nodes, features, window)` natively and would probably do better with it; if
  `causal` or `climate` shows promise, that is the obvious follow-up.
- **Origins overlap in training data**, necessarily — expanding-window folds
  share history. Only the *test* blocks are disjoint, which is what the
  clustering assumes.
- **DCRNN runs 3 seeds, not 5.** It is the slowest architecture and +1.78 above
  the floor; it should not take wall clock it cannot repay.
- **`--quick` numbers are degraded by design** and must never reach the log or
  the paper.
