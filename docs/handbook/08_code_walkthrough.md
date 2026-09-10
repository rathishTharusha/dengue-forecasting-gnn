# Chapter 8 — Code Walkthrough

*Prerequisites: Chapters 2–7.*

A tour of every module, what it does, and — where it is instructive — why it is
written the way it is. Use this as a reference alongside Chapter 10.

---

## 8.0 Repository map

```
dengue-forecasting-gnn/
├── src/dengue_gnn/          # torch-free shared library: metrics, seir, data
├── crosscheck/              # independent re-implementation; shares NO code with src/
├── reproduction/            # runs the ORIGINAL authors' code on Kaggle
├── analysis/                # EDA, the verified baseline, the physics, the sweeps
├── notebooks/               # Phase-1 deliverable + generated reproduction notebooks
├── docs/                    # decisions, experiment log, this handbook
├── results/                 # index + the SEIR validation table
├── scripts/                 # verify_seir_paper.py
└── tools/                   # repo hygiene checks
```

**The three-workspace split is deliberate** and Chapter 1 §1.5 explains it. The
critical rule:

```python
"""Independent re-implementations for the paper cross-check workspace.

This package deliberately shares **no code** with ``src/dengue_gnn``. Every
formula here is re-derived from the paper that defines it, so that agreement
between the two implementations is evidence, and disagreement is a finding.
Importing ``dengue_gnn`` from this package would defeat its only purpose.
"""
```

If `crosscheck` imported `src`'s SMAPE, a bug in the denominator would agree with
itself perfectly and prove nothing.

---

## 8.1 `src/dengue_gnn/` — the project library

### `__init__.py` — the promotion rule

```python
"""Shared library code for the dengue forecasting GNN project.

Code starts life in ``notebooks/`` and moves here once it stabilizes and is used
in more than one place, so that every phase of the project scores its results
with exactly the same implementation.

Torch-dependent modules (``models``, ``losses``) are not imported here, so that
[CI can run without a deep-learning stack]
"""
```

Two policies worth adopting in your own work:

**Promotion.** Code is not written in `src/` from the start. It is written in a
notebook, and moves once it stabilises *and is used twice*. Premature abstraction
in a research repo produces interfaces designed around one use case.

**Torch isolation.** `models` and `losses` are not imported at package level, so
`import dengue_gnn` works without torch installed. CI installs only
`numpy pytest ruff` and still tests all the metric code. Anything importing torch
runs on Colab or Kaggle instead.

### `metrics.py` — the single scoring implementation

Extracted numerically verbatim from the Phase-1 notebook. **Never re-implement
scoring in a notebook — import this.**

The contract:

- inputs are **raw counts, already inverse-transformed** out of log1p/residual space
- horizon on the last axis

`tests/test_metrics.py` guards it, because a silent change to the SMAPE
denominator or the MAPE mask shifts every reported number in the project without
breaking anything visibly.

### `data.py` — the minimal loader

Three lines of real work, and it exists for one reason: CI installs numpy, pytest
and ruff only, so anything importing torch cannot be covered there.
`tests/test_seir.py` and `scripts/verify_seir_paper.py` both need the case series
to check the derived growth ceiling against what the record actually contains, and
neither needs torch. The richer loader that builds the graph and the folds lives in
`analysis/lib/adaptive.py` and does pull in torch.

### `seir.py` — the compartmental model

Implements Phaijoo & Gurung (2018). Not fitted to data; used to **derive
constraints**.

```python
@dataclass
class Params:
    b: float; beta_h: float; beta_v: float; m: float
    mu_v: float; nu_v: float; nu_h: float; gamma_h: float
    mu_h: float = 0.000046

    @property
    def alpha(self):  return self.b * self.beta_h * self.m
    @property
    def delta(self):  return self.b * self.beta_v
    @property
    def beta(self):   return self.nu_h + self.mu_h
    @property
    def gamma(self):  return self.gamma_h + self.mu_h
    @property
    def eps(self):    return self.mu_v
```

Composite parameters as **properties** rather than stored fields, so they cannot
drift out of sync with the primitives they derive from. Change `b` and `alpha`
follows automatically. Storing both invites the classic bug where one is updated
and the other is not.

Key functions:

| function | returns |
|---|---|
| `r0(p)` | $\sqrt{\alpha\delta\nu_h\nu_v / (\beta\gamma\epsilon(\epsilon+\nu_v))}$ |
| `growth_rate(p)` | spectral abscissa of $F - V$ — initial exponential growth rate/day |
| `sensitivity_indices(p, h=1e-6)` | normalised forward sensitivities, central differences |
| `weekly_log_growth_ceiling(...)` | $7\lambda$ — the weekly bound |

`PAPER_SECTION4` fixes the paper's §4 values, with `m = 2500000/(0.25 × 5071126)`.

**The reparameterisation note** (Chapter 2 §2.5) lives in this module's docstring:
`mu_v` comes out at −0.818 here against the paper's −1.318, a difference of
exactly 0.5, because this module holds $m = N_v/N_h$ fixed while the paper holds
$\pi_v$ fixed. Neither is wrong. The docstring says so, which is why nobody has
"fixed" it.

### `losses.py` — removed, and what it taught

The Phase-2 objective added a graph-Laplacian smoothness term and a non-negativity
term on raw counts. Both are gone (tag `phase23-archive`), but two lessons from
them survive and are worth more than the code was.

**A constraint has to act in the right space.** `nonnegativity_loss` penalised
negative *predicted case counts*, which is correct only on raw counts. Applied to
the network's residual output, "negative" does not mean "negative cases" — it means
"fewer cases than last week", which is a correct and frequent forecast. Penalising
it would have biased the model against ever predicting a decline, on a seasonal
disease. The API forced the right space precisely because the wrong one is so easy
to reach.

**The smoothness term was penalising geography.** It charged squared differences in
predicted counts between neighbouring districts — but counts differ by orders of
magnitude between Colombo and rural districts, so the term was mostly a tax on
population. The scale-free quantity is `R`, and EXP-025 measured whether a
Laplacian belonged there instead: **Moran's I for `log R` is 0.005**, against 0.274
for `log1p(cases)`. `R` has no local spatial structure, so the term has no correct
version on this data. Measuring before rebuilding saved a second wrong implementation.

One implementation detail is worth stealing verbatim:

```python
if lambda_phys == 0.0:
    return pred_counts.sum() * 0.0  # keeps the graph connected, contributes nothing
```

The `λ=0` ablation row is then **exactly** the unregularised model rather than an
approximation of it — and `* 0.0` rather than a bare `0.0` keeps the tensor in the
autograd graph, so gradients still flow correctly. The same trick is in
`analysis/lib/physics.py` today.

### The modules that were removed, and what they taught

The Phase-2/3 implementation was deleted once the work moved to architectures
verified against the published papers — keeping two generations side by side was a
standing invitation to confuse them. All of it is at tag `phase23-archive`, and
EXP-001 – EXP-014 keep their configs and unrounded numbers inline in the experiment
log, so those runs stay citable.

| module | purpose | why it went |
|---|---|---|
| `models.py` | Phase-2 GNNs, reconstructed from the framework section after the originals were lost in a git-ignored `scratch/` | superseded by `analysis/lib/reproduced.py`, whose architectures reproduce published numbers |
| `experiment.py` | rolling-origin harness ported from the Phase-1 notebook | superseded by `analysis/lib/adaptive.py`; its one surviving function is now `dengue_gnn/data.py` |
| `mechanistic.py` | constraints on observed quantities only | EXP-014: the corrected ceiling was inert, contributing zero gradient on all 96 rows. Its two derived constants moved into `seir.py` |
| `augment.py` | contribution (b), GAN augmentation | never reached a result; the design lesson below survives |
| `baselines.py` | non-graph competitors under our protocol | superseded by `reproduction/`, which runs the authors' own code |
| `provenance.py` | config + commit SHA + seeds at run time | folded into per-job JSON records |
| `results_logger.py` | appended result CSVs, loudly | ditto |

Three of these encode lessons worth repeating.

**`augment.py`:**

> **The comparator is not optional.** Any GAN must be judged against cheap
> augmentation — jittering and window warping — on *downstream* forecast accuracy,
> never on how realistic the synthetic series look.

A GAN that produces beautiful synthetic outbreaks and does not improve forecasts
has produced nothing.

**`baselines.py`:**

> The Phase-2 paper declined to compare numerically against Weng et al.,
> correctly, because their static 70/30 split is not commensurable with
> rolling-origin. The answer to that is not to skip the comparison but to
> **re-run the competitors under our protocol**.

That is the right response to incommensurable numbers, and it is what
`reproduction/` exists to make possible.

**`mechanistic.py`** is the cautionary one:

> `MAX_WEEKLY_LOG_GROWTH` was 0.70, guessed from `ln(8)/3`. The SEIR–SEI model
> actually implies **2.3884**. The old value declared **20.9% of the observed
> record physically impossible** and penalised forecasts for tracking real
> outbreaks.

**A guessed physics constant is not physics.** Derive it, then check what fraction
of your real data it forbids — 1.07% is the right order for a genuine ceiling;
20.9% means you have built a bug with a Greek letter on it.

**`results_logger.py`** was rewritten after a review finding:

> The previous version fell back to `0.0` for every metric whose column was
> missing, and to `1.0` for the learned gate — a value that reads as "the model
> used pure geography" rather than "this was never recorded". A logging failure
> that writes a plausible wrong number into the file a paper cites is worse than
> one that crashes, because nothing [surfaces it].

**Fail loudly. A plausible wrong number is worse than an exception.**

---

## 8.2 `crosscheck/lib/xcheck/` — the independent implementation

| module | contents |
|---|---|
| `metrics.py` | `rmse`, `mae`, `mape_weng`, `mape_masked`, `smape`, `morans_i`, `crps_gaussian`, `picp`, `mpiw` |
| `seir.py` | SEIR and SEIR–SEI, $R_0$ both ways, sensitivity indices analytic and numeric |
| `data.py` | Loading and windowing, **two ways on purpose** |
| `graph.py` | The DengueGNN hybrid adjacency (Eq. 2–4), torch-free |
| `models.py` | Minimal reimplementations |
| `protocol.py` | Segmentation and the metric conventions |

### The design principle: reproduce *and* correct

```python
"""Loading and windowing the Sri Lanka dengue array, two ways on purpose.

The Weng et al. reference implementation makes three data-handling choices that
this workspace reproduces *and* corrects, so the difference between them can be
measured rather than argued about
"""
```

Both paths are implemented. Running both and differencing turns "I think their
normalisation leaks" into "their normalisation shifts the scale by 6.5%".

**Measure the disagreement instead of asserting it.** Every finding in Chapter 5
came from this discipline.

### Defensive input handling

```python
def _pair(pred, truth):
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs truth {y.shape}")
    if p.size == 0:
        raise ValueError("cannot score empty arrays")
    return p, y
```

Every metric goes through this. It catches:

- **Shape mismatch** — otherwise NumPy broadcasting silently compares
  `(3,25)` against `(25,)` and returns a plausible number.
- **Empty arrays** — `np.mean([])` returns `nan` with a warning, and a `nan`
  quietly poisons every aggregate downstream.
- **dtype** — float32 accumulation error is visible at these magnitudes.

Three lines, and they eliminate an entire class of silent failure.

### Two implementations of one formula

```python
def seir_sei_r0(p, method="closed_form"):
    if method == "closed_form":
        num = p.alpha * p.delta * p.nu_h * p.nu_v
        den = p.beta * p.gamma * p.epsilon * (p.epsilon + p.nu_v)
        return float(np.sqrt(num / den))
    if method == "spectral":
        f, v = seir_sei_next_generation(p)
        return float(max(abs(np.linalg.eigvals(f @ np.linalg.inv(v)))))
```

They agree to 1.1 × 10⁻¹⁶. Similarly, `sensitivity_indices` (analytic) and
`sensitivity_indices_numeric` (finite differences) are both implemented and
checked against each other.

**Two independent derivations agreeing is strong evidence. One derivation is a
hypothesis.**

### The tests

`crosscheck/tests/test_xcheck.py` — 40 tests. The most important one:

```python
def test_agrees_with_project_metrics():
    ...
```

It imports **both** implementations and asserts identical output on the same
inputs. This is the only place the two workspaces meet, and it is the payoff of
keeping them separate everywhere else.

Others pin properties rather than values:

```python
def test_jensen():
    """Jensen: averaging per-window RMSE cannot exceed the pooled RMSE."""
```

A property test cannot be broken by a legitimate refactor, and cannot be
accidentally satisfied by a wrong implementation.

---

## 8.3 `analysis/lib/` — EDA and the improved baseline

Covered in detail across Chapters 4, 6 and 7. Summary:

| module | contents |
|---|---|
| `adaptive.py` | `load_dataset`, `Fold`, `build_folds`, `STGNN` (4 graph modes), `train_one`, `evaluate`, `pooled_scores`, `persistence_scores` |
| `reproduced.py` | The five architectures under our protocol |
| `improved.py` | `Increments`, `TemporalAttention`, `make_head`, `gaussian_nll`, `artifact_windows` |

### `improved.py` — every increment has a citation

```python
"""Architecture increments, each justified by a paper or an EDA finding.

Every option here has a citation. Nothing is included because it is fashionable
or because it might help; if an increment cannot be traced to something we read
or measured, it is not in this module.
"""
```

| increment | justification |
|---|---|
| `per_horizon_heads` | GulMohamed Eq. (11) + EDA F6: autocorrelation falls 0.92→0.82 across horizons, so a shared head averages over three different problems |
| `temporal_attention` | GulMohamed Eq. (6)–(8); their own ablation attributes only +0.5 RMSE, "the smallest of their three, so this is the increment we expect least from" |
| `probabilistic` | Eq. (12)–(14); the roadmap requires CRPS/PICP/MPIW, and a point forecast cannot supply them |
| `huber` | EDA: skewness 8.6, top 1% carry 18.3% of cases |

And a section headed **Deliberately excluded**, listing `week_of_year`
(retracted finding), `wider_window` (EXP-012 found nothing at 8 folds × 3 seeds),
and `learned_adjacency` (sign flips between implementations).

**Documenting what you rejected, and why, is as valuable as documenting what you
built.** It stops the next person re-running a dead end.

### A removed increment, left visible

```python
"""``mask_artifact`` -- **removed, it was a no-op**
    Intended to drop windows overlapping week 395 from *training*. Measured: the
    artifact falls in the **test** set of the origin-0.85 fold and in **no**
    training set under this protocol, so masking training changes nothing.
"""
```

The docstring for a feature that no longer exists. It is kept because the *idea*
is obvious enough that someone will have it again.

### `gaussian_nll`

```python
def gaussian_nll(mu, log_var, target):
    log_var = log_var.clamp(-10.0, 10.0)
    return torch.mean(0.5 * (log_var + (target - mu) ** 2 / torch.exp(log_var)))
```

The negative log-likelihood of a Gaussian, constant term dropped (it does not
affect the gradient).

**Why predict `log_var` rather than `var`?** Variance must be positive.
Exponentiating an unconstrained output guarantees it, with no clamping or
penalty.

**Why clamp?** *"an unconstrained variance lets the model buy loss by declaring
everything uncertain."* The model can reduce NLL by driving `log_var` up on hard
examples until the second term vanishes. The clamp bounds that escape.

**Why it matters:** *"Optimising this is what makes the Eq. (14) intervals
calibrated rather than decorative."* Intervals from a model trained on MSE are
arbitrary; intervals from a model trained on NLL are meaningful.

---

### `renewal.py` — the physics that did not work, kept because the negative is a result

The renewal equation `cases_t = R_t · Σ_s w_s · cases_{t−s}`, with `w` the
generation-interval distribution derived from the SEIR–SEI stage durations
(mean 3.30 weeks). Two entry points: `renewal_penalty` as a soft loss term and
`RenewalDecoder`, which reinterprets a backbone's output as `log R_t` and
reconstructs cases through the mechanism.

Both lost, and Chapter 9 §9.12 has the numbers. The module stays because the
negative is one of this project's real findings and an unreproducible negative is
not a finding at all. It is pinned by **12 tests**, including two that state the
design intent precisely:

- the penalty is exactly **zero** on a renewal-consistent forecast — a constraint
  must not charge for obeying it;
- its gradient points **upward** when `R > 1` — the measured failure mode was
  under-reaction, and a ceiling could only ever push down, which is why EXP-014
  found the old one inert.

One bound is worth reading as a lesson learned from EXP-014:

```python
LOG_R_BOUNDS = (-3.0, 2.5)   # R in [0.05, 12.2]
```

The observed p99 of back-solved `R` is 5.36, so the upper bound sits well above
anything real and cannot forbid a genuine outbreak. That is the opposite of the
0.70 ceiling that ruled out a fifth of the record.

### `physics.py` — three arms, one backbone

`base`, `penalty` and `decoder` share the backbone, the folds, the normalisation,
the loss scale and the protocol, so a difference between them is attributable to
the physics and nothing else. It also owns the conversion between raw counts and
normalised log1p space, because the decoder produces counts while the loss is taken
in `z` — and putting that conversion inside the module keeps the training loop
identical across arms.

`window_history` carries a comment worth generalising:

> Windows near the start of the record have less than `weeks` of history … They are
> **edge-padded** with the first available week rather than zero-padded: zeros would
> understate infectious pressure and make the physics term push the forecast down
> exactly where the record is thinnest.

**Padding is a modelling choice, not a formatting one.** Six windows in the whole
dataset are affected, and zero-padding them would have biased the term in a
direction nobody would have noticed.

---

## 8.4 `reproduction/` — running the authors' code

The hardest engineering in the project, because it must run **their** code
unmodified.

| file | purpose |
|---|---|
| `REPRODUCIBILITY_MATRIX.md` | Which Table I rows are reproducible; the three forced deviations |
| `_build/env_setup.py` | Shared cell sources: `PREAMBLE`, `SETUP`, `VERIFY_ENV`, `CLONE` |
| `_build/cells_weng.py` | Targets from `results.txt`, Table I values, code walkthrough |
| `_build/cells_sweep.py` | The improved-architecture sweep kernel |
| `_build/gen_kernels.py` | Generates notebooks from cell sources |
| `kaggle/setup_kaggle.py` | `--check`, `--init-metadata` |
| `verify_local.py` | Builds the pinned env at `C:/rp`, smoke-tests all five runners |

### Problem 1 — Python version

The authors pin `torch 2.1.2`. Kaggle's image is Python 3.12. **There is no cp312
wheel for torch 2.1.2.**

Solution: bootstrap a standalone Python 3.11 via `uv` inside the kernel. Do not
downgrade the pin; the pin is what makes it a reproduction.

### Problem 2 — verifying the data

An MD5 of the checked-out file was computed on Windows and compared against
Kaggle's Linux clone. **They differ** — git normalises line endings, so CRLF
locally becomes LF there.

Solution: verify **git blob SHAs** instead.

```python
BLOB_NPY = "f7cfa6ec31a4058584fe256a1d6de6800e72a5b1"
BLOB_ADJ = "f3a3cb7f43998850410b0a494f16f331c3830a84"
```

A git blob SHA is computed over the content git stores, so it is identical on
every platform. **When verifying files across platforms, hash what git hashes.**

### Problem 3 — Kaggle slugs

Kaggle requires the kernel `id` slug to match the `title` slug. One push returned
400; another **silently renamed the kernel**, which is worse.

Solution: derive everything from one source.

```python
slug = slugify(title)     # -> directory name, filename, and kernel id
```

### Problem 4 — Google Drive

The project folder syncs to Drive. Building a venv there produced numpy reporting
version `None`, and Windows `MAX_PATH` broke torch's headers under a deep temp
directory.

Solution: build at a short local path, `C:/rp`. The repo's `CLAUDE.md` carries a
standing warning that Drive's sync client can corrupt `.git` mid-write.

---

## 8.5 Generated notebooks

Reproduction and analysis notebooks are **generated**, not hand-edited.

```
analysis/_build/cells_eda.py       ──gen_analysis.py──>  E1_dataset_eda.ipynb
analysis/_build/cells_adaptive.py  ──────────────────>  E2_adaptive_graph.ipynb
```

**Never edit the `.ipynb`.** Edit the cell source and regenerate.

Why: `.ipynb` is JSON with embedded outputs and execution counts. It produces
unreadable diffs and merges catastrophically with six collaborators. A `.py` file
of cell sources diffs like code.

```python
CELLS = [
    md("""## F4 — Week 395 is a reporting artifact ..."""),
    code("""national = cases.sum(1) ..."""),
]
```

### The guard that had to be added

Cell sources are Python strings containing Python code, so `\n` inside them can
collapse into a real newline and produce an unterminated string literal in the
generated notebook. This happened repeatedly.

`tools/check_notebooks.py` now `compile()`s every code cell, so a
non-parsing generated notebook is an error rather than a surprise at run time.
It also checks valid JSON, execution order, and warns when `QUICK_TEST` is left
on — because quick-test numbers are intentionally degraded and must never reach
the report.

**When a mistake recurs, add a check rather than resolving to be careful.**

---

## 8.6 Recording results

From `CLAUDE.md`:

> Every run whose numbers might reach the report gets an entry in
> `docs/EXPERIMENT_LOG.md` (append-only, newest first) with the exact `CFG`,
> commit SHA, seeds, folds, and the unrounded table — plus a CSV in `results/`
> with one row per (fold, seed, horizon), **not** pre-aggregated.
> **No orphan numbers in either direction.**

Two rules doing real work:

**One row per (fold, seed, horizon), not pre-aggregated.** You cannot recover the
distribution from a mean. Keeping the raw rows means a paired t-test, a per-horizon
breakdown, or an artifact split can be computed later without re-running anything.

**No orphan numbers in either direction.** Every number in the report traces to a
logged run; every logged run's numbers are accounted for. The second direction
prevents quietly dropping an unflattering run.

Non-obvious modelling choices get an ADR in `docs/decisions/` — `0001` covers
residual-over-persistence, log1p, and train-only normalisation.

---

## 8.7 The habits worth stealing

1. **Name the coordinate space in the parameter** (`pred_counts`, not `pred`).
2. **Composite values as properties**, so they cannot drift.
3. **Validate shapes and emptiness at the boundary**, once.
4. **Implement critical formulas twice**, independently, and assert agreement.
5. **Fail loudly**; a plausible wrong number is worse than a crash.
6. **Document what you rejected**, and why.
7. **Generate notebooks from `.py`**; never hand-edit the JSON.
8. **When a mistake recurs, add a check**, not a resolution.
9. **State which comparisons your code supports** — and which it does not.
10. **Keep raw per-run rows**, not aggregates.

---

*Next: [Chapter 9 — The Reproduction Story](09_reproduction_story.md)*
