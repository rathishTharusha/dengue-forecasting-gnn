# Stage 3 — mechanistic constraints: what we tried, what happened, what to do next

**Date:** 2026-09-03 · **Data:** `results/phase3_stage3.csv` (144 runs) merged with
`results/phase3_runs.csv` · **Protocol:** 8 disjoint origins × 3 seeds, W=3 → H=3

> **Status: exploratory.** Rule D13 reserves the physics formulation for a human
> author. Everything here is evidence for that decision, not the decision. Every
> constant in `src/dengue_gnn/mechanistic.py` is marked `OWNER:` and must be
> confirmed or replaced before any of it reaches the paper.

---

## 1. Why we did not implement the SEIR–SEI residual

Two findings, both established before any code was written.

### 1.1 The cited reference cannot be used as written

`refs.bib` keys `seirsei2022` to *Analysis of Vector-host SEIR–SEI Dengue
Epidemiological Model* (Hasan, Hobiny & Alshehri, **Int. J. Anal. Appl. 20:57, 2022**).
Two problems:

- **`papers/09_SEIR_model.pdf` is not that paper.** It is a Portland State teaching
  notebook on the standard single-population SEIR, written for influenza in April
  2020. Zero occurrences of "dengue", "mosquito", "*Aedes*" or "bite". Rule D9 says
  to fix constants "from the SEIR–SEI reference"; that file cannot supply them.
- **The real paper's human block is malformed.** In its system,

  ```
  dE_h/dt = β₂·S_h·I_h − μ₁·E_h − α₁·E_h
  dI_h/dt = β₁·S_h·I_m − β₃·I_h − μ₁·I_h − α₁·I_h     ← no E_h term
  ```

  `E_h` receives inflow but never feeds `I_h`, so the exposed class is dynamically
  inert — the human side is SIR with a decorative E. The `β₂·S_h·I_h` term is direct
  human-to-human transmission, which dengue does not exhibit. And there is no
  intrinsic incubation, which is the three-week interval the paper's own
  introduction rests on. Its Table 1 is captioned "Values for baseline parameters"
  but contains only descriptions — no numbers.

  The vector block (`S_m → E_m → I_m`) is well-formed and usable.

### 1.2 The data cannot identify the latent compartments

Of seven compartments, **one** is observed — `I_h`, and only as under-ascertained
reported cases. A hard residual asks the network to infer **150 latent weekly
trajectories** (6 compartments × 25 districts) plus nine rate constants from 25
noisy series. Many latent configurations reproduce the same observed cases and
nothing in the loss prefers the right one. Serotype structure and cross-immunity,
which drive dengue's multi-year cycles, are absent from the model entirely.

**Conclusion:** we constrained quantities computable from predictions alone,
deriving their form from the **generation interval** — the mechanistic quantity
that survives when compartments are unobservable.

---

## 2. What we tried

All three act on *rates of change*, deliberately: the measured gap against
persistence is in peak timing, and a peak is where the growth rate crosses zero.

| Trial | Constraint | Rationale |
|---|---|---|
| **Growth band** | Hinge on `\|Δ log(1+y)\| > 0.70/week` | Generation interval ≈ 3 weeks; R=8 (already extreme) gives ln(8)/3 = 0.69. Penalises only implausible tails |
| **Growth smoothness** | Penalise week-to-week *change* in growth rate | Rₜ moves on the generation-interval timescale, so growth should not jump. The renewal-equation intuition without the kernel |
| **Curriculum λ** | Ramp constraint weight 0 → target over the first half of training | Krishnapriyan et al.: PINN failures are optimisation failures; a constraint at full strength from step one deforms the landscape |

The curriculum was applied both to the new term **and to the Stage-2b spatial term
that failed**, which turns it into a direct test of *why* Stage 2b failed.

### 2.1 The magnitude check changed the design before we ran anything

Per LL-019, magnitudes were measured against the data loss first:

```
L_data (MSE)                    0.2832       1.0     —
L_band (growth ceiling)         0.0000      0.00     INERT
L_smooth (growth 2nd diff)      0.0478      0.17     usable
                                λ for 5/20/50% of loss: [0.31, 1.48, 5.93]
```

Two consequences. **The usable λ range is 0.3–6, not the 0.01–1.0 grid Stage 2b
used** — off by two orders of magnitude, and we would have swept the wrong range.
And **the band term was already inert at initialisation**, because the
residual-over-persistence + log-space + clamp architecture prevents implausible
growth on its own.

---

## 3. Results

Pooled RMSE, mean ± SD over 8 folds × 3 seeds. Paired sign test against the
**unregularised adaptive model** (λ=0), which is the control that isolates the
constraint.

| Config | RMSE | SD | vs adaptive (λ=0) |
|---|---|---|---|
| Persistence floor | **55.12** | 41.03 | — |
| **mech_smooth λ=0.3** | **58.03** | 41.01 | **17/24 wins, p=0.064** |
| adaptive (λ=0, control) | 60.08 | 44.25 | — |
| spatial λ=0.1 **+ curriculum** | 60.11 | 43.34 | 11/21, p=1.000 |
| mech_both λ=1.5 | 62.36 | 49.08 | 10/24, p=0.541 |
| mech_smooth λ=1.5 | 66.47 | 61.93 | 9/24, p=0.307 |
| mech_smooth λ=1.5 + curriculum | 66.79 | 66.83 | 14/22, p=0.286 |
| mech_band λ=1.5 | 69.18 | 73.79 | 12/19, p=0.359 |
| spatial λ=0.1 (Stage 2b, no curriculum) | 70.27 | 60.95 | 11/24, p=0.839 |

### Peak week error — the criterion Stage 3 was supposed to be judged on

| Config | h=1 | h=2 | h=3 | mean |
|---|---|---|---|---|
| Persistence floor | **1.70** | **2.73** | **4.02** | **2.82** |
| **mech_smooth λ=0.3** | 2.18 | 3.35 | 4.77 | **3.43** |
| mech_smooth λ=1.5 + curr | 2.18 | 3.41 | 4.84 | 3.48 |
| mech_band λ=1.5 | 2.37 | 3.49 | 4.72 | 3.53 |
| adaptive (λ=0, control) | 2.41 | 3.54 | 4.84 | 3.60 |
| spatial λ=0.1 + curriculum | 2.43 | 3.58 | 4.94 | 3.65 |
| spatial λ=0.1 (Stage 2b) | 2.99 | 4.25 | 5.05 | 4.10 |

---

## 4. What we learned

### 4.1 A mechanistic constraint helped, for the first time

`mech_smooth` at λ=0.3 improves **both** metrics over its control: RMSE 60.08 → 58.03
and peak error 3.60 → 3.43. It is the only configuration tried in Phase 2 or Phase 3
that improves both. Its 17/24 paired wins at p=0.064 is the same signal strength as
the adaptive graph's own result, which is encouraging but **still not significant at
n=24**.

It closes roughly 40% of the RMSE gap between the adaptive model and persistence,
and about 22% of the peak-timing gap. It does not close either.

### 4.2 Weak constraints work; strong ones do not — regardless of curriculum

λ=0.3 (≈5% of total loss) helps. λ=1.5 (≈20%) hurts badly, at 66.47. And the
curriculum did **not** rescue λ=1.5 — 66.79 with it against 66.47 without. So the
problem at λ=1.5 is not the optimisation path; the constraint is simply too strong
and the data term loses.

**Implication:** for this family, tune λ *downward* from the measured 5% mark, not
upward. The next grid should be λ ∈ {0.05, 0.15, 0.3, 0.6}.

### 4.3 A curriculum removes the catastrophic failures, inconsistently

The largest single effect, though not the most consistent one. The spatial smoothness term at λ=0.1:

- without curriculum: **70.27** — catastrophic, and worse than persistence
- with curriculum: **60.11** — statistically indistinguishable from λ=0 (11/21, p=1.000)

A curriculum recovered **10.2 RMSE on the mean**. But the paired sign test is
only **14/24 (p=0.54)**: the mean improves because a minority of runs improve
enormously, not because most runs improve. The paired SD is 21.2 against a mean
effect of 10.2.

**Corrected claim.** The defensible statement is that a curriculum removes the
catastrophic failures rather than that it uniformly helps. That is still
consistent with Krishnapriyan et al. -- a deformed loss landscape produces
occasional very bad optima, and ramping the constraint avoids them -- but it is a
weaker and more accurate claim than "the failure was an optimisation failure"
full stop. Confirming it needs n>=35, which the 8-seed search provides.

The nuance matters for the paper. The curriculum makes the spatial term **harmless,
not helpful** — it returns to parity with no constraint at all. So the Phase-2
conclusion survives in substance ("spatial smoothness does not help") but its
*explanation* was wrong: we attributed the damage to the constraint being
antagonistic to the learned graph, when a large part of it was simply bad
optimisation.

### 4.4 The growth-band constraint is inert, and we can prove it

Five of 24 `mech_band` runs produced **bit-identical RMSE** to the λ=0 control
(e.g. fold 1 seed 0: 92.751984 in both). An exact tie means the penalty contributed
zero gradient for the entire run — the model never once forecast growth above the
ceiling.

This is the architecture doing the constraint's job already: residual-over-
persistence in log space, with `to_counts` clamped, cannot produce implausible
growth. **A constraint that is already enforced structurally is not a contribution.**
Where the band did activate (λ=1.5, other folds) it only hurt.

### 4.5 Nothing beats persistence, and the ordering is now unambiguous

Across Phase 3: persistence 55.12 < random forest 56.31 < XGBoost 57.41 <
**mech_smooth 58.03** < LSTM-no-graph 57.95 < adaptive GCN 60.08 < dense GCN 64.67.

The best mechanistic variant is competitive with the tabular baselines and still
below persistence. Combined with EXP-007 — where a no-graph LSTM beats both GNNs —
the honest summary is that **on 25 nodes the graph does not currently pay for
itself**, and a mechanistic constraint narrows but does not close that gap.

---

## 5. What to implement next

Ordered by expected value per hour.

1. **Sweep λ downward for `mech_smooth`: {0.05, 0.15, 0.3, 0.6}.** The optimum is at
   or below the smallest value tried. Cheap — one run of the existing harness.
2. **Combine `mech_smooth` λ≈0.3 with the curriculum.** Untested at the *useful* λ;
   we only tried curriculum at λ=1.5 where the constraint was already too strong.
3. **Drop the growth-band term.** Inert by construction (§4.4). Report it as a
   negative result with the bit-identical evidence, which is unusually clean.
4. **Re-frame the Stage-2b result in the paper** using §4.3. The finding changes from
   "spatial smoothness is antagonistic" to "spatial smoothness is neutral once
   optimised correctly, and the Phase-2 failure was largely an optimisation
   artefact". That is a better and more defensible claim, and it cites literature
   the team already holds.
5. **Report peak timing as the headline metric**, with RMSE secondary. It separates
   the configurations more cleanly and it is what an early-warning system is judged on.

## 6. What needs further research

- **Adaptive loss weighting (Wang, Yu & Perdikaris).** We tested a linear curriculum;
  their NTK method calibrates the weight *from the training dynamics*. Given §4.2 —
  where the right λ is narrow and the wrong one is catastrophic — an adaptive scheme
  is more promising than any fixed grid.
- **Whether the graph should be there at all.** EXP-007 says a no-graph LSTM beats
  both GNNs. Before adding more machinery to the graph model, it is worth testing the
  mechanistic constraint *on the LSTM* — if it helps there too, the contribution is
  the constraint, not the graph, and the paper's framing should follow the evidence.
- **A corrected vector-host model, if a latent-state route is ever attempted.** The
  cited paper's human block must be repaired (add `σ_h·E_h → I_h`, delete
  `β₂·S_h·I_h`) and a different reference cited. §1.2 still applies: identifiability,
  not correctness, is the binding constraint.
- **Spectral bias.** The team holds a paper showing low frequencies are learned
  first. Outbreak onset is the high-frequency component, which may explain why every
  model trails persistence on timing. Worth testing whether a frequency-aware
  objective moves peak error where λ-tuning has not.
- **Under-ascertainment.** Every constraint here treats reported cases as incidence.
  Reporting rates vary by district and season, so a fitted observation model may
  matter more than any refinement of the transmission constraint.

## 7. Provenance

| Artefact | Path |
|---|---|
| Constraint implementations | `src/dengue_gnn/mechanistic.py` (17 tests) |
| Trial configs | `scripts/run_phase3.py` → `stage3_configs()` |
| Raw results | `results/phase3_stage3.csv` (144 runs, per fold/seed/horizon) |
| Magnitude tool | `scripts/measure_loss_scale.py` |
| Reference analysis | this document §1, and the SEIR–SEI decision brief |

Reproduce with:

```bash
python scripts/measure_loss_scale.py
python scripts/run_phase3.py --only stage3 --out results/phase3_stage3.csv
```
