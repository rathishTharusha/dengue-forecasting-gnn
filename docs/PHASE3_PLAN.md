# Phase 3 — Complete Research Paper Execution Plan

**Weight:** 34% of project marks — larger than Phases 1 and 2 combined.
**Deliverable:** complete conference-style research paper + LaTeX zipfile + code,
in two versions (clean + colour-highlighted per-member contributions).
**Deadline:** ⚠️ see §1 — the handout's date is not trustworthy.
**Conference submission proof:** 11:59 PM 15 November 2026.

Written 2026-08-30, after the Phase-2 review (`docs/PHASE2_REVIEW.md`). Terminology
in this repo now follows `docs/ROADMAP.md`: **Phase** = a course deliverable,
**Stage** = a technical increment. They used to collide and it caused confusion.

---

## 1. Confirm the deadline before planning anything

`CS3631 - Project - 2026.pdf` §12 says:

> Phase 3: Complete Research Paper, LaTeX Zipfile and Code — Week 12 —
> (11.59 PM **April 26, 2026**) — 34% of Project Marks

That date cannot be right. Phase 1 was Week 5 / 2 Aug 2026 and Phase 2 was
Week 8 / 23 Aug 2026 — exactly three weeks apart for three course weeks. Week 12
on the same cadence is **≈ 20 September 2026**, which is about three weeks from
today. April 2026 is almost certainly stale text from a previous offering.

**Action, today:** ask the lecturer or a TA to confirm, and write the answer here.
Everything below assumes ~20 September. If it is later, the optional items in §5
come back into scope; if it is sooner, cut from the bottom of §4.

---

## 2. What Phase 3 actually asks for

### 2.1 Required paper structure (handout §9)

- Abstract · Introduction · Related Work
- **Proposed Framework**, including **Implementation Details**
- **Experimental Setup**: Dataset (overview only) · Baseline Models · Experiments
  - Ablation Studies
  - **Comparative Analysis with existing methods**
  - Hyper-parameter Tuning experiments
  - **Computational Analysis**
  - Other experiments demonstrating the advantages of the proposed approach
- Discussion · Conclusion · References

### 2.2 Where the short paper already stands

| Required | Status after Phase 2 |
|---|---|
| Abstract, Intro, Related Work, Discussion, Conclusion | ✅ written |
| Proposed Framework | ✅ written |
| Implementation Details | ⚠️ absent as a section; the content exists in `src/` |
| Ablation Studies | ✅ six rows, one control, per-`(fold, seed, horizon)` |
| Hyper-parameter Tuning | ⚠️ only the λ sweep |
| **Comparative Analysis with existing methods** | ❌ **absent** |
| **Computational Analysis** | ❌ **absent** |
| Other experiments | ❌ absent |

Two genuine pieces of work, both in §5.

### 2.3 Marks, and the one lever that dominates

| Outcome | Maximum |
|---|---|
| Accepted, high-rank / high h-index conference | 100% |
| Accepted, any recognised conference | 95% |
| **Rejected, but submitted** | **84%** |
| **Never submitted** | **50%** |

Submitting is worth up to **34 percentage points of ceiling** and has nothing to
do with model quality. No modelling improvement available to us in three weeks
comes close. Treat venue selection and submission as the primary deliverable and
the science as what fills it.

### 2.4 Authorship (prescribed — not our choice)

Group members first, **ordered by contribution to the paper**, then the assigned
TAs, then Prof. Dulani and Dr. Sandareka last. No titles anywhere in the author
list. `paper/main.tex` currently lists only the six of us and must be updated.

Agree the contribution order at the §4 Day-0 meeting, not the night before.

---

## 3. Positioning — what Phase 2 established

Read `docs/PHASE2_REVIEW.md` before writing a sentence. The short version:

**What holds.** The gated adaptive graph improves on a matched fixed-graph control
(45.47 → 45.00 RMSE). It is the one contribution with a defensible effect, and the
control that isolates it now exists.

**What does not.** The spatial regulariser is worse at every weight tested
(45.92 / 45.79 / 67.72). Its non-negativity term is provably inert. Neither is a
mechanistic constraint, so the paper no longer claims "physics-informed".

**What nothing clears.** Persistence, at 44.80, beats every configuration. The best
wins 4 of 9 paired runs — indistinguishable from chance.

**Where the real gap is.** Peak timing. Persistence places the outbreak peak at
0.99 / 1.90 / 3.44 weeks error across horizons; the adaptive model at
2.92 / 4.70 / 6.56. Both graph models call the peak **two to three weeks late**.
For an early-warning system that is the number that matters, it is a large and
newly-visible gap, and it is where a mechanistic loss should help.

**The framing for Phase 3.** Not "our model wins". Rather: *a learned graph
recovers most of the deficit a spatio-temporal GNN carries against a naive
baseline; a spatial smoothness prior actively hurts, and we explain why; and the
operationally important failure is timing, not magnitude.* That is a defensible
paper, and the handout explicitly rewards research thinking over model application.

---

## 4. Priorities, ordered by marks per hour

Work top-down. If time runs out, it runs out at the bottom.

| # | Item | Why here | Est. |
|---|---|---|---|
| 1 | Confirm the deadline (§1) | Gates everything | 5 min |
| 2 | Choose a conference (§6) | Blocks format, page limit, and the 34-point lever | 1 day |
| 3 | Computational Analysis (§5.1) | Required, cheap, and a listed novelty route | 1 day |
| 4 | Comparative Analysis (§5.2) | Required, and the largest missing section | 3–4 days |
| 5 | More rolling origins (§5.3) | Buys the statistical power every other result needs | 1 day compute |
| 6 | Implementation Details + hyper-parameter section (§5.4) | Required, mostly writing | 1 day |
| 7 | SEIR–SEI mechanistic loss, targeting peak timing (§5.5) | The proposal's unkept promise; real novelty | 4–5 days |
| 8 | GAN augmentation (§5.6) | Highest risk, lowest marginal marks | optional |

Items 1–6 are the difference between a compliant paper and a non-compliant one.
Item 7 is the difference between a compliant paper and a good one. Item 8 is a
stretch goal and the plan must not depend on it (this is the same call the Phase-2
plan made, and it was right).

---

## 5. Technical specification

### 5.1 Computational Analysis

Required by §9 and currently absent. Cheap because the harness already records
what is needed.

Report, per configuration: trainable parameter count; wall-clock training time per
fold; inference latency for one 25-node forward pass; peak memory. Compare against
the classical baselines, which are far cheaper, and state the honest conclusion —
a GNN that matches persistence costs orders of magnitude more compute than
persistence does.

That is not an embarrassment; the handout lists **computational efficiency** as a
valid novelty route, and an efficiency-aware framing is a genuine contribution on a
25-node graph where most published work assumes hundreds.

Extend `scripts/run_phase2.py` to log timing and parameter counts per run, into the
same per-`(fold, seed, horizon)` CSV. Do not measure this by hand.

### 5.2 Comparative Analysis with existing methods

The largest gap. The Phase-2 paper *declines* to compare against Weng et al., on
the correct grounds that their static 70/30 split is not commensurable with
rolling-origin. That reasoning stands — so **re-run their models under our
protocol** rather than citing their numbers.

The models already exist: `notebooks/02_baselines_gnn.ipynb` reproduces STGAT,
A3TGCN, ASTGCN, DCRNN and AAGCN. The work is porting them into
`src/dengue_gnn/models.py` and running them through `experiment.py` so every row
sits under the frozen protocol with per-`(fold, seed, horizon)` rows.

Also port the classical baselines from `notebooks/01_baselines_classical.ipynb`
(ARIMA/SARIMA, Random Forest, XGBoost, LSTM) — the proposal promised them and they
are cheap.

**Constraint:** `torch-geometric-temporal` is what notebook 02 uses, and ADR-0002
rejected it for the main pipeline. Either port the architectures onto our dense
propagation path (preferred — keeps the comparison controlled) or run them
separately and state clearly that the propagation operator differs. Do not blur
the two; that is finding F6 all over again.

### 5.3 More rolling origins

Fold-to-fold SD is ±16 to ±21 against effects of ±0.5. Three origins cannot resolve
that, and every Phase-3 contribution will land in the same noise unless this
changes.

Move from 3 origins to **6–8**, keeping window, horizon, seeds and normalisation
identical. This is the one protocol change sanctioned for Phase 3 (D1 said "no
until Phase 3"), and it requires **re-running every row of the ablation table** —
budget the compute and do it once, early, before anything else is measured against
the new folds.

Report the paired test over the larger matched set. If the adaptive graph's
advantage survives at 6–8 origins, that is a real result; if it does not, that is
also a real result and §3's framing already accommodates it.

### 5.4 Implementation Details and hyper-parameter experiments

Mostly writing. A short subsection under Proposed Framework covering the dense
propagation path, row normalisation and why it differs from `GCNConv`, the gate
initialisation, the residual-over-persistence and `log1p` target handling, and the
`log1p`-space application of the regulariser with the magnitude argument from F5.

For hyper-parameter tuning, the λ sweep exists. Add at least one more axis — hidden
width or embedding dimension `d` — swept under the same protocol. Two axes is the
minimum that reads as a tuning study rather than a single sweep.

### 5.5 SEIR–SEI mechanistic loss — Stage 3

The proposal's unkept promise, and the strongest available novelty.

`L = L_data + λ · L_residual`, where `L_residual` penalises deviation from an
SEIR–SEI host–vector compartmental update. Fix the epidemiological constants from
the reference (D9) — do not learn ρ; it is unidentifiable against the unobserved
`E_h` and will wreck training.

**Success criterion — and this is the important change from the old roadmap:**
target **peak week error**, not RMSE. Report RMSE for completeness, but the claim
is that a mechanistic constraint improves outbreak *timing*, where there is a 2–3
week gap to close and a mechanistic argument for why it should help. RMSE
differences of ±0.5 are unresolvable and chasing them wasted Phase 2.

**Why this should not repeat the smoothness failure:** the smoothness penalty is
minimised by a more uniform graph, so it fought the adaptive component and pulled
the gate back toward geography (σ(g) 0.62 → 0.95 as λ rose). A dynamics residual
constrains the *trajectory*, not the spatial structure, so it has no such conflict.
State this argument in the paper — it is the reason a reader should expect a
different outcome.

**Before running any sweep:** measure the residual's magnitude against `L_data`,
as `_regularisation_target` documents. A term four orders of magnitude larger makes
λ a switch rather than a weight, and the sweep measures nothing (F5, LL-019).

**Fallback:** if the hard residual over-constrains — likely, since `E_h` and
infected-mosquito compartments are unobserved — fall back to EINN-style latent
dynamics transfer rather than abandoning the contribution. Decide by λ sweep, and
report whichever outcome occurs (D8).

### 5.6 GAN augmentation — Stage 4, optional

Only if §5.1–5.5 land with a week to spare. Conditional time-series GAN (TimeGAN or
RCGAN-style) with WGAN-GP, judged on **downstream** RMSE/peak-timing against a
cheap jittering / window-warping comparator — not on distributional similarity.

If it is not started by two weeks before the deadline, it stays as future work. A
half-trained GAN in the paper is worse than an honest "not attempted".

---

## 6. Choosing a conference

Blocking, because it determines page limit and template — the two things the
handout leaves open.

Selection criteria, in order: (1) a recognised venue with a CFP deadline that
clears 15 November 2026; (2) scope covering applied ML for public health, epidemic
forecasting, or spatio-temporal modelling; (3) a page limit our material fits.

Realistic families to look at: regional IEEE/ACM conferences with health-informatics
tracks, epidemiological modelling workshops attached to larger ML conferences, and
Sri Lankan / South Asian CS conferences where the dataset's regional specificity is
an advantage rather than a curiosity.

Note that a negative-result paper with a strong protocol is a better fit for a
workshop or a "lessons learned" track than for a main conference. That is not a
downgrade — it is a venue whose reviewers reward exactly what we have.

Write the chosen venue, its deadline, its page limit and its template into this
document once decided.

---

## 7. Standing rules for Phase 3

Inherited from `.antigravity/project/rules.md`, which is the authority. The four
that Phase 2 actually broke:

1. **Code that produces a number is committed before the number is quoted.** Never
   `scratch/`. Every Phase-2 result was unreproducible because of this (F1).
2. **Rows per `(fold, seed, horizon)`, aggregated only at reporting time**, and
   every table states which aggregation it used (F9).
3. **One variable at a time, each against a control sharing the rest of the
   pipeline** (F6).
4. **Tables are generated by `scripts/make_tables.py`**, never typed (F3).

Plus, new for this phase:

5. **The protocol changes exactly once**, in §5.3, and everything is re-run against
   it. After that it is frozen again.
6. **Peak timing is a first-class metric**, reported alongside RMSE in every table.
7. **Verify a metric against a case with a known answer before trusting it** —
   persistence at horizon *h* must score ≈ *h* weeks of peak lag (LL-018).

---

## 8. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Deadline is not 20 September | High | §1 — confirm today, before planning |
| No suitable conference deadline clears 15 Nov | Medium | Start §6 in week 1; a workshop or regional venue is acceptable and appropriate |
| Comparative analysis needs `torch-geometric-temporal`, which is fragile | Medium | Port architectures onto our dense path instead; if not feasible, run separately and state the operator difference explicitly |
| Re-running everything at 6–8 origins exceeds available compute | Medium | 45 runs took 30 min on CPU; 8 origins ≈ 2.5× that. Route sweeps to Kaggle (D13) |
| SEIR–SEI residual over-constrains unobserved compartments | High | λ sweep with magnitudes measured first; EINN-style latent transfer as fallback |
| Nothing beats persistence, again | Medium | This is already the paper's framing (§3). It is a finding, and D8 says negative results ship |
| Six people, one paper, three weeks | High | Ownership table in `docs/ROADMAP.md`; one owner per section; contribution order agreed at Day 0 |

---

## 9. What "done" looks like

- [ ] Deadline confirmed and written into §1
- [ ] Conference chosen; venue, page limit and template recorded in §6
- [ ] Protocol moved to 6–8 origins and **every** ablation row re-run against it
- [ ] Computational Analysis section, generated from logged measurements
- [ ] Comparative Analysis against GNN and classical baselines under our protocol
- [ ] Implementation Details and a two-axis hyper-parameter study
- [ ] SEIR–SEI residual attempted, with peak-timing as the criterion, outcome
      reported either way
- [ ] Author list in prescribed order, TAs and supervisors appended
- [ ] Both PDFs build from one source (`cd paper && ./build.sh both`)
- [ ] LaTeX zipfile produced (`python scripts/make_overleaf.py`)
- [ ] Code committed, CI green, every number traceable to a committed CSV
- [ ] Submitted to the chosen venue, proof retained for 15 November
