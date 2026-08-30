# Roadmap

Derived from `Group05_Proposal.pdf` §4–5 and `PROJECT_PROPOSAL_GUIDE.md`, revised
2026-08-30 after `docs/PHASE2_REVIEW.md`. Each stage has a **falsifiable success
criterion** — if a stage misses it, we report that honestly and fall back rather
than quietly moving the goalposts.

---

## Two vocabularies — do not mix them

This project counts in two directions, and the collision has already caused
confusion. From here:

| Term | Means | Numbered by |
|---|---|---|
| **Phase** | a course deliverable with a deadline and marks | the module handout |
| **Stage** | a technical increment in the model | this document |

"Phase 3" is the complete research paper. "Stage 3" is the mechanistic loss. They
are unrelated, and a Stage can span Phases.

### Course phases

| Phase | Deliverable | Due | Marks |
|---|---|---|---|
| 1 | Project proposal (2 pp) | 2 Aug 2026 | 25% |
| 2 | Short paper (4 pp) | 23 Aug 2026 | 25% |
| 3 | Complete paper + LaTeX zip + code | ⚠️ handout says 26 Apr 2026; cadence implies **≈ 20 Sep 2026** | 34% |
| — | Conference submission proof | 15 Nov 2026 | gates the ceiling |

Phase-3 marks are capped by publication outcome: accepted 95–100%, **rejected but
submitted 84%, never submitted 50%**. Submitting is the largest single lever in
the project. See [`PHASE3_PLAN.md`](PHASE3_PLAN.md) §1 and §2.3 — including why
the handout's date is not trustworthy.

---

## Done

### ✅ Stage 0 — Literature review & proposal
Three pillars (spatio-temporal GNN, physics-informed loss, GAN augmentation) plus
the gap: no published work combines all three for dengue. Hedged as "to the best of
our knowledge" — absence-of-evidence from a targeted search, not proof.

### ✅ Stage 1 — Baseline
GCN/GAT on PyTorch Geometric, `W=3 → H=3`, rolling-origin CV over 3 origins × 3
seeds. Residual-over-persistence (RMSE 66 → 45) and `log1p` targets are what made
it competitive; see
[`decisions/0001-baseline-training-refinements.md`](decisions/0001-baseline-training-refinements.md).

### ✅ Stage 2 — Adaptive graph (Contribution c) — *partially succeeded*
Graph WaveNet-style learned adjacency `A_adp = softmax(ReLU(E₁E₂ᵀ))`, blended with
geography under a learnable gate.

**Result:** improves RMSE 45.47 → 45.00 against a matched fixed-graph control on
the same propagation path. Real, but inside the noise band, and it does **not**
clear the persistence floor at 44.80. Success criterion *not* met.

### ✅ Stage 2b — Spatial regularisation — *failed, and reported as such*
Graph-Laplacian smoothness plus a non-negativity term. Worse at every weight
(45.92 / 45.79 / 67.72 against 45.00 unregularised). The non-negativity term is
provably inert. Not a mechanistic loss, and no longer described as one.

**Why it failed, which matters for Stage 3:** a smoothness penalty is minimised by
a *more uniform graph*, so it fights the adaptive component — the gate rose from
σ(g) 0.62 to 0.95 as λ increased. The two contributions are mildly antagonistic.

### Where things actually stand

| Model | RMSE | Peak err h1/h2/h3 (wks) |
|---|---|---|
| **Persistence floor** | **44.80 ± 21.49** | **0.99 / 1.90 / 3.44** |
| Dense GCN, fixed graph | 45.47 ± 16.91 | 1.98 / 4.59 / 5.47 |
| + adaptive graph | 45.00 ± 16.18 | 2.92 / 4.70 / 6.56 |
| + spatial reg. (λ=0.01 / 0.1 / 1.0) | 45.92 / 45.79 / 67.72 | — |

Nothing clears the floor; the best configuration wins 4 of 9 paired runs.
**The graph models call the outbreak peak two to three weeks later than
persistence.** That gap is large, was invisible until the peak-timing metric was
fixed, and matters more for early warning than RMSE does. It is the target for
Stage 3.

---

## Next

### ☐ Stage 3 — Mechanistic SEIR–SEI loss (Contribution a)
`L = L_data + λ·L_residual` against an SEIR–SEI host–vector compartmental update.
Constants fixed from the reference, never learned — ρ is unidentifiable against the
unobserved `E_h`.

**Success criterion — changed from the original roadmap:** improves **peak week
error** against Stage 2. RMSE is reported for completeness, but ±0.5 RMSE is
unresolvable at this sample size and chasing it wasted Stage 2. Timing has a 2–3
week gap and a mechanistic argument for why a dynamics constraint should close it.

**Why this need not repeat Stage 2b:** a dynamics residual constrains the
trajectory, not the spatial structure, so it does not fight the learned graph.

**Before any sweep:** measure the residual's magnitude against `L_data`. A term
orders of magnitude larger makes λ a switch, not a weight.

**Fallback:** EINN-style latent-dynamics transfer if the hard residual
over-constrains the unobserved compartments.

### ☐ Stage 4 — Comparative and computational analysis
Required by the handout's paper structure, and absent today. Port the notebook-02
GNN baselines (STGAT, A3TGCN, ASTGCN, DCRNN, AAGCN) and the notebook-01 classical
baselines onto our protocol, and add parameter counts, training time and inference
latency. Detail in [`PHASE3_PLAN.md`](PHASE3_PLAN.md) §5.1–5.2.

### ☐ Stage 5 — GAN augmentation (Contribution b) — optional
Conditional time-series GAN with WGAN-GP, judged on **downstream** accuracy against
a jittering / window-warping comparator, never on distributional similarity.
Highest risk, lowest marginal marks. If it is not started two weeks before the
Phase-3 deadline, it stays as future work.

### ☐ Stretch — PID-GAN-style coupling
Physics residual informs the GAN discriminator. Maximum novelty, maximum risk. The
project must not depend on it.

---

## Evaluation protocol

Frozen since Stage 1, with **one sanctioned change in Phase 3**: origins move from
3 to 6–8, and every ablation row is re-run against the new folds. After that it is
frozen again. Nothing else moves.

- Rolling-origin (expanding-window) CV, 3 seeds, averaged over matched runs.
- Window `W = 3` weeks → horizon `H = 3`; metrics reported **per horizon** and
  pooled. A table always states which aggregation it used — they differ by ~0.8
  RMSE here.
- Point: RMSE, MAE, SMAPE, masked MAPE(≥1).
- **Peak week error**, in weeks, over districts reaching `PEAK_FLOOR` cases.
  Sanity check: persistence at horizon *h* must score ≈ *h*.
- Probabilistic (Stage 3 on): CRPS, PICP, MPIW.
- Spatial: Moran's I on residuals — promised in the proposal, still absent.
- Normalisation from **training-fold statistics only**. No shuffling, ever.
- Rows recorded per `(fold, seed, horizon)`; aggregation happens at reporting time.
- Every comparison is against **persistence**, not only against the previous model.

---

## The ablation table

The deliverable that carries the novelty claim. Every row under one protocol.

| # | Configuration | RMSE | Peak err | Status |
|---|---|---|---|---|
| 0 | Persistence floor | 44.80 | 0.99/1.90/3.44 | ✅ |
| 1 | Dense GCN, fixed graph (control) | 45.47 | 1.98/4.59/5.47 | ✅ |
| 2 | + adaptive graph | 45.00 | 2.92/4.70/6.56 | ✅ |
| 3 | + spatial regularisation | 45.79 | — | ✅ negative |
| 4 | + SEIR–SEI residual | | | ☐ Stage 3 |
| 5 | + GAN augmentation | | | ☐ optional |
| — | Classical baselines (ARIMA, RF, XGBoost, LSTM) | | | ☐ Stage 4 |
| — | Published GNNs (STGAT, A3TGCN, ASTGCN, DCRNN, AAGCN) | | | ☐ Stage 4 |

Regenerate with `python scripts/make_tables.py`. Never type these values.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phase-3 deadline is not ≈20 Sep | High | Confirm with the lecturer before planning — `PHASE3_PLAN.md` §1 |
| No conference deadline clears 15 Nov | Medium | Choose the venue in week 1; a workshop or regional venue is appropriate for a strong-protocol negative result |
| Contributions never beat the persistence floor | **Realised** | Already the paper's framing. A well-run negative result with the ablation intact is a valid deliverable |
| Results produced by code that is not committed | **Realised in Phase 2** | Never `scratch/`; see `PHASE2_REVIEW.md` F1 and rule D3 |
| Effects smaller than fold-to-fold variance | **Realised** | Move to 6–8 origins; report paired tests, not means |
| SEIR–SEI residual over-constrains unobserved compartments | High | Measure magnitudes first; λ sweep; EINN-style latent transfer as fallback |
| GAN instability on a 459×25 dataset | High | WGAN-GP; fall back to jittering/window-warping and report it |
| `torch-geometric-temporal` fragility blocks the comparative analysis | Medium | Port architectures onto our dense path; if not feasible, state the operator difference explicitly |
| Notebook merge conflicts across six people | High | One owner per notebook at a time; stabilised code moves to `src/` |
| Google Drive corrupts `.git` | **Realised** | 81 stray `desktop.ini` files broke `fetch`. Work from a non-Drive clone — `docs/SETUP.md` |
| Several cited works are 2026 preprints | Medium | Re-verify each before submission |

---

## Ownership

Fill in names — `_TBD_` here on the day work starts is itself a risk.

| Area | Owner |
|---|---|
| Conference selection + submission logistics | _TBD_ |
| Stage 3 — SEIR–SEI residual | _TBD_ |
| Stage 4 — comparative analysis (GNN baselines) | _TBD_ |
| Stage 4 — computational analysis | _TBD_ |
| Protocol re-run at 6–8 origins | _TBD_ |
| Paper assembly, tables, contribution highlighting | _TBD_ |
| Stage 5 — GAN augmentation (optional) | _TBD_ |

Author order on the paper is **by contribution** and is prescribed by the handout,
not by preference. Agree it at the start of the phase, not the night before.
