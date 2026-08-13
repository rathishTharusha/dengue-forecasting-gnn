# Roadmap

Derived from `Group05_Proposal.pdf` §4–5 and `PROJECT_PROPOSAL_GUIDE.md`. Each stage has a
**falsifiable success criterion** — if a stage misses it, we report that honestly and fall back
rather than quietly moving the goalposts.

---

## Done

### ✅ Phase 0 — Literature review & proposal
Three pillars (spatio-temporal GNN, physics-informed loss, GAN augmentation) plus the gap:
no published work combines all three for dengue. Hedged as "to the best of our knowledge" —
it is absence-of-evidence from a targeted search, not proof.

### ✅ Phase 1 — Baseline
GCN/GAT on PyTorch Geometric, 3-week window → 3-week horizon, rolling-origin CV over 3
chronological origins × 3 seeds. Result: **RMSE ≈ 45.3 (GCN) / 45.5 (GAT) vs persistence 44.8**
— matches the floor, does not beat it.

Two refinements were what made it competitive at all (ablation in the notebook §9):
residual-over-persistence (RMSE 66 → 45) and log1p target space. See
[`decisions/0001-baseline-training-refinements.md`](decisions/0001-baseline-training-refinements.md).

---

## Next

### ☐ Phase 2 — Adaptive graph (Contribution c) — *do this first, it's the cheapest*
Replace the fixed distance-based adjacency with learned node embeddings,
`A_adaptive = softmax(ReLU(E₁E₂ᵀ))` (Graph WaveNet-style), optionally with a gravity-mobility
prior. Lowest-risk of the three, defensible as a novelty increment on its own, and it de-risks
the pipeline before the harder pieces land.

**Success criterion:** RMSE below the persistence floor (< 44.8) on the identical rolling-origin
protocol, or a clear win on peak-week timing.

### ☐ Phase 3 — Physics-informed loss (Contribution a)
`L = L_data + λ_p·L_phys + λ_c·L_cons + λ_s·L_smooth`, grounded in the SEIR–SEI host–vector
compartmental model. `L_cons` = population conservation; `L_smooth` = neighbour smoothness
across the district graph.

**Success criterion:** improves 3-step-ahead RMSE *and* peak-season (out-of-distribution)
accuracy vs. Phase 2.
**Fallback if it over-constrains:** dengue's exposed-human and infected-mosquito compartments
are unobserved, so a hard ODE residual may over-penalize. Fall back to EINN-style latent
dynamics transfer instead of a hard residual. Decide by λ sweep, not by vibes.

### ☐ Phase 4 — GAN augmentation (Contribution b)
Conditional time-series GAN (TimeGAN or RCGAN-style, conditioned on meteorological covariates
and outbreak labels) synthesizing extra `(window → horizon)` sequences. Use WGAN-GP /
gradient penalty — GANs mode-collapse readily on ~460-timestep series.

**Success criterion:** measurable downstream RMSE/CRPS improvement on held-out rolling origins
— *not* distributional similarity to real data.
**Mandatory comparator:** a cheap jittering / window-warping augmentation baseline. The GAN has
to beat it to justify its complexity.

### ☐ Stretch — PID-GAN-style coupling
Physics residual feeds the GAN discriminator directly, so generated series are
epidemiologically consistent. Maximum novelty, maximum risk. Stretch goal only — the project
must not depend on it.

### ☐ Phase 5 — Full ablation & write-up
**This table is the deliverable that proves the novelty claim.** Every row under the identical
protocol:

| # | Configuration | RMSE | MAE | SMAPE | CRPS | PICP/MPIW | Moran's I |
|---|---|---|---|---|---|---|---|
| 0 | Persistence | 44.8 | 15.7 | | — | — | |
| 1 | GCN baseline (residual + log) | 45.3 | 15.9 | | — | — | |
| 2 | + adaptive graph | | | | | | |
| 3 | + physics loss | | | | | | |
| 4 | + GAN augmentation | | | | | | |
| 5 | all three | | | | | | |

Plus the external comparators from the proposal: ARIMA/SARIMA, Random Forest, XGBoost, LSTM,
GRU, ConvLSTM, DCRNN, STGCN, EpiGNN, ColaGNN.

---

## Evaluation protocol (frozen)

Changing any of this invalidates cross-row comparison. If it must change, re-run **every** row.

- Rolling-origin (expanding-window) CV, 3 chronological origins, averaged over 3 seeds.
- Window `W = 3` weeks → horizon `H = 3` weeks; metrics reported **per horizon** and overall.
- Point: RMSE, MAE, SMAPE, masked MAPE(≥1).
- Probabilistic (from Phase 3 on): CRPS, PICP, MPIW.
- Spatial: Moran's I on residuals — confirms the graph captures genuine spatial structure
  rather than spurious spillover.
- Normalization from **training-fold statistics only**. No shuffling, ever.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| GAN instability / mode collapse on a 459×25 dataset | High | WGAN-GP; fall back to jittering + window-warping and report that honestly |
| Physics loss over-constrains unobserved compartments | Medium | λ sweep; fall back to EINN-style latent-dynamics transfer |
| Contributions fail to beat the persistence floor | Medium | The floor-matching baseline is already the honest framing; report negative results with the ablation intact — a well-run negative result is a valid deliverable |
| Several cited works are 2026-dated preprints | Medium | Re-verify each before final submission; a preprint may be revised or withdrawn |
| Notebook merge conflicts across 6 people | High | One owner per notebook at a time (see CONTRIBUTING.md); stabilized code moves to `src/` |
| Compute exceeds free Colab/Kaggle tier | Low | Graph is 25 nodes / 141 edges; only the GAN is heavier. Keep `QUICK_TEST` for iteration |
| Google Drive corrupts `.git` | Medium | Clone to a local non-Drive path (docs/SETUP.md) |

---

## Suggested ownership split

From `PROJECT_PROPOSAL_GUIDE.md` §4, adapted for six people — fill in names as the team agrees:

| Area | Owner |
|---|---|
| Data pipeline + notebooks 00/01 (classical baselines) | _TBD_ |
| Notebook 02 (GNN reproduction, heavier compute) | _TBD_ |
| Phase 2 — adaptive graph | _TBD_ |
| Phase 3 — physics loss | _TBD_ |
| Phase 4 — GAN augmentation | _TBD_ |
| Evaluation harness, ablation table, write-up | _TBD_ |

> Don't let baseline reproduction become the whole project. It's one section of the report,
> not the deliverable.
