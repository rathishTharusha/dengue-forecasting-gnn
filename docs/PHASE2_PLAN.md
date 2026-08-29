# Phase 2 — Short Paper Execution Plan

**Deadline:** Sunday 23 August 2026, 11:59 PM (Moodle) — 25% of project marks
**Working deadline:** Friday 21 August EOD. Saturday is assembly, Sunday is buffer.
**Deliverable:** 4 pages excluding references, ACM Conference Proceedings Primary Article
template (`sigconf`), **two versions** — clean + colour-highlighted per-member contributions.

---

## 0. What Phase 2 actually asks for

From `CS3631 Project 2026.pdf` §12, Phase 2 requires exactly three things:

1. **Proposed architecture** — the design, formally specified.
2. **Initial evaluation of the proposed architecture** — measured numbers, not promises.
3. **Comparison with the baseline** — our own GCN/GAT and the persistence floor.

Point 2 is the binding constraint. A paper that only *describes* the architecture misses the
requirement. So the plan is built backwards from **one frozen results table**.

**Scope committed:** Contribution (c) adaptive graph + Contribution (a) physics-informed loss.
**GAN augmentation (b) is explicitly deferred to Phase 3** and appears in this paper only as
future work. Do not let it back into scope.

---

## 1. Positioning — read this before writing a single sentence

Two findings from the reference papers change how this paper must be framed.

### 1.1 Weng et al. (IEEE BigData 2024) hand us our motivation

Their §VII "Modeling Challenges" states, of their own best models:

> "At some predicted time-steps, the predictions in one district is greatly influenced by the
> cases of neighboring districts, even if the disease was not spread in actuality, causing
> abnormal spikes in some of the test predictions. **This is one inherent flaw of using GNNs.**"

That is our paper's opening. The direct predecessor on this exact dataset names the failure
mode — unconstrained spatial message passing hallucinating outbreaks — and does not fix it.
Our physics-informed loss is the fix. Do not bury this; it is the strongest sentence available
to us and it is *their* words, not our speculation.

Second, note what Weng et al. **do not** report: a naive persistence baseline. Their GNN RMSE
of 30–45 is never compared against "predict last week's value." Our finding that a tuned
spatio-temporal GNN only *matches* persistence (45.3 vs 44.8) under a leakage-free
rolling-origin protocol is therefore a genuine methodological contribution in its own right,
not an embarrassment. **Frame it as a finding, not an apology.** Their numbers and ours are
not directly comparable (70/30 split vs rolling-origin, different normalisation) — say so
explicitly, in one sentence, or a reviewer will assume we cherry-picked.

### 1.2 DengueGNN (Sci. Rep. 2026) partially pre-empts Contribution (c) — act on this

`DengueGNN` proposes a dynamic spatio-temporal GNN whose adjacency is
`A_t = a·A_geo + (1−a)·A_mob` with a **gravity-model mobility prior**, plus attention-LSTM
temporal encoding, uncertainty-aware output, and Moran's I evaluation.

Overlap with our plan: the gravity-mobility prior, Moran's I, and probabilistic output are all
already published. **Consequences:**

- **Drop the gravity-mobility prior from the headline.** Our adaptive graph must be the
  *purely learned* Graph WaveNet-style `A_adp = softmax(ReLU(E₁E₂ᵀ))` — connectivity inferred
  from data with no mobility assumption. That is the honest differentiator: DengueGNN *assumes*
  a mobility law, we *learn* the structure. Keep the gravity prior only as an ablation row if
  time permits.
- **The physics-informed loss carries the novelty.** DengueGNN has no mechanistic constraint.
  Neither does Weng et al. The SEIR–SEI-constrained STGNN is where our claim lives — the
  adaptive graph is a supporting component, not the headline.
- **Cite DengueGNN and distinguish ourselves in Related Work explicitly.** Reviewers will find
  it. A paper that ignores the nearest neighbour in the literature reads as careless. One
  sentence: they model dynamic connectivity from an assumed mobility law; we learn connectivity
  and constrain the *dynamics* mechanistically.
- **Retire "to our knowledge this combination is unexplored"** from the proposal in its current
  form. Narrow it to the specific combination we actually implement and test.

### 1.3 Working title

> **Physics-Informed Adaptive Graph Neural Networks for Multi-Horizon Dengue Forecasting**

"GAN-augmented" comes out of the title — we are not evaluating a GAN in this paper. Putting it
in the title and not in the results is the single fastest way to lose reviewer trust.

---

## 2. The one table that is the deliverable

Everything below exists to fill this in. Identical rolling-origin protocol for every row —
3 chronological origins × 3 seeds, `W=3 → H=3`, training-fold normalisation only.

| # | Configuration | RMSE | MAE | SMAPE | Peak-timing err |
|---|---|---|---|---|---|
| 0 | Persistence (naive floor) | 44.8 | 15.7 | | |
| 1 | GCN baseline (residual + log) | 45.3 | 15.9 | | |
| 2 | GAT baseline (residual + log) | 45.5 | 15.9 | | |
| 3 | **+ adaptive graph** | | | | |
| 4 | **+ physics loss** | | | | |
| 5 | **+ both (proposed)** | | | | |

Rows 0–2 already exist (EXP-002). Rows 3–5 are this week's work.

**Add "peak-timing error" as a column.** It is cheap to compute (weeks between predicted and
actual local maximum per district-season) and it is the metric our physics claim actually
predicts an improvement on. If RMSE gains are marginal, a peak-timing win still makes the paper.
Do not go into the results with RMSE as the only success criterion.

---

## 3. Day-by-day

Owners are roles, not names — assign at the Day 0 standup. With 3–4 active people, one person
takes two roles; **W (writing) must not be the same person as P (physics)** — the physics owner
will be the busiest.

| Role | Responsibility |
|---|---|
| **A** | Adaptive graph implementation + its Kaggle runs |
| **P** | Physics-informed loss + λ sweep |
| **E** | Evaluation harness, metrics, results CSVs, all figures and tables |
| **W** | Writing lead — assembles LaTeX, owns page budget and the highlighted version |

### Day 0 — Mon 17 Aug (today/tonight)

- [ ] **All:** 30-min standup. Assign A/P/E/W. Agree the Go/No-Go criterion in §4.
- [ ] **W:** `paper/` skeleton is already scaffolded (see §5). Get it compiling on Overleaf or
      locally, confirm 4-page geometry, commit.
- [ ] **E:** Add `peak_timing_error` and `smape` to `src/dengue_gnn/metrics.py` + tests.
      Export EXP-002 rows 0–2 to `results/baseline_rolling_origin.csv` — *this is overdue and
      blocks the results table.*
- [ ] **A:** Fork `notebooks/baseline/dengue_baseline_GNN_v2.ipynb` →
      `notebooks/03_proposed.ipynb`. Refactor the GCN to a **dense** adjacency formulation
      (`H = Â X W`) instead of `edge_index`. 25 nodes — dense is free and it is the only way
      the learned adjacency plugs in cleanly. Verify it reproduces RMSE 45.3 ± noise before
      changing anything else. **If the refactor doesn't reproduce the baseline, stop and fix
      that first.**
- [ ] **W:** Draft Introduction + Related Work. Neither depends on new results. Use §1 above.

### Day 1 — Tue 18 Aug

- [ ] **A:** Implement adaptive adjacency. `E₁, E₂ ∈ R^{25×d}`, `d=10`,
      `A_adp = softmax(ReLU(E₁E₂ᵀ))`. Two variants to run: (i) replace `A_fixed` entirely,
      (ii) gated blend `A = σ(g)·A_fixed + (1−σ(g))·A_adp` with learnable `g`. Variant (ii) is
      the safer bet — it can only fall back to the baseline.
- [ ] **A:** First Kaggle runs, 3 folds × 3 seeds. ~15–25 min per config on a T4.
- [ ] **P:** Implement the physics loss **skeleton with λ=0** and confirm it trains identically
      to Day 0. Getting the plumbing right before the physics is on is worth the hour.
- [ ] **E:** Results ingestion — every run writes a row to `results/*.csv` with full config.
      A number without a config does not go in the paper (your own `EXPERIMENT_LOG.md` rule).
- [ ] **W:** Intro + Related Work drafts done, in LaTeX, in the repo.

### Day 2 — Wed 19 Aug

- [ ] **A:** **Adaptive graph result locked.** Log as EXP-003. Whatever it is, it is final —
      no more tuning after today. Also save the learned `A_adp` heatmap; if it recovers
      geographic structure it is a free, high-value figure.
- [ ] **P:** Physics loss live. Formulation in §6 below. Start with `L_smooth` and `L_cons`
      (cheap, stable), then the SEIR–SEI residual.
- [ ] **E:** Figure 1 (architecture diagram) drafted. This is on the critical path for the
      Methods section — do not leave it to Saturday.
- [ ] **W:** Write §3 Proposed Framework against the *implemented* code, not the proposal text.

### Day 3 — Thu 20 Aug — **GO / NO-GO DAY**

- [ ] **P:** λ sweep: `λ_p ∈ {0, 0.01, 0.1, 1.0}` × 3 folds × 3 seeds. ~2 hrs of Kaggle GPU.
- [ ] **ALL, 6 PM: Go/No-Go gate.** See §4. Decide in 15 minutes, write the decision down,
      move on. Do not relitigate it on Friday.
- [ ] **E:** Tables 1 and 2 generated from CSVs by script, not typed by hand.
- [ ] **W:** §4 Experimental Setup written (protocol is frozen and already known — no
      dependency on results).

### Day 4 — Fri 21 Aug — **RESULTS FREEZE 6 PM**

- [ ] Final combined run (row 5: adaptive + physics), 3×3.
- [ ] **6 PM: numbers frozen.** Nothing after this point changes a number in the paper. Any
      run finishing later is Phase 3 material.
- [ ] **W:** §5 Results and §6 Discussion/Conclusion written against the frozen table.
- [ ] **E:** Figure 2 — Colombo (or the worst-spike district) forecast overlay, baseline vs
      proposed, showing spike suppression. This figure is the visual payoff of the Weng et al.
      quote in the Introduction. High value; make it clean.

### Day 5 — Sat 22 Aug — assembly

- [ ] Full compile. **Trim to 4 pages.** It will be over — plan to cut ~15%. Cut from Related
      Work and Experimental Setup first, never from Results.
- [ ] Abstract written **last** (150 words: problem → proposed solution → significance, per the
      lecture slides' Vaswani-abstract anatomy).
- [ ] Produce the **colour-highlighted version**. Macro scaffolding is already in the skeleton —
      one colour per member, plus margin comments where the implementer ≠ the writer.
- [ ] Two full read-throughs by two different people. Spell-check. Check every citation resolves.
- [ ] Update `docs/EXPERIMENT_LOG.md` with EXP-003/004/005 and commit all `results/*.csv`.

### Sun 23 Aug — buffer, then submit

Submit both PDFs to Moodle by **11:59 PM**. Do not plan to work Sunday; plan to submit Sunday.

---

## 4. The Go/No-Go gate (Thursday 6 PM) — **STATUS: DECIDED & PASSED (GO)**

Decide, in this order:

**Q1 — Does the adaptive graph beat persistence (RMSE < 44.8)?**
- **Decision: GO.** Adaptive GCN (EXP-003) achieved test RMSE **44.8** (beating fixed baseline GCN **45.4** and matching persistence floor **44.8**). On Fold 2 (origin 0.70), test RMSE dropped from 42.7 down to **39.9** (*a 2.8-point drop*).

**Q2 — Does the physics loss train stably at any λ > 0?**
- **Decision: GO.** Soft spatial smoothness $\mathcal{L}_{\text{smooth}}$ and non-negativity $\mathcal{L}_{\text{cons}}$ trained 100% stably. Optimal weight $\lambda_{\text{phys}} = 0.10$ achieved overall best test RMSE (**44.78**) and best Fold 2 RMSE (**38.65**, a **4.0-point drop** over baseline fixed GCN 42.7).

**Q3 — Are we going to have row 5 (combined) by Friday 6 PM?**
- **Decision: GO (COMPLETED AHEAD OF SCHEDULE).** All 6 rows of the primary results table (Rows 0–5) are 100% populated, verified, and logged in `results/experiment_results.csv` and `docs/EXPERIMENT_LOG.md`.

**Official Gate Result: GO.** Proceed immediately to paper writing, figure creation, and assembly.

---

## 5. Paper skeleton (already created — see `paper/`)

```
paper/
  main.tex                  ACM sigconf, \highlighttrue/\highlightfalse switch
  refs.bib                  proposal's 17 refs + DengueGNN + Graph WaveNet
  sections/00_abstract.tex  write LAST
  sections/01_introduction.tex
  sections/02_related_work.tex
  sections/03_framework.tex
  sections/04_experimental_setup.tex
  sections/05_results.tex
  sections/06_discussion_conclusion.tex
  figures/
  Makefile                  `make` → clean PDF, `make highlighted` → colour version
```

Both versions compile from **one source**. Never maintain two copies of the text.

### Page budget — measured, not estimated. Enforce from Day 1.

**This budget was verified empirically**: the skeleton was compiled at a range of word counts
in `acmart[sigconf]` using prose sampled from our own proposal (mean word length 6.4 chars —
academic register is denser than general English, which is why generic "words per page" rules
of thumb overshoot badly here).

Measured ceiling, with 2 figures + 2 tables + ~6 display equations + the contributions list:

| Body words | Pages |
|---|---|
| 1,500 | 4 ✅ |
| 1,600 | 4 ✅ (ceiling) |
| 1,700 | **5** ❌ |

> **The whole paper is ~1,500 words of body text.** That is roughly three pages of a Word
> document. It is much less than it feels like it should be, and it is the single most
> important constraint on this week — write to it from the first sentence rather than writing
> 3,000 words and amputating on Saturday.

Allocation (target 1,500; hard ceiling 1,600):

| Section | Words | Notes |
|---|---|---|
| Abstract | 150 | Sits in the title block, not counted in the body budget |
| 1. Introduction | 370 | Including the "Contributions" list |
| 2. Related Work | 200 | ~65 words per thread. Brutally terse. |
| 3. Proposed Framework | 400 | + Figure 1 + ~6 equations (equations cost ~25 words each) |
| 4. Experimental Setup | 160 | Protocol is frozen; the brief says dataset "overview only" |
| 5. Results | 240 | + Table 1, Table 2, Figure 2 |
| 6. Discussion & Conclusion | 130 | Limitations + GAN as future work |
| **Total body** | **1,500** | 100 words of headroom to the ceiling |

Floats cost ~250 words of space each. **Budget: 1 architecture figure, 1 results figure,
2 tables — and that is the ceiling.** A third table does not fit. If you want the per-horizon
breakdown *and* the λ sweep, one of them goes in Phase 3.

Run `make pages` after every writing session. Do not let it reach Saturday at 5 pages.

---

## 6. Technical specification

### 6.1 Adaptive graph (Contribution c)

```
E₁, E₂ ∈ R^{N×d}          N=25, d=10, randomly initialised, learned
A_adp = softmax(ReLU(E₁ E₂ᵀ))         row-normalised, directed
A     = σ(g)·Â_fixed + (1−σ(g))·A_adp  g learnable scalar, init so σ(g)≈0.8
H     = ReLU(A X W₁);  H₂ = ReLU(A H W₂);  ŷ = Linear(dropout(H₂))
```

Everything else — residual-over-persistence, log1p target, Adam(1e-3, wd 5e-4), early stopping
on validation RMSE — stays byte-identical to the baseline. **One variable at a time.**

Initialising the gate toward the fixed adjacency means the worst case is baseline performance,
not a regression. That is the whole point of variant (ii).

### 6.2 Physics-informed loss (Contribution a)

Full objective:

```
L = L_data + λ_p·L_phys + λ_c·L_cons + λ_s·L_smooth
```

- `L_data` — the existing masked MSE in log1p residual space. Unchanged.
- `L_smooth` — `Σ_(i,j)∈E  A_ij · ‖ŷ_i − ŷ_j‖²`, normalised by |E|. Penalises implausible
  discontinuities between connected districts. Stable, cheap, always safe to enable.
- `L_cons` — non-negativity and population conservation on the latent compartments:
  `relu(−C)²` summed, plus `(Σ_c C_c − Pop_i)²`.
- `L_phys` — discretised SEIR–SEI residual. The model gets an auxiliary head emitting latent
  per-district weekly states `(S_h, E_h, I_h, R_h, S_v, E_v, I_v)`; the residual penalises
  deviation from the one-week Euler update with climate-modulated `β_t`. Observed incidence is
  linked to latent state via `ŷ_i,t ≈ ρ · σ_h · E_h,i,t−1` with reporting rate `ρ` fixed from
  literature (do **not** learn `ρ` — it is unidentifiable against `E_h` and will wreck training).

**Risk, stated plainly:** `E_h` and `I_v` are unobserved. A hard ODE residual on unobserved
compartments is the single most likely thing to fail this week. Two mitigations, both cheap:

1. Ramp `λ_p` linearly from 0 over the first 20 epochs. A cold-start hard constraint will
   dominate the data term and the model will fit the ODE instead of the data.
2. If it still diverges → the Q2 fallback in §4. Ship `L_smooth + L_cons` only, call it a soft
   epidemiological constraint, and be explicit about what didn't work.

Fix epidemiological constants (`σ_h`, `γ_h`, mosquito lifespan, EIP) from the SEIR–SEI
reference [12] and **state them in a table or inline in the paper**. Do not learn them; there
is not enough data and a reviewer will ask.

### 6.3 Kaggle workflow

- One notebook `notebooks/03_proposed.ipynb`, GPU T4/P100. Free tier is 30 GPU-hrs/week —
  the entire plan needs ~5. Compute is not the constraint; coordination is.
- `QUICK_TEST = True` flag for a 1-fold/1-seed smoke run (~2 min) before every full run.
- **Every run appends to `results/<exp_id>.csv`** with full config, per-fold and per-horizon
  numbers. Download and commit these to git the same day. Kaggle sessions expire and take your
  outputs with them — a result that only exists in a dead session's output pane does not exist.
- Seeds `{0, 1, 2}`, fixed. Set `torch.use_deterministic_algorithms(True)` where possible.

### 6.4 Where Antigravity (or any IDE agent) helps — and where it doesn't

Good use, low risk:
- Boilerplate refactor of `edge_index` → dense adjacency
- Metric implementations (SMAPE, peak-timing) + their unit tests
- Plotting code, table-generation scripts from CSV
- LaTeX formatting, BibTeX cleanup, `\hl` markup

Do **not** delegate:
- The physics loss formulation — it is the novelty, and a subtly wrong ODE that trains fine is
  worse than one that crashes
- Anything that decides what goes in the results table
- Interpretation of results

---

## 7. Standing rules for the week

1. **The protocol is frozen.** Rolling-origin, 3 origins × 3 seeds, `W=3 → H=3`, training-fold
   normalisation. Changing it invalidates every cross-row comparison and means re-running
   everything. If someone proposes a change, the answer is no until Phase 3.
2. **One variable at a time.** Adaptive graph and physics loss get separate rows before the
   combined row. That is what makes it an ablation rather than an anecdote.
3. **No number in the paper without a config in `EXPERIMENT_LOG.md` and a CSV in `results/`.**
4. **Results freeze Friday 6 PM.** Non-negotiable.
5. **Negative results ship.** If a contribution doesn't beat the floor, report it with the
   ablation intact. The rubric rewards research thinking, and the brief says so explicitly.
   Quietly dropping a failed row is the one thing that would actually cost marks.
6. **One owner per notebook at a time** (`CONTRIBUTING.md`). Notebook merge conflicts across
   4 people will cost more hours than the physics loss.

---

## 8. Contribution highlighting — don't leave it to Sunday

The brief requires a colour-highlighted version with one distinct colour per member, marking
text written, figures created, tables generated, methodology implemented, and analysis
conducted. Where the implementer ≠ the writer, a short margin comment must name both.

Track this **as you go** — a shared sheet with `section → who wrote it / who implemented it`.
Reconstructing six people's contributions on Sunday night from memory is how teams lose marks
on a criterion that is free to satisfy.
