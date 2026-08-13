# Producing the Baseline & Completing the Proposal

This is the working guide for turning `GNN for Dengue Forecasting.md` (literature
review) + the `notebooks/` baseline pipeline into a finished project proposal for the
DNN module.

I don't have your course's actual proposal template/rubric (page limit, required
sections, submission format) — this outline follows standard ML-research-proposal
structure and maps directly onto what you already have. If your module gives you a
template, slot this content into it rather than following my section headers literally.

---

## 1. What you already have

| Asset | Where |
|---|---|
| Literature review (3 pillars: GNN, physics-informed loss, GAN augmentation) | `GNN for Dengue Forecasting.md` |
| Direct baseline paper (Weng et al. 2024) | `papers/GraphRepresentation_Dengue_IEEEbigData2024.pdf` |
| SEIR mechanistic model reference | `papers/09_SEIR_model.pdf` |
| Reference implementation (cloned for inspection) | `reference_repo/` — `evaluation.py`, `gnn_models.py`, `arima.py`, `lstm.py`, `random_forest.py`, `sri_lanka_adj_list.json`, `results.txt` (their actual numbers) |
| Runnable baseline notebooks | `notebooks/00_data_setup_eda.ipynb`, `01_baselines_classical.ipynb`, `02_baselines_gnn.ipynb` |

`reference_repo/` is read-only reference material pulled from
[`MLOpenSourceOpenScience/disease_modeling_MLOS2`](https://github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2)
— useful to cross-check the notebooks against, not something to run directly.

---

## 2. Running the baseline notebooks

Order matters — each notebook writes files the next one reads.

1. **`00_data_setup_eda.ipynb`** — mounts your Drive (or Kaggle), locates
   `MLSO2_Final.csv` / `sri_lanka_2013-2022_shifted.npy` / `sri_lanka_adj_list.json`
   from your teammate's shared folder, runs EDA, writes `data_manifest.json`.
   - **Before running:** open the shared Drive folder once and click **"Add shortcut to
     Drive"** so it's visible under your own `MyDrive` — Colab can't mount someone
     else's Drive directly. The notebook falls back to `gdown` if you skip this, but
     the shortcut is more reliable.
2. **`01_baselines_classical.ipynb`** — ARIMAX, Random Forest, XGBoost, ARNN, LSTM
   (+ naive/seasonal-naive) per district, rolling-origin evaluated. Writes
   `results/baseline_classical_all.csv`.
3. **`02_baselines_gnn.ipynb`** — STGAT, A3TGCN, ASTGCN, DCRNN, AAGCN using the exact
   reference architectures (cloned live from the repo). Writes
   `results/baseline_gnn_all.csv` and a combined leaderboard.

**Workflow:** every notebook has a `QUICK_TEST` flag. Run with `QUICK_TEST = True`
first (few districts, 1 segment, few epochs) to confirm the whole pipeline executes
without errors — this takes minutes. Once it's clean, set `QUICK_TEST = False` and run
the full reproduction (all 25 districts, all 5 rolling segments, full epoch counts) —
budget real time for this, particularly ARIMAX and the GNNs; use a GPU runtime for
notebook 02 if you can.

Each notebook prints your numbers next to Weng et al.'s published numbers
(`REFERENCE_CV_SHIFTED` dict, sourced from their Table I / `results.txt`). Exact parity
isn't the bar — matching *ordering* is: GNNs (esp. STGAT) should beat all classical
baselines, and Random Forest/XGBoost should beat plain ARIMA/LSTM on this feature set.
If your numbers are wildly off (not just shifted, but reordered), that's a bug to chase
before writing anything up.

---

## 3. Proposal outline

### 3.1 Title & framing
Working title from the lit review:
*"Physics-Informed, GAN-Augmented Graph Neural Networks for Dengue Incidence
Forecasting."* State up front (the TL;DR of your lit review is almost this verbatim):
a spatio-temporal GNN backbone is well-justified, epidemic-mechanistic losses and GAN
time-series augmentation are each independently validated, but **no published work
combines all three** — that's your gap.

### 3.2 Problem & motivation
- Dengue as a public-health problem in Sri Lanka (2017 outbreak: 186,000+ cases, 440
  deaths — cite Tissera et al. 2020, already in Weng et al.'s references).
- Why spatial structure matters for a vector-borne disease (mosquito movement +
  human mobility across district borders) — this is *why* GNN over plain time-series.
- Why data scarcity motivates the other two pillars: ~520 weeks × 25 districts is a
  small dataset by deep-learning standards — physics-informed losses inject domain
  knowledge to compensate, GAN augmentation manufactures more training signal.

### 3.3 Related work
Your literature review's five areas map directly to a related-work section:
1. GNN / spatio-temporal methods for epidemic forecasting (§ Area 1)
2. Physics-informed / mechanistic-model-informed NNs (§ Area 2)
3. GAN-based augmentation for scarce time series (§ Area 3)
4. Combined/hybrid approaches — this is where you state the gap explicitly (§ Area 4)
5. Evaluation practices and baselines (§ Area 5)

Keep the **"to the best of our knowledge"** hedge from the lit review's caveats — the
novelty claim is absence-of-evidence from a targeted search, not an exhaustive proof.
Also flag the preprint-status caveats (several 2026-dated / bioRxiv sources) so a
reviewer doesn't catch you treating unrefereed claims as settled.

### 3.4 Baseline reproduction — **this is what the notebooks produce**
- State clearly: *"We reproduce Weng et al. (2024)'s benchmark on the identical
  dataset and protocol (25 Sri Lanka districts, 2013–2022, 3-week window → 3-week-ahead
  forecast, rolling-origin CV) before introducing any novel components."*
- Include the combined leaderboard table/chart from `02_baselines_gnn.ipynb`'s last
  section (classical vs. GNN, sorted by RMSE).
- Report how closely your numbers track the paper's (`results/` CSVs + your own
  narrative — a table like the printed "Our vs Paper" comparison from the notebooks).
- **Worth a paragraph of critique** (this strengthens the proposal, doesn't weaken it):
  the paper's classical baselines are 1-step-ahead / contemporaneous-feature models
  while the GNNs are genuinely 3-step-ahead — the notebooks flag this exact
  inconsistency. Noting it shows you understood the protocol deeply enough to spot its
  own limitation, which is good research literacy for a proposal.

### 3.5 Proposed method (the actual novelty)
Structure as staged additions, each with its own falsifiable success criterion — this
is straight from the lit review's Recommendations section:

**Stage 0 — architecture increment (low-risk, do first).**
Upgrade Weng et al.'s fixed distance-based adjacency to a learned/adaptive adjacency
(Graph WaveNet-style `A_adaptive = g(E_A · E_Aᵀ)`). Cheap to try, itself a defensible
novelty increment, and de-risks the pipeline before adding the harder pieces.

**Stage 1 — physics-informed loss.**
`L = L_data + λ_phys·L_residual + λ_cons·L_population-conservation + λ_smooth·L_neighbor-smoothness`,
grounded in the SEIR-SEI host-vector model (Ross-Macdonald framework; cite EINN,
CausalGNN, dengue DINN). **Success criterion:** improves 3-step-ahead RMSE and
peak-season (out-of-distribution) accuracy vs. the Stage-0 GNN. If it doesn't, the lit
review's fallback is latent-dynamics transfer (EINN-style) rather than a hard residual —
dengue's exposed-human/infected-mosquito compartments are unobserved, so a hard
constraint may over-penalize.

**Stage 2 — GAN augmentation.**
TimeGAN or a conditional GAN (RCGAN-style, conditioned on meteorological covariates and
outbreak labels), with WGAN-GP / gradient penalty given how prone GANs are to mode
collapse on ~520-timestep series. **Always keep a jittering/window-warping baseline** as
a cheap fallback comparator. **Success criterion:** measurable downstream RMSE/CRPS
improvement on held-out rolling origins — not just distributional similarity to real
data.

**Stretch goal — PID-GAN-style coupling.**
Physics residual informs the GAN discriminator directly, so generated series are
epidemiologically consistent. Maximizes the novelty claim but is the highest-risk
component — position it as a stretch goal, not a deliverable the proposal depends on.

### 3.6 Evaluation plan
- Rolling-origin CV (already in the baseline notebooks — carry the same protocol
  forward for the novel components, so baseline and novel results are comparable).
- Point metrics: RMSE, MAE, MAPE (already computed by the notebooks).
- Add for the novel-component stages: CRPS, PICP/MPIW (probabilistic calibration —
  needed once the GAN/physics components produce distributions, not just point
  forecasts), Moran's I on residuals (checks the graph captures genuine spatial
  structure rather than spurious spillover — the paper itself flags this as a known
  GNN failure mode).
- **The ablation table is the deliverable that proves the novelty claim:** (a) GNN
  alone [Stage 0 / this baseline], (b) GNN + physics loss, (c) GNN + GAN augmentation,
  (d) all three. Structure your results section around filling in this table.

### 3.7 Timeline & risk register
Be concrete about what's already done vs. remaining:
- ✅ Literature review
- ✅ Baseline reproduction (classical + GNN) — *this milestone*
- ☐ Stage 0 (adaptive adjacency)
- ☐ Stage 1 (physics-informed loss) + ablation vs. Stage 0
- ☐ Stage 2 (GAN augmentation) + ablation vs. Stage 1
- ☐ Full ablation table + write-up

Risks worth naming explicitly (straight from the lit review's Caveats section):
- GAN training instability on ~520×25 data — mitigation: WGAN-GP, fallback to
  jittering/window-warping if mode collapse persists.
- Physics loss over-constraining latent/unobserved compartments — mitigation: fall
  back to EINN-style latent-dynamics transfer instead of a hard ODE residual.
- Several cited papers are 2026-dated preprints — verify each before the final
  submission in case a preprint gets revised or withdrawn.
- Compute: GNN training is materially heavier than the classical baselines (the paper
  notes this too) — budget Colab/Kaggle GPU time accordingly, especially for Stage 2.

### 3.8 References
Reuse the literature review's citation list wholesale — it's already organized by area
and has DOIs/arXiv IDs for nearly everything. Priority citations for the proposal
itself (not just background): Weng et al. (2024) [baseline], EINN (Rodríguez et al.
2023) [physics loss], TimeGAN (Yoon et al. 2019) [augmentation], CausalGNN (Wang et al.
2022) [closest GNN+physics prior art].

---

## 4. Practical notes

- **Don't let the baseline reproduction become the whole project.** It's one section
  of the proposal, not the deliverable — budget time to at least prototype Stage 0/1
  before the proposal is due, even if results are preliminary, so the proposal isn't
  purely speculative about the novel components.
- **Keep `QUICK_TEST` runs for iteration, full runs for the numbers you actually cite.**
  Don't accidentally paste QUICK_TEST leaderboard numbers into the proposal — they're
  intentionally degraded (fewer districts/epochs) and won't match the paper.
- If your team splits work, the natural split is: one person owns notebook 00+01
  (classical baselines + data pipeline), another owns notebook 02 (GNN baselines,
  heavier compute), and Stage 0/1/2 prototyping starts once both are validated.
