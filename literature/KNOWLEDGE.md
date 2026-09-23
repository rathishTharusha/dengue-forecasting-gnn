# What the literature says about the architectures we are trying to beat

Knowledge extracted from 32 papers (`manifest.csv`), set against what this
project has measured. Organised by **weakness**, not by paper: each section says
what the literature reports, what we measured on our own data, what others did
about it, and whether we have tested that remedy.

**The PDFs** are third-party copyrighted works and are git-ignored, like
`papers/`. Rebuild the folder with `python literature/fetch_papers.py`, which
downloads every row of `manifest.csv`, checks each file really is a PDF, and
writes sizes and hashes to `fetch_log.csv`. 31 of 32 fetch automatically.
`kim2022revin` needs fetching by hand — OpenReview refuses automated requests.
Europe PMC and OSTI sometimes refuse or drop a request and succeed on a retry;
the script deletes any file lacking the PDF end-of-file trailer, so a truncated
download is never kept.

Our own measurements quoted below come from `seirgnn2/results/diagnose_arch.txt`
(EXP-039): validation windows only, frozen three origins, seeds averaged. Test
is not used to diagnose, because designing around test errors spends the
held-out set.

### Integrity note on `full_paper/references/`

Checking every committed reference PDF's first page against its filename found
two that are the wrong document:

| file | claims to be | actually is |
|---|---|---|
| `Guo_2019_ASTGCN.pdf` | Guo et al., AAAI 2019 | arXiv:1901.07727, a condensed-matter physics paper on spin-exchange models |
| `Hasan_2022_SEIR_SEI_Dengue.pdf` | Hasan et al. 2022, *Int. J. Analysis & Applications* | an untitled 2020 explainer, "The SEIR model of infectious diseases" |

`Raissi_2019_Physics_Informed_Neural_Networks.pdf` is the 2017 arXiv precursor
("Physics Informed Deep Learning, Part I") rather than the cited J. Comp.
Physics 2019 paper — same authors and method, so minor. The correct ASTGCN is in
this folder (AAAI OJS). The `full_paper/` copies are left untouched pending a
decision, since that package advertises all its references as verified.

---

## 1. None of the five baselines was designed for epidemics

| baseline | original paper | original task | original scale |
|---|---|---|---|
| STGAT | Huang et al., ICCV 2019 | **pedestrian trajectories** (ETH/UCY crowds) | seq-to-seq over 8 observed frames |
| A3TGCN | Zhu et al., 2020 (after T-GCN, Zhao et al. 2019) | **road traffic** speed | 12-step windows, hundreds of road segments |
| ASTGCN | Guo et al., AAAI 2019 | **highway traffic** flow (PeMS) | 12-step windows + daily/weekly periodic inputs, 307+ sensors |
| AAGCN | Shi et al., CVPR 2019 / TIP 2020 | **skeleton action recognition** | 300 video frames, 25 body joints |
| DCRNN | Li et al., ICLR 2018 | **road traffic** (METR-LA, PEMS-BAY) | seq-to-seq, 12 steps, 207–325 sensors |

Weng et al. (2024) took these off the shelf through PyTorch Geometric Temporal
and applied them to Sri Lankan dengue at window 3. Every temporal mechanism in
them was sized for 8–300 steps. Section 2 shows what that does to each one.

Liu et al.'s SEIR-LSTM *is* epidemic-specific; it is treated separately in
section 4.

---

## 2. Architecture by architecture

### STGAT — a trajectory model with a national bottleneck
- **Designed as:** graph attention between pedestrians at each time step, then
  an LSTM encoding each pedestrian's history, in a seq-to-seq decoder.
- **As wired in Weng et al.:** after the GAT, the LSTM input each week is the
  whole country as one 25-dim vector, and a single 64-d state emits all 75
  forecasts. A district is identified only by its position in that vector.
- **Measured:** worse than persistence at every horizon (native skill −0.40 /
  −0.27 / −0.15). Its residual is **47–50% predictable** by a linear model on
  the same inputs (out-of-sample R² 0.474–0.496) — it is underfitting,
  not capped.
- **Literature link:** this is a textbook case of STID's "indistinguishability"
  (Shao et al. 2022): the model cannot tell districts apart except by index.

### A3TGCN — no memory across weeks
- **Designed as:** T-GCN (a GRU whose gates are GCNs) plus attention over time
  periods.
- **As implemented in PyG-Temporal:** each week runs through the cell from a
  *zero* hidden state — the state is never carried between weeks — and the
  "attention" is three learned constants shared by every input. Three
  independent one-step graph-GRUs, averaged with fixed weights.
- **Measured:** the worst remaining model (native skill −0.82 / −0.62 / −0.46).
  Its residual is **60–62% predictable** by a linear model. Broken wiring, not
  an information limit.

### ASTGCN — the only input-dependent attention
- **Designed as:** spatial and temporal attention computed *from the current
  input*, steering a Chebyshev graph convolution, plus a temporal convolution.
  The original also takes recent, daily-periodic and weekly-periodic input
  segments — **Weng et al. use only the recent segment**, so ASTGCN runs
  without the periodic branches it was designed around.
- **Measured:** one of the three working models (skill +0.07 native, +0.10 to
  +0.14 with NB + season). Residual only 0–4% linearly predictable: it has
  extracted what the inputs contain.

### AAGCN — three views of a graph that has only two
- **Designed as:** three fixed graph partitions (self / towards body centre /
  away from it), a learned adaptive adjacency, spatial-temporal-channel
  attention and a 9-tap temporal convolution.
- **As wired:** the adaptive adjacency is off (hard-wired off in Weng et al.).
  "Inward" and "outward" are the same matrix on a district map with only 2 of
  59 borders asymmetric. The 9-tap filter over 3 weeks reads mostly padding.
- **Measured:** the best single model (skill +0.07 native, +0.11 to +0.16
  best). 1,878 parameters — its advantage is probably that it is too small to
  overfit.

### DCRNN — dropped (cost and rank)
One GRU step from zero with the 3 weeks as channels, diffusion depth K = 32 on
a graph of diameter 7 (the random walk is within 0.005 of stationary by K = 32,
so most of its 825k parameters learn from copies of a national average). Kept
in the manifest for completeness.

---

## 3. Cross-cutting weaknesses, and what others did about them

### W1. The last-value baseline is very hard to beat — for everyone
- **Literature.** SpatialEpiBench (Lyu, Turcan & Wilder 2026), 11 epidemic
  datasets: *"most methods underperform a simple last-value baseline from 1 day
  to 1 month ahead, even during outbreaks and with these priors."* The COVID-19
  Forecast Hub (Cramer et al. 2022): two-thirds of models beat a naive baseline,
  but only the **ensemble** beat it in every location and every week. DLinear
  (Zeng et al. 2023): one-layer linear models beat sophisticated Transformers
  on most long-horizon benchmarks.
- **Ours.** Cases at t−1 explain r² = 0.85 of cases at t. Best working models
  beat persistence on validation by 7% natively and 10–16% with NB + season.
- **Implication.** Beating persistence is the result that matters. A method
  that beats the five baselines but not persistence has not forecast anything.

### W2. Geographic adjacency carries little signal
- **Literature.** SpatialEpiBench's third failure mode: *"limited utility of
  common geographic adjacency for epidemiological spatial information."*
  BasicTS (Shao et al. 2024) finds spatial models help only on datasets whose
  series are strongly spatially dependent, and that ignoring this heterogeneity
  explains most of the field's conflicting claims. Panagopoulos et al. (2021)
  and MepoGNN (Cao et al. 2023) use **mobility** flows, not borders, as the graph.
- **Ours.** Removing message passing entirely costs 0.07 RMSE. Case *levels*
  are spatially clustered (Moran's I 0.27), but the reproduction number is not
  (Moran's I 0.005): neighbours share how many cases they have, not whether
  cases are about to rise. Residuals leave a small neighbour excess
  (+0.17 vs +0.12 between non-neighbours) on top of a **national common
  component** (+0.12 between any two districts).
- **What others did.** Learned or mobility-based graphs (Graph WaveNet, Cola-GNN,
  MepoGNN); we tested learned adjacency — `adaptive` / `hybrid` — and it did
  not help. We have no mobility data.

### W3. Distribution shift between regimes
- **Literature.** RevIN (Kim et al. 2022): time-series statistics change over
  time, and normalising each *input instance* by its own mean and variance, then
  restoring them on the output, fixes much of it. Non-stationary Transformers
  (Liu et al. 2022) warn that over-normalising hides bursty events and add the
  statistics back into attention.
- **Ours.** The series switches regime. The 2017 DENV-2 epidemic peaked at
  **5.4×** anything in the training data of the fold whose validation window
  contains it; every model collapsed there (validation RMSE 140–227 against
  13–48 on every other fold). 2020–2022 then ran at 5–10% of the historical
  peak. Every architecture here normalises with **one mean and spread per fold**.
- **Status:** untested. **Candidate remedy R1.**

### W4. Districts are indistinguishable to the model
- **Literature.** STID (Shao et al. 2022): *"the indistinguishability of samples
  in both spatial and temporal dimensions"* is a key bottleneck. It attaches a
  learned spatial embedding E ∈ R^{N×D} and time-of-day / day-of-week
  embeddings to a plain MLP, and matches or beats STGNNs at a fraction of the
  cost.
- **Ours.** No working encoder receives a district identity; STGAT gets one only
  as a vector position. The temporal half of STID is what our seasonal feature
  already is — and that was the largest input-side gain we measured (−0.88).
- **Status:** spatial half untested. **Candidate remedies R2 (identity on our
  encoders) and R3 (STID itself as a competitor).**

### W5. Outbreaks are under-predicted — worse than by persistence
- **Literature.** SpatialEpiBench failure mode 1: *"poor outbreak anticipation."*
  Ding et al. (2019): deep models trained on squared error systematically miss
  extreme events; they propose an extreme-value loss and a memory module.
  Johansson et al. (2019), 16 teams forecasting dengue in Peru and Puerto Rico:
  *"early season skill was low, and skill was generally lowest for high
  incidence seasons, those for which forecasts would be most valuable."* The
  same failure, in dengue, from a different continent.
- **Ours.** On outbreak cells the working models under-predict by 22–25 cases
  natively (persistence: 15) and 17–23 with NB + season. The top 5% of cells
  carry ~60% of all squared error.
- **Status:** NB likelihood addressed part of it (−0.41). Tail-weighted losses
  untested and risky for mean RMSE.

### W6. The models are damped persistence
- **Ours.** Regressing predicted on realised log-growth gives slope 0.23–0.31;
  predicted growth varies half as much as real growth (sd ratio 0.50–0.61).
  They under-predict rising weeks by 11–14 cases and over-predict falling weeks
  by 9–11 — the signature of a lag. 80–85% of their squared error sits in weeks
  that actually moved.
- **Literature.** This is the conditional-mean behaviour DLinear and the
  forecasting literature describe for near-random-walk targets: under squared
  error, shrinking toward the last value is *optimal* when the change is mostly
  unpredictable. It is a property of the target, not a bug in any one model.

### W7. The loss does not match the metric
- **Literature.** DeepAR (Salinas et al. 2020) trains count series under a
  negative-binomial likelihood rather than squared error on transformed values.
- **Ours.** Squared error on log1p, scored by RMSE on counts, is biased low by
  construction (it targets the median; RMSE rewards the mean). Switching to NB
  was the largest controlled gain in the study: −0.41, 9/9, p_adj = 0.013.
- **Status:** done, and carried into every candidate below.

### W8. Physics used as a decoder, rather than as a regulariser
- **Literature.** EINNs (Rodríguez et al. 2023) keep the mechanistic ODE out of
  the forecast path: they *"do not need to numerically solve the ODE equations
  during training,"* and transfer what a physics-informed network learns into a
  separate forecaster. MepoGNN (Cao et al. 2023) and CausalGNN (Wang et al.
  2022) let the GNN estimate time- and region-varying epidemic *parameters*
  inside a metapopulation model whose spatial coupling comes from mobility.
  The GNN review (Liu et al. 2024) classifies this as the hybrid family and
  notes it is where interpretability lives.
- **Ours.** Routing the forecast *through* SEIR (the `foi` head) costs ~7 RMSE
  and survives swapping in real encoders; the λ = 0 floor is unreachable for
  14–16% of targets; λ is only r² ≈ 0.25 predictable. The gated `foi_res` works
  by turning the physics off. The manuscript's own finding — a constraint pays,
  a reconstruction does not — is the same conclusion EINN reached by design.
- **Status:** physics-as-regulariser untested in `seirgnn2`. **Candidate R6.**

### W9. The working models make the same mistakes
- **Literature.** Every multi-team forecasting evaluation — the dengue challenge
  (Johansson et al. 2019), the COVID-19 Forecast Hub (Cramer et al. 2022) —
  finds the ensemble most consistently accurate. Cramer et al. are precise
  about *how*: the ensemble was the only model in the top half for over 75% of
  its forecasts, *"although it made the single best forecast less frequently
  than any other model."* It wins on consistency, not on being best — and that
  only happens because component errors differ.
- **Ours.** ASTGCN, AAGCN and the LSTM have **residual correlations of
  0.98–0.99**. They are, to within noise, the same forecaster. Averaging them
  cannot help, and it doesn't: the mean of the five direct encoders scores
  16.73 against 15.85 for AAGCN alone.
- **What would help.** Ensemble members from genuinely different families. The
  2026 dengue network meta-analysis (Benjarattanaporn et al.) ranks k-nearest-
  neighbour analogue forecasting, vector autoregression, Kalman filtering and
  GLMs at the top on RMSE — ahead of or level with neural networks, and all
  ahead of the naive baseline. **Candidate remedies R4 and R5.**

### W10. How results get reported
- **Literature.** Bracher et al. (2021) set the weighted interval score as the
  standard for probabilistic epidemic forecasts; point RMSE alone discards the
  uncertainty that makes a forecast usable. Johansson et al. (2019) and the
  2025 neural-network dengue review both flag inconsistent horizons, selection
  and baselines across published studies.
- **Ours.** The S9 audit (EXP-037) found a "9-origin paired test" that ran on 3
  origins against hardcoded scalars, with one p-value typed in. Liu et al.
  compare against a plain LSTM only, with no naive baseline (section 4).

---

## 4. Liu et al. (2025) SEIR-LSTM — what "beating it" can mean

PLOS Computational Biology, *An explainable covariate compartmental model for
predicting the spatio-temporal patterns of dengue in Sri Lanka*.

- **Model.** An LSTM over covariates (household income, over-60 share, mean
  temperature, precipitation, NDVI) and new cases outputs
  λ(t) = λ_min + (λ_max − λ_min)·σ(·) with λ ∈ [0, 0.01]; a human SEIR,
  **discretised weekly by forward Euler**, turns λ into cases; true infections
  are taken as 11× reported.
- **Initial state.** Susceptibles from district population minus infected and
  immune (seroprevalence), E = 0, I from the first week's cases.
- **Immunity reset.** They set population immunity to zero at the start of 2017
  for the DENV-2 introduction and report this *"improves predictions,
  particularly at the beginning of the outbreak."* The switch date is known only
  in hindsight; under our plan R4 this is an `oracle_*` arm.
- **Protocol.** 25 districts merged into **16** regions; train 2011–2018,
  validate on 2019; the model **forecasts the whole year recursively**, feeding
  its own predictions back. Loss: SMAPE.
- **Results.** Hybrid RMSE 733 vs pure LSTM 764 (MAE 491 vs 527) on 2019. **No
  naive or statistical baseline.** The authors write that *"both models struggle
  to reliably predict the spatio-temporal occurrence of dengue cases"* in 2019.

**Consequence for this project.** Liu et al. and we solve different problems —
a 52-week recursive forecast on 16 merged regions versus 1–3 weeks ahead on 25
districts — so their numbers cannot be compared with ours, and "outperform
SEIR-LSTM" only has meaning inside one harness. `seirgnn2` provides that: Liu's
encoder behind the same heads, loss and folds as every GNN. Two of their
choices are known weaknesses our harness already repairs — one λ held across
the horizon, and weekly Euler steps (unstable with their γ; `seir_sim` uses
daily exponential flows).

---

## 5. Remedies, ranked by evidence

| id | remedy | weakness | evidence from literature | evidence from our data | cost |
|---|---|---|---|---|---|
| R1 | Reversible instance normalisation | W3 | RevIN, NS-Transformer | 2017 at 5.4× train max; every model collapsed there | low |
| R2 | District identity embeddings on our encoders | W4 | STID, BasicTS | no encoder can tell districts apart | low |
| R3 | STID (MLP + identities) as a competitor | W4, W1 | STID beats STGNNs | graph worth 0.07 | low |
| R4 | Strong statistical members: NB-GLM / linear | W9, W1 | dengue NMA, DLinear | residual corr 0.98 among neural models | low |
| R5 | Ensemble across *different* families | W9 | Johansson, Cramer | same | none (post hoc) |
| R6 | Physics as auxiliary regulariser, not decoder | W8 | EINN, MepoGNN | decoder costs ~7 RMSE | medium |
| — | Mobility graph | W2 | MepoGNN, Panagopoulos | Moran's I of R ≈ 0 | **needs data we do not have** |
| — | Tail-weighted loss | W5 | Ding et al. | top 5% of cells = 60% of SSE | risky for RMSE |

Pre-registered as `docs/REMEDIES_PLAN.md`. **Result (EXP-040): none adopted.**

| id | outcome on validation, vs the best model B |
|---|---|
| R1 | RevIN **hurt**: +1.27, lost 9/9 (mean-only variant +0.96). Removing each window's level discards information — the over-stationarisation Liu et al. 2022 warn about |
| R2 | district identity: +0.06, n.s. |
| R3 | STID: +0.71, lost 7/9 |
| R4 | NB-GLM +1.32, k-NN +0.87 — both still beat persistence |
| R5 | ensembles: −0.01 to +0.13, n.s. — no member errs differently enough |
| R6 | SEIR as auxiliary constraint: −0.04 at w = 0.1, 6/9, n.s. Neither helps nor hurts |

**The finding that matters:** every family — including k-NN, which has no network
and no training — produces errors correlated 0.91–0.99 with B's. Different
machinery, same mistakes: the remaining error is in the data. This is also why
ensembles, the most reliable remedy in this literature, have nothing to work with.

---

## 6. Index

| key | category | one-line takeaway |
|---|---|---|
| weng2024 | baseline | The benchmark: five off-the-shelf ST-GNNs on Sri Lankan dengue; no naive baseline reported |
| liu2025seirlstm | baseline | LSTM→λ→weekly-Euler SEIR; 52-week recursive forecast on 16 regions; oracle immunity reset |
| huang2019stgat | baseline | Pedestrian-trajectory model; GAT per step + LSTM, seq-to-seq |
| zhao2019tgcn | baseline | GRU with GCN gates — the cell inside A3TGCN |
| zhu2020a3tgcn | baseline | T-GCN + attention over periods (library version carries no state between weeks) |
| guo2019astgcn | baseline | Input-dependent spatial/temporal attention; designed with periodic input branches Weng drops |
| shi2019agcn | baseline | Skeleton actions: fixed body partitions + adaptive adjacency |
| shi2020msaagcn | baseline | The AAGCN proper: adds spatial-temporal-channel attention |
| li2018dcrnn | baseline | Diffusion convolution in a seq-to-seq GRU; K-step random walks |
| zeng2023dlinear | critique | One-layer linear models beat Transformers on most LTSF benchmarks |
| shao2022stid | critique | Identity embeddings + MLP rival ST-GNNs; indistinguishability is the bottleneck |
| shao2024basicts | critique | Fair benchmark of 45+ models; spatial models help only on spatially dependent data |
| kim2022revin | critique | Normalise each instance, restore on output; fixes distribution shift (manual fetch) |
| liu2022nstransformer | critique | Stationarisation can erase bursty events; restore statistics inside the model |
| ding2019extreme | critique | Squared-error models miss extremes; extreme-value loss + memory |
| salinas2020deepar | critique | Probabilistic forecasting with count likelihoods (NB) |
| kapoor2020covidgnn | epidemic | ST-GNN on US counties with mobility; 6% RMSLE over best baseline |
| panagopoulos2021transfer | epidemic | MPNN(+LSTM) on mobility graphs; transfer across countries |
| deng2020colagnn | epidemic | Cross-location attention learns the graph for ILI |
| wang2022causalgnn | epidemic | Causal/SIRD model guides GNN embeddings |
| cao2023mepognn | epidemic | GNN estimates parameters of a metapopulation SIR with a learned propagation graph |
| rodriguez2023einn | epidemic | Mechanistic model as knowledge transfer, not a decoder; no ODE solve in training |
| zheng2024heatgnn | epidemic | Epidemiology-informed GNN handling regional heterogeneity |
| liu2024gnnreview | epidemic | Taxonomy of GNNs in epidemic modelling; hybrid family |
| rodriguez2022datacentric | epidemic | Survey of data-centric epidemic forecasting |
| lyu2026spatialepibench | epidemic | 11 datasets: most models lose to last-value; adjacency of limited use |
| johansson2019dengue | evaluation | 16 teams, Peru & Puerto Rico dengue: skill lowest in high-incidence seasons; ensembles help |
| cramer2022ensemble | evaluation | COVID Forecast Hub: ensemble most consistent, rarely the single best; 2/3 of models beat naive |
| bracher2021wis | evaluation | Weighted interval score for probabilistic epidemic forecasts |
| dengue_nn_review2025 | evaluation | 62 studies: mostly shallow nets; inconsistent horizons and evaluation |
| benjarattanaporn2026nma | evaluation | Network meta-analysis: k-NN, VAR, Kalman, GLM top dengue RMSE; all beat naive |
| dengue_hybrid_guangdong2025 | evaluation | Hybrid deep-learning/mechanistic dengue framework (Guangdong) |

---

## 7. Generative augmentation — would synthetic data help?

Asked after EXP-040: if the limit is the data, can a GAN generate better data?
Fifteen further papers (`category = augmentation` in `manifest.csv`), set against
two measurements of our own.

### What our data says first

- **The premise does not hold for more-of-the-same.** A learning curve (EXP-041)
  trains the best model B on 25 / 50 / 75 / 100% of its real training windows:
  validation RMSE 15.85 / 15.88 / 15.64 / 15.66. Four times the real data buys
  0.19 (one seed sd), and nothing past 75%. A generator fitted to the training
  windows can at best add more samples of the same distribution, so it cannot
  beat what four times the real data does not.
- **No overfitting to regularise away.** B's skill over persistence on its own
  training windows is *lower* than on validation (mean −0.12 vs +0.12; the 2017
  epidemic dominates training RMSE). Augmentation as a regulariser addresses
  variance; this model is not variance-limited.
- **It has been tried here.** EXP-010 (legacy data, older model): a GAN scored
  RMSE 93.81 against 61.81 with no augmentation; jittering and window-warping
  were also worse than none. Verdict then: *"augmentation does not help this
  dataset."*

### What the literature says about GANs for time series

- **TimeGAN** (Yoon, Jarrett & van der Schaar, NeurIPS 2019) is the reference
  method: an embedding network, a supervised next-step loss and an adversarial
  loss trained jointly. Its benchmarks (sines, stocks, energy, events) supply
  thousands of sequences. Ours supply ~300 training windows per fold.
- **Train on synthetic, test on real (TSTR)** (Esteban, Hyland & Rätsch 2017) is
  the standard fidelity test: if a model trained only on synthetic data does
  much worse than one trained on real data, the generator has not captured what
  matters for the task.
- **Generated data loses the tails.** Shumailov et al. (Nature 2024): training
  on model-generated data causes *"irreversible defects ... where tails of the
  original content distribution disappear."* On this series the tail *is* the
  problem: the top 5% of cells carry ~60% of squared error, and outbreaks are
  under-predicted by every model (EXP-039).
- **Gains shrink as data grows.** Semenoglou, Spiliotis & Assimakopoulos
  (Pattern Recognition 2023), nine augmentation methods for forecasting: gains
  are larger for deeper networks and *"become less significant as the initial
  size of the set increases"*; combining and upsampling series worked best.
  (Paywalled; not in `pdfs/`.)
- **Assumptions must fit the data.** Iwana & Uchida (2021): jittering, for
  example, *"assumes that it is normal for the time series patterns of the
  particular dataset to be noisy"*; no augmentation family helps universally.
- **Pooled augmentation helps global models when data is scarce.** Bandara et
  al. (2021) use GRATIS, moving-block bootstrap and DBA, with gains in *"less
  data-abundant settings."* Chronos (Ansari et al. 2024) pretrains foundation
  models partly on Gaussian-process synthetic series.

### Where synthetic data *did* help epidemic forecasting — and how

Every case found generates data from a **mechanistic simulator**, not a GAN:

| study | generator | result |
|---|---|---|
| DEFSI (Wang, Chen & Marathe, AAAI 2019) | agent-based epidemic simulations | trained on synthetic data; significantly beat other methods at county level, where real data is thinnest |
| Osthus et al. (PLOS Comp Bio 2026) | agent-based model MutAntiGen, ~36,000 series through an observation-noise model | models trained on synthetic data beat real-data models in over 75% of bootstrap samples; best rMAE 0.777 vs persistence |
| Dimarco et al. (2025) | calibrated compartmental model plus uncertainty | *"significantly improved predictive performance"* for neural forecasters |

Osthus et al. explain their gain as **covariate shift**: the COVID-19 test data
was *"overwhelmingly classified as synthetic data rather than non-COVID-19, real
respiratory data"* — the simulator covered dynamics the real history had never
shown. That is the one mechanism by which synthetic data adds something a
learning curve cannot measure: it changes *which* situations training covers,
not how many samples there are. Our catastrophic fold is that situation — the
2017 epidemic at 5.4× anything in its training data.

### Alternatives for the tail that generate nothing

Deep imbalanced regression (Yang et al., ICML 2021) reweights training by the
smoothed density of target values (label distribution smoothing), so rare large
targets carry more weight without synthesising any data.

### Implication

| arm | expected | why |
|---|---|---|
| GAN (TimeGAN) augmentation | no gain, possible harm | flat learning curve; ~300 windows is too little to train a GAN; tails collapse; EXP-010 |
| jitter / magnitude scaling | no gain | same distribution; EXP-010 |
| SEIR-simulated epidemics | the only arm with a mechanism | covers regimes absent from training, as in Osthus / DEFSI |
| outbreak reweighting | better outbreak bias, likely worse RMSE | reweighting is not new information |

Pre-registered as `docs/AUGMENTATION_PLAN.md`; results as EXP-042.
