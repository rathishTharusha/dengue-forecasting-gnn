# Chapter 11 — Glossary and References

---

## Symbols

### Forecasting

| symbol | meaning |
|---|---|
| $y_t^{(i)}$ | cases in district *i*, week *t* |
| $\hat{y}$ | a prediction |
| $W$ | window — input weeks (3 here) |
| $H$ | horizon — forecast weeks (3 here) |
| $N$ | nodes / districts (25) |
| $T$ | weeks (459) |
| $F$ | feature channels (11) |
| $B$ | batch size |

### Graphs

| symbol | meaning |
|---|---|
| $A$ | adjacency matrix, $N \times N$ |
| $\hat{A}$ | symmetrically normalised: $D^{-1/2}AD^{-1/2}$ |
| $\tilde{A}$ | learned/adaptive adjacency |
| $D$ | degree matrix (diagonal, $D_{ii}=\sum_j A_{ij}$) |
| $L$ | graph Laplacian, $D - A$ |
| $H$ | node feature matrix, $N \times F$ (context distinguishes it from horizon) |
| $E_1, E_2$ | learned node embeddings, $N \times d$ |
| $\mathcal{N}(i)$ | the neighbours of node *i* |
| $\alpha_{ij}$ | attention weight on edge $i \to j$ |

### Learning

| symbol | meaning |
|---|---|
| $\theta$ | all model parameters |
| $\mathcal{L}$ | loss |
| $\eta$ | learning rate |
| $\lambda$ | regularisation weight (also: epidemic growth rate — context) |
| $\sigma(\cdot)$ | a non-linearity (also: standard deviation — context) |
| $\nabla_\theta$ | gradient with respect to $\theta$ |

### Epidemiology

| symbol | meaning |
|---|---|
| $S, E, I, R$ | Susceptible, Exposed, Infectious, Removed |
| subscript $h$ / $v$ | host (human) / vector (mosquito) |
| $R_0$ | basic reproduction number |
| $\beta$ | transmission rate (SEIR); $\nu_h + \mu_h$ (SEIR–SEI) |
| $\gamma$ | recovery rate (SEIR); $\gamma_h + \mu_h$ (SEIR–SEI) |
| $\sigma$ | incubation rate (SEIR) |
| $\alpha, \delta$ | vector→host, host→vector transmission force |
| $\epsilon$ | vector death rate, $= \mu_v$ |
| $b$ | biting rate |
| $\mu_h, \mu_v$ | host, vector death rates |
| $\nu_h, \nu_v$ | host, vector incubation rates |
| $\pi_v$ | vector recruitment rate |
| $m$ | vector-to-host ratio $N_v/N_h$ |
| $\mathbf{F}, \mathbf{V}$ | new-infection and transition matrices |
| $\rho(\cdot)$ | spectral radius (largest \|eigenvalue\|) |
| $\Upsilon^{R_0}_q$ | normalised forward sensitivity index of $R_0$ to $q$ |

---

## Terms

**A3TGCN** — Attention Temporal GCN. A GCN inside a GRU, plus attention over the
window. §7.2

**AAGCN** — Adaptive Adjacency GCN. Two-stream; its adaptive branch exists but is
hard-wired off in the reference. §7.5

**Adaptive adjacency** — a learned graph, $\text{softmax}(\text{ReLU}(E_1E_2^\top))$.
Null result here. §6.4, §6.5

**Adjacency matrix** — $A_{ij}=1$ when an edge joins *i* and *j*. §6.1

**ADR** — Architecture Decision Record. `docs/decisions/`. ADR 0001 covers
residual-over-persistence, log1p, and train-only normalisation.

**ASTGCN** — Attention-based Spatio-Temporal GCN. Spatial attention + Chebyshev
convolution + temporal attention + 1-D convolution. §7.3

**Attention** — learned, input-dependent weighting. §6.4

**Backpropagation** — the chain rule applied backwards through a network. §4.3

**Batch** — a subset of training examples processed together. §4.3

**Chebyshev convolution** — polynomial approximation of a spectral graph filter;
`K` gives a `K`-hop receptive field in one operation. §7.3

**CRPS** — Continuous Ranked Probability Score. Generalises MAE to distributions;
rewards accuracy *and* appropriate confidence. §5.1

**Cross-validation, rolling-origin** — expanding-window CV for time series;
training always precedes test. §5.3

**DCRNN** — Diffusion Convolution Recurrent Network. A GRU with diffusion
convolutions; `K=32` here. §7.4

**Diffusion convolution** — aggregates multiple hop-distances at once, each with
its own weights. §6.4

**Dropout** — randomly zeroing activations during training. Must be off at
evaluation. §4.4

**Early stopping** — halt when validation stops improving; **restore the best
weights**. §4.4

**Edge index** — a `(2, E)` tensor of source/destination indices; PyG's sparse
graph representation. §7.6

**Endemic equilibrium** — a steady state with nonzero infection. §2.2

**Epoch** — one full pass over the training set. §4.3

**Expanding window** — see rolling-origin. §5.3

**Extrinsic incubation period** — time between a mosquito biting an infected host
and becoming infectious (8–12 days). §1.1

**GAT** — Graph Attention Network. Learns per-edge weights, but only over existing
edges. §6.4

**GCN** — Graph Convolutional Network. $H'=\sigma(\hat{A}HW)$. §6.2

**Gradient clipping** — bounding the gradient norm so one bad batch cannot destroy
learned weights. §4.3

**GRU** — Gated Recurrent Unit. An LSTM with two gates instead of three. §7.0

**Herd immunity threshold** — $v^* = 1 - 1/R_0$. §2.2

**Huber loss** — quadratic near zero, linear far away; robust to outliers. §4.2

**Jensen's inequality** — for a concave function, $f(\text{mean}) \ge
\text{mean}(f)$. Why per-window RMSE understates pooled RMSE by 39% here. §5.4

**Leakage** — any path by which test-period information reaches the model. Four
kinds catalogued in §4.6.

**log1p** — $\log(1+x)$; defined at zero, numerically stable for small *x*. §3.3

**LSTM** — Long Short-Term Memory. Additive cell state with three gates; solves
vanishing gradients. §7.0

**MAE / RMSE** — mean absolute / root mean squared error. §5.1

**MAPE** — mean absolute percentage error. Undefined on zeros; the reference
implementation's `1e-15` floor makes it explode to ~10¹⁷. §5.1

**Message passing** — `A @ H`: each node's new features are an aggregate of its
neighbours'. §6.1

**Moran's I** — spatial autocorrelation; a correlation coefficient for space. §5.1

**MPIW** — Mean Prediction Interval Width. Meaningless without PICP. §5.1

**Next-generation matrix** — $\mathbf{F}\mathbf{V}^{-1}$; its spectral radius is
$R_0$. §2.3

**Node embedding** — a learned vector per node. §6.4

**Overfitting** — memorising training data; training loss falls while validation
loss rises. §4.4

**Persistence** — $\hat{y}_{t+h}=y_{t-1}$. The baseline that matters. §5.2

**PICP** — Prediction Interval Coverage Probability. §5.1

**Physics-informed loss** — a penalty derived from a mechanistic model. Here,
blocked by unobserved compartments. §2.6

**Pooled vs per-window** — one RMSE over all predictions, versus the mean of
per-window RMSEs. Differ by 39% here. §5.4

**ReLU** — $\max(0,x)$. §4.1

**Residual over persistence** — predict the *correction* to last week's value.
RMSE 66 → 45. §4.7

**$R_0$** — expected secondary infections from one case in a fully susceptible
population. §2.2

**Rolling origin** — see cross-validation. §5.3

**SEIR / SEIR–SEI** — compartmental models; the latter adds a vector population.
§2.2, §2.3

**Self-loop** — an edge from a node to itself; essential when a node's own history
is the strongest predictor. §6.1

**Sensitivity index (normalised forward)** — $\frac{\partial R_0}{\partial q}\cdot
\frac{q}{R_0}$; percentage-to-percentage. Meaningless without stating what is held
fixed. §2.4

**SMAPE** — symmetric MAPE, bounded at 200%. §5.1

**Spectral abscissa** — largest real part of a matrix's eigenvalues; the growth
rate of $\mathbf{F}-\mathbf{V}$. §2.6

**Spectral radius** — largest absolute eigenvalue. §2.3

**STGAT** — Spatio-Temporal GAT. Graph attention then two LSTMs; the LSTM
collapses the node axis. §7.1

**Symmetric normalisation** — $D^{-1/2}AD^{-1/2}$; prevents degree from distorting
magnitude. §6.2

**Weight decay** — L2 penalty on parameters. §4.4

**Window** — the input span, $W$ weeks. §4.5

**z-score** — $(x-\mu)/\sigma$, with $\mu,\sigma$ from **training data only**. §4.6

---

## The five numbers to remember

| number | meaning | chapter |
|---|---|---|
| **0.92 / 0.68** | pooled / per-district lag-1 autocorrelation — why persistence is hard to beat | 3 |
| **70%** | share of the "cross validated" column that is training data | 5, 9 |
| **39%** | how much per-window averaging understates pooled RMSE | 5, 9 |
| **+0.07** | what the graph adds above a 0.55 national common mode | 3, 6 |
| **90%** | share of one fold's squared error carried by 6 of 68 windows | 3, 5 |

---

## References

### Papers reproduced

**Weng et al. (2024).** *Graph Representation Learning for Dengue Forecasting.*
IEEE BigData 2024.
Code: `MLOpenSourceOpenScience/disease_modeling_MLOS2` @ `45f1c08`.
Five GNNs on Sri Lankan district data. The primary benchmark. Reproduced in
`reproduction/`; five of Table I's twenty rows are exactly reproducible.

**GulMohamed et al. (2026).** *DengueGNN.* Scientific Reports 16:10584.
No code released; the OpenDengue subset is unspecified — **not reproducible, per
the article's own statements.** Source of the per-horizon head (Eq. 11), temporal
attention (Eq. 6–8), and probabilistic head (Eq. 12–14).

**Gopalakrishnan (2020).** *The SEIR Model of Infectious Diseases.* MTH 271.
Code printed in the source. Basic SEIR, reproduced in `crosscheck/R3`.

**Phaijoo & Gurung (2018).** *Sensitivity Analysis of SEIR–SEI Model of Dengue
Disease.* GAMS J. Math. & Math. Biosci. 6(a), 41–49.
SEIR–SEI, $R_0$, sensitivity indices. Reproduced in `crosscheck/R4`, which records
three defects (F4.2–F4.4): Table 1's baseline column contradicts its own printed
indices; the §4 parameters give $R_0 \approx 0.78 < 1$ while its figures show an
outbreak; and the printed endemic equilibrium is not a fixed point in its vector
components.

### Methods cited

- **Kipf & Welling (2017).** *Semi-Supervised Classification with Graph
  Convolutional Networks.* ICLR. — the GCN layer.
- **Veličković et al. (2018).** *Graph Attention Networks.* ICLR. — GAT.
- **Li et al. (2018).** *Diffusion Convolutional Recurrent Neural Network.* ICLR. —
  DCRNN.
- **Wu et al. (2019).** *Graph WaveNet for Deep Spatial-Temporal Graph Modeling.*
  IJCAI. — the adaptive adjacency construction used in Contribution (c).
- **Gneiting & Raftery (2007).** *Strictly Proper Scoring Rules, Prediction, and
  Estimation.* JASA. — the closed-form Gaussian CRPS.
- **van den Driessche & Watmough (2002).** *Reproduction numbers and sub-threshold
  endemic equilibria.* Math. Biosci. — the next-generation matrix method.

### Data

- **Case counts** — Sri Lanka Ministry of Health / Epidemiology Unit weekly
  reports, 2013–2022, per district.
- **Environmental covariates** — NASA EarthData (GLDAS, GPM, MODIS NDVI),
  aggregated weekly per district.
- **District boundaries** — GADM shapefiles.
- **OpenDengue** — Clarke et al., *Scientific Data* 11:296 (2024). >56M records
  across 102 countries. Reserved as an out-of-country generalisation test; not
  part of the core protocol.

### Software

```
python 3.11        torch 2.1.2              torch_geometric 2.4.0
torch_geometric_temporal 0.54.0             numpy ~1.26.2
pandas ~2.2.0      scikit-learn 1.4.0       statsmodels 0.14.1
```

The three forced deviations from the authors' pins are documented in
`reproduction/REPRODUCIBILITY_MATRIX.md` and repeated in §9.2.

---

## Project documents

| document | contents |
|---|---|
| `CLAUDE.md` | Repository conventions and non-negotiables |
| `docs/DATA.md` | The array, channel by channel, with the defects |
| `docs/ROADMAP.md` | The frozen protocol, phase criteria, risk register |
| `docs/EXPERIMENT_LOG.md` | Append-only run record, newest first |
| `docs/decisions/0001-*` | Residual, log1p, train-only normalisation |
| `docs/RECONCILIATION_WITH_PRIOR_WORK.md` | How old numbers relate to current ones |
| `crosscheck/FINDINGS.md` | F1.1–F1.6, F2.1–F2.3, F3.1, F4.1–F4.4 |
| `reproduction/REPRODUCIBILITY_MATRIX.md` | What reproduces, and the deviations |

---

## Reproducing anything in this handbook

```bash
# Dataset findings (Chapter 3)
jupyter lab analysis/notebooks/E1_dataset_eda.ipynb

# Graph experiment (Chapters 6, 9)
jupyter lab analysis/notebooks/E2_adaptive_graph.ipynb

# Independent implementation and its agreement tests (Chapter 8)
cd crosscheck && pytest

# Exact reproduction of the benchmark (Chapters 7, 9)
python reproduction/verify_local.py

# Repo hygiene
ruff check src tests tools
pytest
python tools/check_notebooks.py
```

**If a number here disagrees with what those produce, the notebook is right and
this document is stale.**

---

*Back to [00 — Start Here](00_START_HERE.md)*
