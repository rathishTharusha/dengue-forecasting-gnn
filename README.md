# Physics-Informed, GAN-Augmented Graph Neural Networks for Multi-Horizon Dengue Forecasting

District-level weekly dengue incidence forecasting for Sri Lanka (25 districts, 2013–2022),
built on a spatio-temporal Graph Neural Network baseline and extended with three stackable
contributions: an **SEIR–SEI physics-informed loss**, **GAN-based data augmentation**, and a
**learned adaptive graph**.

CS3631 group project — Group 05, Department of Computer Science and Engineering,
University of Moratuwa.

---

## Status

| Phase | Deliverable | State |
|---|---|---|
| 0 | Literature review (3 pillars + gap analysis) | ✅ Done |
| 0 | Project proposal | ✅ Submitted — [`Group05_Proposal.pdf`](Group05_Proposal.pdf) |
| 1 | GCN/GAT baseline, rolling-origin CV | ✅ Done — matches persistence floor |
| 2 | Adaptive graph (Graph WaveNet-style adjacency) | ☐ Not started |
| 3 | Physics-informed SEIR–SEI loss | ☐ Not started |
| 4 | GAN augmentation (TimeGAN / RCGAN) | ☐ Not started |
| 5 | Full stacked ablation + write-up | ☐ Not started |

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the detailed plan, owners, and risk register.

### Baseline results (rolling-origin CV, mean over 3 folds × 3 seeds)

| Model | RMSE | MAE |
|---|---|---|
| Persistence (naive reference) | 44.8 | 15.7 |
| GCN (residual + log1p) | 45.3 | 15.9 |
| GAT (residual + log1p) | 45.5 | 15.9 |

The baseline **matches** the persistence floor rather than beating it — that is the honest
starting point, and it is deliberate: it leaves clear, measurable room for the three
contributions to push below it. See
[`notebooks/baseline/RESULTS_v1_findings_and_v2_fixes.md`](notebooks/baseline/RESULTS_v1_findings_and_v2_fixes.md)
for why, and [`docs/decisions/0001-baseline-training-refinements.md`](docs/decisions/0001-baseline-training-refinements.md)
for the decisions behind it.

---

## Repository layout

```
.
├── Group05_Proposal.pdf              # submitted proposal (source of truth for scope)
├── Literature Review - GNN ....md    # literature review, 5 areas + gap analysis
├── PROJECT_PROPOSAL_GUIDE.md         # working guide: baseline → proposal mapping
├── notebooks/
│   ├── _build/                       # generator scripts for notebooks 00–02
│   ├── 00_data_setup_eda.ipynb       # data location, EDA, data_manifest.json
│   ├── 01_baselines_classical.ipynb  # ARIMAX, RF, XGBoost, ARNN, LSTM
│   ├── 02_baselines_gnn.ipynb        # STGAT, A3TGCN, ASTGCN, DCRNN, AAGCN
│   └── baseline/                     # ← the Phase-1 deliverable
│       ├── dengue_baseline_GNN_v2.ipynb        # RUN THIS ONE
│       ├── dengue_baseline_GNN.ipynb           # v1, kept for provenance
│       ├── sri_lanka_2013-2022_shifted.npy     # (459, 25, 11) processed array
│       ├── sri_lanka_adj_list.json             # 25-node district adjacency
│       └── RESULTS_v1_findings_and_v2_fixes.md
├── src/dengue_gnn/                   # shared library code extracted from notebooks
├── tests/                            # unit tests for src/
├── tools/                            # repo maintenance scripts
├── results/                          # committed metric tables & figures
└── docs/                             # setup, data, roadmap, experiment log, ADRs
```

**Local-only (git-ignored, see [`.gitignore`](.gitignore)):** `papers/` (copyrighted PDFs),
`reference_repo/` (third-party clone), `datasets/` (3.5 MB raw dump), the course handout.
These stay in the shared Drive folder — they are reference material, not project output.

---

## Quick start

```bash
git clone https://github.com/<org-or-user>/dengue-forecasting-gnn.git
cd dengue-forecasting-gnn
```

Then follow [`docs/SETUP.md`](docs/SETUP.md). Short version — the baseline notebook runs on
free Colab with no local install:

1. Open `notebooks/baseline/dengue_baseline_GNN_v2.ipynb` in Google Colab.
2. Set `USE_DRIVE = False` in §2 to pull the two data files straight from this repo.
3. **Runtime ▸ Run all.** CPU is fine (~minutes); GPU is faster for the CV sweep.

For local work:

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -r requirements.txt
```

---

## Data

`sri_lanka_2013-2022_shifted.npy` — shape `(459 weeks, 25 districts, 11 features)`, target =
weekly cases at feature index **5**. Environmental covariates are already lag-shifted by their
empirically optimal delay (precipitation ≈ 12 weeks, min NDVI ≈ 17 weeks). Full column
documentation, provenance, and licensing in [`docs/DATA.md`](docs/DATA.md).

The target is heavy-tailed (median 13, max 2631, 9.7% zeros) with lag-1 autocorrelation ≈ 0.68.
That single fact drives most of the modelling decisions in this repo.

---

## Working on this project

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before your first commit. The three things that
matter most:

1. **Branch per piece of work** — `feat/adaptive-adjacency`, `exp/physics-loss-lambda-sweep`.
   Never commit directly to `main`.
2. **Log every real experiment** in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) with the
   config that produced it. Numbers without a reproducible config do not go in the report.
3. **Keep the evaluation protocol identical** across baseline and contributions — same
   rolling-origin folds, same seeds, same metrics — or the ablation table is meaningless.

> ⚠️ **This repo lives inside a Google Drive folder.** Drive's sync client can corrupt `.git`
> if two machines write at once. Pause Drive sync while running `git` operations, or clone the
> GitHub repo to a local (non-Drive) path and work there. See [`docs/SETUP.md`](docs/SETUP.md#google-drive-warning).

---

## Team — Group 05

Tharusha Perera · Praveen De Silva · Janith Mahanama · Bimsara Udurawana ·
Maleesha Kumarasinghe · Praveen Nawarathna

Department of Computer Science and Engineering, University of Moratuwa.

## Key references

- Weng et al. (2024), *Graph Representation Learning for Dengue Forecasting*, IEEE BigData — the benchmark we reproduce.
- Rodríguez et al. (2023), *EINNs: Epidemiologically-Informed Neural Networks*, AAAI — physics-loss line.
- Yoon et al. (2019), *Time-series Generative Adversarial Networks*, NeurIPS — augmentation.
- Wu et al. (2019), *Graph WaveNet*, IJCAI — adaptive adjacency.

Full list in `Literature Review - GNN for Dengue Forecasting.md` and the proposal's references.

## License

Code is released under the [MIT License](LICENSE). Data and third-party papers are **not**
covered by it — see [`docs/DATA.md`](docs/DATA.md) for their terms.
