# Dengue GNN Baseline — Colab bundle

Everything needed to run the **GCN/GAT baseline** for the CS3631 project.

## Files
- `dengue_baseline_GNN.ipynb` — the notebook (run top-to-bottom in Colab).
- `sri_lanka_2013-2022_shifted.npy` — data array `(459 weeks, 25 districts, 11 features)`. Target = weekly cases at feature index 5.
- `sri_lanka_adj_list.json` — district adjacency (25 nodes) used to build the graph.

## Setup (2 minutes)
1. Create a folder in Google Drive, e.g. `MyDrive/dengue_baseline/`.
2. Upload **both** data files (`.npy` and `.json`) into that folder.
3. Open `dengue_baseline_GNN.ipynb` in Google Colab.
4. In **Cell 2**, keep `USE_DRIVE = True` and set
   `DATA_DIR = "/content/drive/MyDrive/dengue_baseline"` to match your folder.
   *(Or set `USE_DRIVE = False` to auto-download the two files from GitHub instead — no Drive needed.)*
5. **Runtime ▸ Run all.**

CPU is fine (~a few minutes). For faster tuning use Runtime ▸ Change runtime type ▸ GPU.

## What it produces (for the proposal / Phase 1)
- Per-horizon (1/2/3-week) **RMSE, MAE, SMAPE, MAPE** for the GCN baseline (§7).
- A **persistence** sanity baseline the model must beat (§7).
- A **hyperparameter-tuning table** selected on validation RMSE (§8).
- A **prediction plot** (observed vs predicted, §9).
- To get the **GAT** baseline table: set `CFG.model = 'GAT'` (and `MODEL_TO_TUNE='GAT'` in §8) and re-run §7–8.

## Notes on metrics
Dengue counts are frequently 0 in low-incidence district-weeks, so plain MAPE explodes.
We report **SMAPE** (bounded, symmetric) as the primary percentage metric plus a
zero-masked **MAPE(>=1)** for comparability. RMSE/MAE are on the original case scale.

## Design choices (so results are comparable to Weng et al.)
- Same data conventions as the reference repo `disease_modeling_MLOS2` (target column,
  window=3, horizon=3, adjacency + self-loops).
- Chronological 70/10/20 split — **no shuffling**, so no temporal leakage.
- Normalization uses **train-set statistics only** (a correctness fix vs. normalizing the whole array).
- Built on plain **PyTorch Geometric** (not `torch-geometric-temporal`, which breaks often on Colab),
  so the pipeline is stable and fully under our control — the clean seam where the
  **physics-informed loss** and **GAN augmentation** contributions attach later (see §10 of the notebook).
