# Setup

Two supported paths: **Colab** (what we actually use for results) and **local**
(for editing `src/`, running tests, and quick iteration).

---

## Google Drive warning

This working folder is inside Google Drive. A `.git` directory inside a Drive-synced folder is
a known corruption risk: Drive can partially sync loose objects and index files while git is
mid-write, and two teammates syncing the same folder will fight over `.git/index`.

**Recommended:** clone the GitHub repo to a normal local path and work there.

```bash
git clone https://github.com/<org-or-user>/dengue-forecasting-gnn.git C:\dev\dengue-gnn
```

Keep the Drive folder for the git-ignored reference material (`papers/`, `datasets/`,
`reference_repo/`). If you must run git inside the Drive folder, pause Drive sync first
(system tray ▸ Drive ▸ Pause syncing) and resume when the git command finishes.

---

## Path A — Google Colab (recommended for experiments)

No local install. Free-tier GPU is sufficient; the graph is 25 nodes / 141 edges and a fold
trains in minutes.

1. Open `notebooks/baseline/dengue_baseline_GNN_v2.ipynb` in Colab
   (GitHub ▸ paste the repo URL, or upload the `.ipynb`).
2. In **§2 Get the data**, set `USE_DRIVE = False` to download
   `sri_lanka_2013-2022_shifted.npy` and `sri_lanka_adj_list.json` directly from this repo.
   *(Or set `USE_DRIVE = True` and point `DATA_DIR` at a Drive folder holding both files.)*
3. **Runtime ▸ Change runtime type ▸ T4 GPU** (optional but faster for §8's CV sweep).
4. **Runtime ▸ Run all.**

### Notebooks 00–02 (Weng et al. reproduction)

These need the teammate-shared Drive folder containing `MLSO2_Final.csv`. Before running
`00_data_setup_eda.ipynb`, open that shared folder once and click **"Add shortcut to Drive"** —
Colab cannot mount someone else's Drive directly. Run them in order: `00` → `01` → `02`; each
writes files the next one reads.

Every notebook has a `QUICK_TEST` flag. Run `QUICK_TEST = True` first (a few districts, one
segment, few epochs) to confirm the pipeline executes, then `QUICK_TEST = False` for numbers
you intend to cite.

---

## Path B — Local

Python **3.11**.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
```

Install PyTorch first — the right wheel depends on your hardware. CPU-only is fine for
`src/` development and tests:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA, pick the matching index URL from <https://pytorch.org/get-started/locally/>.

Then PyTorch Geometric and the rest:

```bash
pip install torch-geometric
pip install -r requirements.txt
```

> We deliberately use plain **PyTorch Geometric**, not `torch-geometric-temporal` — the latter
> breaks often on Colab and pins old torch versions. Notebooks 00–02 (the Weng et al.
> reproduction) are the exception; they use the reference architectures as published.

Verify:

```bash
python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)"
pytest -q
ruff check src tests tools
```

---

## Regenerating notebooks 00–02

Notebooks `00`–`02` are generated from Python cell sources, so they can be reviewed as diffable
text:

```bash
cd notebooks/_build
python gen_notebooks.py
```

Edit `_cells_00.py` / `_cells_01.py` / `_cells_02.py`, not the `.ipynb` files.
`notebooks/baseline/*.ipynb` are hand-maintained and are edited directly.

---

## Data files

The two files needed by the baseline notebook are tracked in this repo under
`notebooks/baseline/`. The larger raw sources are git-ignored — see [`DATA.md`](DATA.md) for
where they come from and how to regenerate the processed array.
