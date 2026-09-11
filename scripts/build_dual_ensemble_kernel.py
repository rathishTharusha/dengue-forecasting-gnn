"""Generate the beat-floor-dual-ensemble Kaggle kernel directory for AAGCN + ASTGCN."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_DIR = REPO / "reproduction" / "kaggle" / "kernels" / "beat-floor-combination"
DST_DIR = REPO / "reproduction" / "kaggle" / "kernels" / "beat-floor-dual-ensemble"
DST_DIR.mkdir(parents=True, exist_ok=True)

# Metadata
meta = {
    "id": "tharushaperera16/beat-floor-dual-ensemble",
    "title": "Beat floor dual champion ensemble",
    "code_file": "beat_floor_dual_ensemble.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": []
}

(DST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

# Notebook
nb = json.loads((SRC_DIR / "beat_floor_combination.ipynb").read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell.get("cell_type") == "markdown":
        src = "".join(cell.get("source", []))
        if "Beat the Floor - cross-architecture combination" in src:
            cell["source"] = [
                "# Beat the Floor - Dual Champion Ensemble (AAGCN + ASTGCN)\n",
                "\n",
                "AAGCN and ASTGCN in **one** session.\n",
                "Evaluates individual models, simple averaging (multi_raw),\n",
                "least-squares persistence blending (multi_blend_raw),\n",
                "validation-optimal simplex weights (multi_opt_raw),\n",
                "and dual-ensemble combining both models + persistence (multi_super_raw).\n"
            ]
    elif cell.get("cell_type") == "code":
        new_source = []
        for line in cell.get("source", []):
            mod = line.replace('ARCHS = ["A3TGCN", "STGAT", "ASTGCN"]', 'ARCHS = ["AAGCN", "ASTGCN"]')
            mod = mod.replace('"combo"', '"dual"')
            new_source.append(mod)
        cell["source"] = new_source

(DST_DIR / "beat_floor_dual_ensemble.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Generated {DST_DIR / 'beat_floor_dual_ensemble.ipynb'} and metadata successfully.")
