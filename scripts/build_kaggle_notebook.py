"""Assemble the Kaggle notebook from ``full_paper/kaggle/src/*.py`` (percent format).

Each source file is plain Python with ``# %%`` cell markers; ``# %% [markdown]``
cells hold the prose as ``#`` comment lines. The files run in name order, so the
notebook is the concatenation of their cells. Edit the sources, never the notebook.

    python scripts/build_kaggle_notebook.py            # write the notebook and the kernel folder

Outputs
-------
``full_paper/kaggle/dengue_physics_gnn.ipynb``   the notebook
``full_paper/kaggle/kernel/``                    notebook + kernel-metadata.json for ``kaggle kernels push``
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "full_paper" / "kaggle" / "src"
OUT = REPO / "full_paper" / "kaggle" / "dengue_physics_gnn.ipynb"
KERNEL = REPO / "full_paper" / "kaggle" / "kernel"
USER = "tharushaperera16"
DATASET = f"{USER}/dengue-physics-gnn-sources"


def cells_from(text: str) -> list[dict]:
    cells, kind, buf = [], None, []

    def flush():
        body = "\n".join(buf).strip("\n")
        if kind and body:
            if kind == "markdown":
                body = "\n".join(ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in body.split("\n"))
            lines = body.split("\n")
            cell = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {},
                    "source": [f"{ln}\n" for ln in lines[:-1]] + [lines[-1]]}
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)

    for line in text.split("\n"):
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def main() -> None:
    cells = [c for f in sorted(SRC.glob("*.py")) for c in cells_from(f.read_text(encoding="utf-8"))]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    KERNEL.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT, KERNEL / OUT.name)
    (KERNEL / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/dengue-physics-gnn-reproduction",
        "title": "dengue physics gnn reproduction",
        "code_file": OUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=1), encoding="utf-8")
    n_code = sum(c["cell_type"] == "code" for c in cells)
    print(f"wrote {OUT.relative_to(REPO)}: {len(cells)} cells ({n_code} code) from {len(list(SRC.glob('*.py')))} sources")


if __name__ == "__main__":
    main()
