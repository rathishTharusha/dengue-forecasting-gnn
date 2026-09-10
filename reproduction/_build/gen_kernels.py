"""Generate the Kaggle reproduction notebooks and their kernel metadata.

Same discipline as ``notebooks/_build`` and ``crosscheck/_build``: the ``.ipynb``
files are generated, the cell sources are plain Python that reviews and diffs.

The Weng et al. reproduction is emitted once per architecture, because Kaggle
caps a session at 12 hours and running all five in one kernel risks losing the
whole run. Each notebook calls the authors' own ``run_<model>`` function with the
authors' own segment list -- exactly what their ``if __name__ == "__main__"``
block does.

Run::

    cd reproduction/_build
    python gen_kernels.py

Writes notebooks into ``reproduction/kaggle/kernels/<slug>/`` alongside a
``kernel-metadata.json``, and a copy into ``reproduction/notebooks/`` for
reading in the repo.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = ROOT / "kaggle" / "kernels"
NOTEBOOKS_DIR = ROOT / "notebooks"

#: Placeholder until `setup_kaggle.py --init-metadata` writes the real username.
USERNAME_PLACEHOLDER = "YOUR-KAGGLE-USERNAME"


def slugify(title: str) -> str:
    """Kaggle's slug rules: lowercase, non-alphanumerics to hyphens, collapsed.

    Kaggle rejects a push whose ``id`` slug does not match the slug its ``title``
    resolves to -- it answers 400 "Your kernel title does not resolve to the
    specified id", or silently files the kernel under the title's slug instead.
    Deriving the slug from the title makes the two agree by construction.

    Titles are kept to letters, digits and spaces upstream so this stays
    faithful to Kaggle's own transformation.
    """
    out = "".join(c.lower() if c.isalnum() else "-" for c in title)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def _cell_id() -> str:
    return uuid.uuid4().hex[:8]


def _source(text: str) -> list[str]:
    text = text.strip("\n")
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(text: str) -> dict:
    """A markdown cell."""
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {}, "source": _source(text)}


def code(text: str) -> dict:
    """A code cell with no stored output."""
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def notebook(cells: list[dict]) -> dict:
    """Wrap cells in nbformat-4 metadata."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def kernel_metadata(slug: str, title: str, notebook_name: str) -> dict:
    """Kaggle kernel settings.

    ``enable_gpu`` is False deliberately: the paper states the models were
    trained on a CPU (§II.D). ``enable_internet`` is required because the
    notebook clones the authors' repository and installs pinned wheels.
    """
    return {
        "id": f"{USERNAME_PLACEHOLDER}/{slug}",
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }


def write_kernel(title: str, cells: list[dict]) -> None:
    """Emit one kernel directory plus a readable copy under notebooks/.

    The directory name, the notebook filename and the Kaggle id all derive from
    the title's slug, so they cannot drift apart.
    """
    slug = slugify(title)
    name = f"{slug.replace('-', '_')}.ipynb"
    payload = json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n"

    kernel_dir = KERNELS_DIR / slug
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / name).write_text(payload, encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata(slug, title, name), indent=2) + "\n", encoding="utf-8"
    )

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    (NOTEBOOKS_DIR / name).write_text(payload, encoding="utf-8")
    print(f"wrote {kernel_dir}/{name} ({len(cells)} cells) + kernel-metadata.json")


if __name__ == "__main__":
    import cells_graph
    import cells_physics_sweep
    import cells_seir
    import cells_sweep
    import cells_weng

    # Titles use only letters, digits and spaces so the slug is predictable.
    # The probe validates the environment cheaply; push it before the model runs.
    write_kernel("Weng 2024 reproduction environment probe", cells_weng.build_probe())

    for model in cells_weng.MODELS:
        write_kernel(f"Weng 2024 exact reproduction {model}", cells_weng.build(model))

    write_kernel("SEIR model reproduction Gopalakrishnan", cells_seir.basic())
    write_kernel("SEIR SEI sensitivity reproduction Phaijoo Gurung", cells_seir.sei_sensitivity())

    # Not a reproduction: the verified architectures under this project's own
    # protocol, plus the increments the papers and our EDA justify. One kernel
    # per architecture -- DCRNN alone runs ~10 h against Kaggle's 12 h CPU limit.
    for model in cells_sweep.ARCHITECTURES:
        write_kernel(f"Improved architecture sweep {model}", cells_sweep.build(model))

    # One encoder, four graph modes: the only place the adjacency is a free
    # variable with everything else held identical.
    write_kernel("Graph mode sweep dengue GNN", cells_graph.build())

    # Physics-informed loss sweep: relaxed biological envelope and spatial regularizers
    write_kernel("Physics informed sweep dengue GNN", cells_physics_sweep.build())
