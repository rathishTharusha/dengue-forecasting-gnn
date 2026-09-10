"""Generate the four cross-check notebooks as valid nbformat-4 ``.ipynb`` files.

The notebooks are generated rather than hand-edited for the same reason as
``notebooks/_build`` in the main project: a ``.ipynb`` is a JSON blob that
reviews and merges badly, while these cell sources are plain Python that diffs
line by line.

Run::

    cd crosscheck/_build
    python gen_notebooks.py

Writes into ``crosscheck/notebooks/``. Edit ``_cells_r*.py``, never the
``.ipynb`` files -- regeneration overwrites them.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def _cell_id() -> str:
    return uuid.uuid4().hex[:8]


def _source(text: str) -> list[str]:
    """Split text into nbformat's list-of-lines-with-trailing-newlines form."""
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
    """Wrap cells in nbformat-4 notebook metadata."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_nb(filename: str, cells: list[dict]) -> None:
    """Serialize one notebook to ``OUT_DIR``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    path.write_text(
        json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    import _cells_r1
    import _cells_r2
    import _cells_r3
    import _cells_r4

    write_nb("R1_weng2024_graph_representation.ipynb", _cells_r1.CELLS)
    write_nb("R2_denguegnn_dynamic_stgnn.ipynb", _cells_r2.CELLS)
    write_nb("R3_seir_basic_gopalakrishnan.ipynb", _cells_r3.CELLS)
    write_nb("R4_seir_sei_sensitivity_phaijoo.ipynb", _cells_r4.CELLS)
