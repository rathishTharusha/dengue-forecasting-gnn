"""Generate the analysis notebooks from their cell sources.

Same convention as ``notebooks/_build``, ``crosscheck/_build`` and
``reproduction/_build``: ``.ipynb`` files are generated artifacts, the cell
sources are plain Python that reviews and diffs line by line.

Run::

    cd analysis/_build
    python gen_analysis.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def _source(text: str) -> list[str]:
    text = text.strip("\n")
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(text: str) -> dict:
    """A markdown cell."""
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": _source(text),
    }


def code(text: str) -> dict:
    """A code cell with no stored output."""
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def write_nb(filename: str, cells: list[dict]) -> None:
    """Serialize one notebook into ``analysis/notebooks``."""
    payload = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    import cells_adaptive
    import cells_eda

    write_nb("E1_dataset_eda.ipynb", cells_eda.CELLS)
    write_nb("E2_adaptive_graph.ipynb", cells_adaptive.CELLS)
