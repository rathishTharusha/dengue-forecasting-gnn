"""
Generates the three baseline notebooks as valid .ipynb (nbformat 4) files.
Run: python gen_notebooks.py
Writes into ../ (i.e. DNN/notebooks/).
"""
import json
import os
import uuid

OUT_DIR = os.path.join(os.path.dirname(__file__), "..")


def _cell_id():
    return uuid.uuid4().hex[:8]


def md(text):
    text = text.strip("\n")
    lines = text.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {}, "source": source}


def code(text):
    text = text.strip("\n")
    lines = text.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_nb(filename, cells):
    nb = notebook(cells)
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    import _cells_00
    import _cells_01
    import _cells_02
    import _cells_04

    write_nb("00_data_setup_eda.ipynb", _cells_00.CELLS)
    write_nb("01_baselines_classical.ipynb", _cells_01.CELLS)
    write_nb("02_baselines_gnn.ipynb", _cells_02.CELLS)
    write_nb("04_kaggle_search.ipynb", _cells_04.CELLS)
