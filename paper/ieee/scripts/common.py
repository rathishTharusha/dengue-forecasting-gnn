"""Shared paths and statistics for the IEEE paper scripts.

Everything here reads saved result files only. Nothing trains a model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IEEE = HERE.parent
REPO = IEEE.parent.parent
FIG = IEEE / "figures"
TAB = IEEE / "tables"
RES = IEEE / "results"

P9_FILE = REPO / "seirgnn2" / "results" / "frozen9_plus_audit.json"
KAGGLE = REPO / "full_paper" / "outputs" / "kaggle_run"
IMPROVED = REPO / "analysis" / "results" / "improved_sweep.json"
LEGACY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
REBUILT = REPO / "data" / "corrected" / "rebuilt_cases.npy"
REBUILT_INDEX = REPO / "data" / "corrected" / "rebuilt_index.csv"

# The paired tests are the project's own (exact sign-flip, Benjamini-Hochberg).
sys.path.insert(0, str(REPO / "seirgnn2"))
import stats as project_stats  # noqa: E402

# Okabe-Ito colour-blind-safe palette.
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def write_tex(name: str, text: str) -> None:
    TAB.mkdir(exist_ok=True)
    (TAB / name).write_text(text, encoding="utf-8")
