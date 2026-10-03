"""Minimal, torch-free access to the processed dataset.

This replaces the data-loading half of ``dengue_gnn.experiment``, which was
removed with the rest of the Phase-2/3 implementation (tag ``phase23-archive``).
Only the case series survived the cleanup because only the case series is still
used outside the modelling code -- by ``tests/test_seir.py`` and
``scripts/verify_seir_paper.py``, both of which check the derived growth ceiling
against what the record actually contains.

Deliberately torch-free: CI installs numpy, pytest and ruff only, so anything
importing torch cannot be covered there. ``analysis/lib/adaptive.load_dataset``
is the richer loader used by the modelling code, and it does pull in torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["CASES_INDEX", "load_cases"]

#: Axis-2 index of weekly reported cases in the processed array. See docs/DATA.md;
#: the other ten channels are climate covariates already carrying their own lag.
CASES_INDEX = 5


def load_cases(npy_path: str | Path) -> np.ndarray:
    """Weekly reported cases as ``(weeks, districts)``.

    Args:
        npy_path: Path to ``sri_lanka_2013-2022_shifted.npy``, shape
            ``(459, 25, 11)``.

    Returns:
        Float64 case counts with NaNs zeroed, districts in the order the array
        stores them -- which is ``sorted()`` of the adjacency JSON's keys.

    Raises:
        ValueError: If the array does not have the documented three axes, which
            would mean the channel index is being applied to something else.
    """
    raw = np.load(Path(npy_path), allow_pickle=True)
    if raw.ndim != 3:
        raise ValueError(f"expected (weeks, districts, features), got {raw.shape}")
    return np.nan_to_num(raw).astype(np.float64)[:, :, CASES_INDEX]
