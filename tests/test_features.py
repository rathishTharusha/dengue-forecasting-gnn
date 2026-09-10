"""Tests for multivariate inputs and the missing-data repair.

The covariate arm is only worth running if the inputs are not poisoned. These
guard the two ways this array can poison them: a 0 K temperature left in place,
and a normalisation that peeks past the training split.

Pure numpy except for the fold builder, so most of this runs in CI.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "features", REPO / "analysis" / "lib" / "features.py"
)
features = importlib.util.module_from_spec(_SPEC)
sys.modules["features"] = features
_SPEC.loader.exec_module(features)

NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
HAS_DATA = NPY.exists() and ADJ.exists()


def test_resolve_set_puts_cases_first_and_rejects_its_absence():
    assert features.resolve_set("causal")[0] == features.CASES_INDEX
    assert features.resolve_set((7, 5, 3))[0] == features.CASES_INDEX
    with pytest.raises(ValueError, match="cases channel"):
        features.resolve_set((0, 1, 2))
    with pytest.raises(ValueError, match="unknown feature set"):
        features.resolve_set("nonsense")


def test_scattered_gaps_are_interpolated_along_time():
    raw = np.ones((10, 2, 11)) * 300.0
    raw[4:6, 0, 0] = 0.0
    filled, missing = features.impute_missing(raw)
    assert np.allclose(filled[:, 0, 0], 300.0), "a flat series must interpolate flat"
    assert missing[4:6, 0, 0].tolist() == [1.0, 1.0]
    assert missing[:4, 0, 0].sum() == 0.0


def test_a_fully_missing_district_is_filled_from_its_neighbours():
    """Interpolation along time cannot reach a district with no data at all."""
    raw = np.ones((10, 3, 11)) * 300.0
    raw[:, 0, 0] = 0.0                       # district 0 entirely absent
    raw[:, 1, 0] = 280.0
    raw[:, 2, 0] = 320.0
    filled, missing = features.impute_missing(raw, neighbours={0: [1], 1: [0], 2: []})
    assert np.allclose(filled[:, 0, 0], 280.0), "should take the neighbour's value"
    assert missing[:, 0, 0].sum() == 10.0
    # Without a neighbour map there is nothing to fill from, but the indicator
    # must still mark every week so the failure is visible rather than silent.
    _, missing_nomap = features.impute_missing(raw)
    assert missing_nomap[:, 0, 0].sum() == 10.0


def test_precipitation_zeros_are_left_alone():
    """Zero rainfall is an observation, not a gap."""
    raw = np.ones((10, 2, 11))
    raw[:, :, 7] = 0.0
    filled, missing = features.impute_missing(raw)
    assert np.all(filled[:, :, 7] == 0.0)
    assert missing[:, :, 7].sum() == 0.0


@pytest.mark.skipif(not HAS_DATA, reason="processed array not present")
def test_no_impossible_values_survive_on_the_real_array():
    """The whole point: no 0 K temperatures reach the model."""
    _, stack, _, names = features.load_multivariate(NPY, ADJ, "climate")
    for i, name in enumerate(names):
        if name.endswith("_missing") or name in ("cases",):
            continue
        if name.startswith(("meanTair", "minTair", "maxTair")):
            assert stack[..., i].min() > 250.0, f"{name} still holds a non-physical value"
        if name in ("meanQair", "meanSoilmoi"):
            assert stack[..., i].min() > 0.0, f"{name} still holds a zero"


@pytest.mark.skipif(not HAS_DATA, reason="processed array not present")
def test_jaffna_is_the_missing_district():
    """Pins the finding, so a re-derived array that fixes it shows up as a failure."""
    raw = np.nan_to_num(np.load(NPY, allow_pickle=True))
    names = sorted(json.loads(ADJ.read_text(encoding="utf-8")))
    jaffna = names.index("Jaffna")
    for c in (0, 1, 2, 3, 4):
        assert (raw[:, jaffna, c] == 0.0).all(), f"channel {c} was expected fully missing"


@pytest.mark.skipif(not HAS_DATA, reason="processed array not present")
def test_fold_normalisation_uses_training_weeks_only():
    """A covariate arm that leaks is worse than no covariate arm."""
    pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "analysis" / "lib"))
    cases, stack, _, _ = features.load_multivariate(NPY, ADJ, "causal")
    folds = features.build_folds_mv(cases, stack, window=3, horizon=3, origins=(0.7,))
    fold = folds[0]

    n_feat = stack.shape[-1]
    assert fold.x_train.shape[-1] == n_feat * 3, "inputs pack feature-major over the window"
    assert fold.x_train.shape[1] == 25

    # Splits stay ordered and disjoint -- this is a forecasting task.
    assert fold.train_index.max() < fold.val_index.min()
    assert fold.val_index.max() < fold.test_index.min()

    # Rebuilding with the test tail perturbed must not move the training inputs.
    poisoned = stack.copy()
    poisoned[fold.test_index.min() :] *= 7.0
    again = features.build_folds_mv(cases, poisoned, window=3, horizon=3, origins=(0.7,))[0]
    assert np.allclose(fold.x_train.numpy(), again.x_train.numpy()), (
        "training inputs changed when only test-period values moved -- normalisation leaks"
    )
