"""Synthetic training data must be built from training weeks only.

``docs/AUGMENTATION_PLAN.md`` lets generated data into training sets -- a
logged deviation from plan rule R3 -- on one condition: no validation or test
week may reach it. Both generators are checked by perturbing every case value
from the first validation origin onward and requiring identical output.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "seirgnn2"))
sys.path.insert(0, str(REPO / "analysis" / "lib"))

import augment  # noqa: E402
import core  # noqa: E402
import corrected_data as cd  # noqa: E402


@pytest.fixture(scope="module")
def pair():
    torch.set_num_threads(1)
    data = cd.load()
    fold = next(f for f in core.build_folds(data.cases, data.missing) if f.origin == 0.70)
    cut = int(fold.idx["val"].min())
    tampered = data.cases.copy()
    tampered[cut:] = tampered[cut:] * 9.0 + 77.0
    return data, dataclasses.replace(data, cases=tampered), fold


def _same(a: dict, b: dict) -> None:
    for k in ("x", "y_raw", "p_raw", "pop"):
        np.testing.assert_array_equal(a[k].numpy(), b[k].numpy(), err_msg=k)


def test_seir_generator_ignores_validation_and_test(pair):
    data, tampered, fold = pair
    _same(
        augment.synth_seir(data, fold, 40, seed=3), augment.synth_seir(tampered, fold, 40, seed=3)
    )


def test_calibrated_seir_ignores_validation_and_test(pair):
    data, tampered, fold = pair
    _same(
        augment.synth_seir(data, fold, 40, seed=3, calibrated=True),
        augment.synth_seir(tampered, fold, 40, seed=3, calibrated=True),
    )


def test_timegan_ignores_validation_and_test(pair):
    data, tampered, fold = pair
    a = augment.synth_timegan(data, fold, 16, seed=3, iters=3)
    b = augment.synth_timegan(tampered, fold, 16, seed=3, iters=3)
    _same(a, b)


def test_synthetic_windows_match_the_real_layout(pair):
    data, _, fold = pair
    real = core.build_tensors(data, fold, "train", False, False, True)
    syn = augment.synth_seir(data, fold, 8, seed=0)
    for k in ("x", "y_raw", "p_raw", "y_z", "p_z", "pop"):
        assert syn[k].shape[1:] == real[k].shape[1:], k
