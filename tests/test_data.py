"""Unit tests for dataset loading, rolling-origin CV splits, and tensor generation."""

import os
import numpy as np
import pytest
import torch

from dengue_gnn.data import (
    build_fold_tensors,
    get_persistence_forecast,
    get_rolling_origin_splits,
    load_adjacency,
    load_dataset,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPY_PATH = os.path.join(BASE_DIR, "notebooks", "baseline", "sri_lanka_2013-2022_shifted.npy")
ADJ_PATH = os.path.join(BASE_DIR, "notebooks", "baseline", "sri_lanka_adj_list.json")


def test_load_dataset():
    assert os.path.exists(NPY_PATH), f"Missing {NPY_PATH}"
    assert os.path.exists(ADJ_PATH), f"Missing {ADJ_PATH}"

    raw, edge_index, districts = load_dataset(NPY_PATH, ADJ_PATH, n_nodes=25, self_loops=True)
    assert raw.shape == (459, 25, 11)
    assert len(districts) == 25
    assert edge_index.ndim == 2
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0  # has directed edges including self loops


def test_rolling_origin_splits():
    splits = get_rolling_origin_splits(
        total_timesteps=459,
        window=3,
        horizon=3,
        origins=(0.55, 0.70, 0.85),
        test_frac=0.15,
        val_weeks=30,
    )
    assert len(splits) == 3

    for s in splits:
        train_ids = s["train_ids"]
        val_ids = s["val_ids"]
        test_ids = s["test_ids"]

        # Strictly chronological order
        assert max(train_ids) < min(val_ids)
        assert max(val_ids) < min(test_ids)

        # Zero index leakage / disjointness
        assert len(set(train_ids).intersection(set(val_ids))) == 0
        assert len(set(train_ids).intersection(set(test_ids))) == 0
        assert len(set(val_ids).intersection(set(test_ids))) == 0

        assert len(train_ids) > 0
        assert len(val_ids) == 30
        assert len(test_ids) > 0


def test_build_fold_tensors_and_invertibility():
    raw = np.zeros((100, 25, 11), dtype=np.float32)
    # Fill cases column (index 5) with positive random counts
    np.random.seed(42)
    raw[..., 5] = np.random.poisson(lam=20.0, size=(100, 25)).astype(np.float32)

    train_ids = list(range(3, 70))
    eval_ids = list(range(70, 95))

    fold_data, inv_fn = build_fold_tensors(
        raw=raw,
        train_ids=train_ids,
        eval_ids=eval_ids,
        window=3,
        horizon=3,
        cases_idx=5,
        use_all_feats=True,
        log_transform=True,
    )

    assert fold_data.X.shape == (len(eval_ids), 25, 33)
    assert fold_data.Y.shape == (len(eval_ids), 25, 3)
    assert fold_data.P.shape == (len(eval_ids), 25, 3)

    # Check invertibility: mapping Y back via inv_fn should match raw cases
    y_raw_reconstructed = inv_fn(fold_data.Y)
    y_raw_expected = np.stack([raw[i : i + 3, :, 5].T for i in eval_ids])
    np.testing.assert_allclose(y_raw_reconstructed, y_raw_expected, rtol=1e-4, atol=1e-4)


def test_get_persistence_forecast():
    raw = np.zeros((50, 25, 11), dtype=np.float32)
    raw[..., 5] = np.arange(50)[:, None]  # linear progression

    eval_ids = [10, 20, 30]
    preds, truths = get_persistence_forecast(raw, eval_ids, horizon=3, cases_idx=5)

    assert preds.shape == (3, 25, 3)
    assert truths.shape == (3, 25, 3)

    # For eval week 10, persistence should repeat week 9
    assert np.all(preds[0] == 9.0)
    # And truths should be weeks 10, 11, 12
    assert np.all(truths[0, :, 0] == 10.0)
    assert np.all(truths[0, :, 1] == 11.0)
    assert np.all(truths[0, :, 2] == 12.0)
