"""EXP-063 (docs/PROSPECTIVE_PLAN.md): the purge, the final split, the new-week
loader, the classical baselines and the block bootstrap.

Nothing here reads ``data/new_weeks``: the loader is exercised on a synthetic
directory built from development rows, so the tests can pass before the new
weeks are parsed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "seirgnn2"))

import classical  # noqa: E402

H = 3


def _series(t: int = 120, n: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).poisson(30, size=(t, n)).astype(float)


def _targets(idx) -> set[int]:
    return {int(i) + h for i in idx for h in range(H)}


# ---- classical baselines (no torch) ---------------------------------------------------------
def test_seasonal_naive_reads_the_row_52_weeks_before_each_target():
    y = _series()
    idx = np.array([60, 70, 100])
    pred, fallback = classical.seasonal_naive(y, idx)
    for k, i in enumerate(idx):
        for h in range(H):
            assert np.array_equal(pred[k, :, h], y[i + h - 52])
    assert fallback == 0


def test_seasonal_naive_falls_back_to_persistence_on_a_missing_row():
    y = _series()
    y[60 - 52 + 1, 2] = np.nan
    pred, fallback = classical.seasonal_naive(y, np.array([60]))
    assert fallback == 1
    assert pred[0, 2, 1] == y[59, 2]


@pytest.mark.parametrize("fn", ["persistence", "seasonal", "ar3"])
def test_baselines_never_read_the_origin_or_later(fn):
    y = _series()
    coef = classical.ar3_fit(y, np.arange(3, 50), 1.0)
    run = {
        "persistence": lambda a, i: classical.persistence(a, i),
        "seasonal": lambda a, i: classical.seasonal_naive(a, i)[0],
        "ar3": lambda a, i: classical.ar3_forecast(a, i, coef),
    }[fn]
    i = np.array([80])
    before = run(y, i)
    z = y.copy()
    z[80:] = 1e6
    assert np.array_equal(run(z, i), before)


def test_ar3_fit_reads_only_training_rows():
    y = _series()
    tr = np.arange(3, 50)
    before = classical.ar3_fit(y, tr, 1.0)
    z = y.copy()
    z[50:] = 1e6
    assert np.allclose(classical.ar3_fit(z, tr, 1.0), before)


def test_ar3_forecast_is_recursive():
    y = _series()
    n, i = y.shape[1], np.array([80])
    shift = np.r_[0.0, 1.0, 0.0, np.zeros(n)]  # predict lag 2
    pred = classical.ar3_forecast(y, i, shift)[0]
    # step 1 = y[i-2]; step 2's lag 2 is the old lag 1 = y[i-1]; step 3's is step 1
    assert np.allclose(pred[:, 0], y[78])
    assert np.allclose(pred[:, 1], y[79])
    assert np.allclose(pred[:, 2], y[78])


def test_ar3_penalty_spares_the_district_intercepts():
    y = _series(t=300)
    coef = classical.ar3_fit(y, np.arange(3, 290), 1e9)
    assert np.allclose(coef[:3], 0, atol=1e-6)
    assert np.allclose(coef[3:], np.log1p(y[3:290]).mean(0), atol=1e-6)


def test_bootstrap_of_identical_arms_is_null():
    loss = np.random.default_rng(1).gamma(2, 100, 60)
    r = classical.block_bootstrap(loss, loss)
    assert r["delta"] == 0 and r["lo"] == 0 and r["hi"] == 0 and r["p"] == 1.0


def test_bootstrap_detects_a_uniform_improvement_and_is_reproducible():
    rng = np.random.default_rng(2)
    b = rng.gamma(2, 100, 80)
    a = 0.5 * b
    r1, r2 = classical.block_bootstrap(a, b), classical.block_bootstrap(a, b)
    assert r1 == r2
    assert r1["delta"] < 0 and r1["hi"] < 0 and r1["p"] < 0.01


def test_bootstrap_resamples_whole_starts_in_contiguous_blocks():
    # Each start's loss is its index, so a replicate's mean exposes which starts it drew.
    n, block = 20, 8
    rng = np.random.default_rng(0)
    starts = rng.integers(0, n, size=(1, -(-n // block)))
    pick = ((starts[:, :, None] + np.arange(block)) % n).reshape(1, -1)[:, :n][0]
    for k in range(0, n, block):
        run = pick[k : k + block]
        assert all((run[j + 1] - run[j]) % n == 1 for j in range(len(run) - 1))


def test_holm():
    assert classical.holm([0.01, 0.04]) == [0.02, 0.04]
    assert classical.holm([0.04, 0.01]) == [0.04, 0.02]
    assert classical.holm([0.03, 0.02]) == [0.04, 0.04]


# ---- splits (core imports torch) ------------------------------------------------------------
def test_purged_nine_origins_share_no_target_week():
    pytest.importorskip("torch")
    import core
    import corrected_data as cd

    data = cd.load()
    args = (data.cases, data.missing, core.WINDOW, core.ORIGINS_F9, core.TEST_FRAC_F9)
    for f in core.build_folds(*args, purge=core.HORIZON - 1):
        tr, va, te = (_targets(f.idx[s]) for s in ("train", "val", "test"))
        assert not (tr & va) and not (va & te) and not (tr & te)
    # the unpurged protocol does overlap -- the defect the purge removes
    f0 = core.build_folds(*args)[0]
    assert _targets(f0.idx["train"]) & _targets(f0.idx["val"])


def test_purge_leaves_test_windows_and_the_persistence_check_value_unchanged():
    pytest.importorskip("torch")
    import core
    import corrected_data as cd

    data = cd.load()
    args = (data.cases, data.missing, core.WINDOW, core.ORIGINS_F9, core.TEST_FRAC_F9)
    rm = []
    for a, b in zip(core.build_folds(*args), core.build_folds(*args, purge=2), strict=True):
        assert np.array_equal(a.idx["test"], b.idx["test"])
        assert (a.mean, a.std) == (b.mean, b.std)
        te = b.idx["test"]
        rm.append(
            core.rmse(
                classical.persistence(data.cases, te),
                np.stack([data.cases[te + h] for h in range(H)], -1),
            )
        )
    assert round(float(np.mean(rm)), 4) == 28.5410


def test_final_fold_boundaries():
    pytest.importorskip("torch")
    import core

    y = _series(t=600, n=3)
    miss = np.zeros(600, bool)
    f = core.build_final_fold(y, miss, last_dev=558)
    tr, va, te = f.idx["train"], f.idx["val"], f.idx["test"]
    assert (va.min(), va.max(), len(va)) == (527, 556, 30)
    assert tr.max() == 524 and te.min() == 559 and te.max() == 597
    assert max(_targets(va)) == 558 and min(_targets(te)) == 559
    assert not (_targets(tr) & _targets(va))
    z = y.copy()
    z[525:] = 1e6
    g = core.build_final_fold(z, miss, last_dev=558)
    assert (g.mean, g.std) == (f.mean, f.std)


# ---- the new-week loader --------------------------------------------------------------------
def _fake_new_weeks(tmp_path: Path, n_new: int = 30, missing_row: int = 5) -> Path:
    import csv

    import corrected_data as cd

    wer = cd.WER_DISTRICTS
    cases = np.array([[1000.0 * j + r for j in range(len(wer))] for r in range(n_new)])
    cases[missing_row] = np.nan
    np.save(tmp_path / "cases.npy", cases[..., None].astype(np.float32))
    start = np.datetime64("2024-03-02")
    with (tmp_path / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["row", "status", "week_start"])
        w.writeheader()
        for r in range(n_new):
            gone = r == missing_row
            w.writerow(
                {
                    "row": r,
                    "status": "missing" if gone else "observed",
                    "week_start": "" if gone else str(start + 7 * r),
                }
            )
    return tmp_path


def test_loader_maps_report_columns_to_graph_districts(tmp_path):
    pytest.importorskip("pandas")
    import corrected_data as cd

    data, last_dev = cd.load_with_new_weeks(_fake_new_weeks(tmp_path))
    dev = cd.load()
    assert last_dev == dev.cases.shape[0] - 1
    assert np.array_equal(data.cases[: last_dev + 1], dev.cases, equal_nan=True)
    for j, name in enumerate(cd.WER_DISTRICTS):
        col = data.cases[last_dev + 1 :, data.names.index(name)]
        ok = ~np.isnan(col)
        assert np.array_equal(col[ok], 1000.0 * j + np.arange(30)[ok])
    assert data.missing[last_dev + 1 + 5] and data.missing.sum() == dev.missing.sum() + 1
    assert data.week_start.iloc[last_dev + 1 + 5] == data.week_start.iloc[
        last_dev + 5
    ] + np.timedelta64(7, "D")
    assert data.population.shape == data.cases.shape
    assert np.isfinite(data.population[last_dev + 1 :]).all()


def test_new_week_input_path_has_no_future_leakage(tmp_path):
    pytest.importorskip("torch")
    import core
    import corrected_data as cd

    data, last_dev = cd.load_with_new_weeks(_fake_new_weeks(tmp_path))
    fold = core.build_final_fold(data.cases, data.missing, last_dev)
    i = int(fold.idx["test"][3])
    one = core.Fold(1.0, {**fold.idx, "test": np.array([i])}, fold.mean, fold.std, fold.window)
    before = core.build_tensors(data, one, "test", False, False, False)
    data.cases[i:] = 1e6
    data.population[i:] = 1.0
    after = core.build_tensors(data, one, "test", False, False, False)
    for k in ("x", "p_z", "pop"):
        assert np.array_equal(before[k].numpy(), after[k].numpy()), k


def test_final_run_smoke(tmp_path):
    """The whole step-5 pipeline on synthetic new weeks, one epoch per run."""
    pytest.importorskip("torch")
    import prospective

    frozen = {"arms": {a: {"lr": 3e-3, "hidden": 32} for a in prospective.ARMS}, "ar_alpha": 1.0}
    prospective.OUT = tmp_path / "out"
    res = prospective.run_final(
        frozen, _fake_new_weeks(tmp_path), workers=1, epochs=1, out_name="smoke"
    )
    assert set(res["losses"]) == {*prospective.ARMS, "Persistence", "Seasonal naive", "AR(3) ridge"}
    assert all(len(v) == res["n_test"] for v in res["losses"].values())
    st = prospective.stats(res)
    assert [p["test"] for p in st["primary"]] == ["P1", "P2"]
    assert st["underpowered"]
