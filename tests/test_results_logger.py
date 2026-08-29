"""Tests for dengue_gnn.results_logger.

Rewritten alongside the module during the Phase-2 review (finding F10). The
theme of these tests is that every failure mode must raise: a results logger
that writes a plausible wrong number is worse than one that crashes, because
nothing downstream can tell a real 0.0 from a missing one.
"""

import csv

import pytest

from dengue_gnn.results_logger import append_rows, read_rows

ROW = {"config": "adaptive", "fold": 1, "seed": 0, "horizon": 1, "rmse": 27.15}


def test_writes_header_and_row(tmp_path):
    path = tmp_path / "r.csv"
    append_rows([ROW], path, stamp=False)

    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["config"] == "adaptive"
    assert float(rows[0]["rmse"]) == pytest.approx(27.15)


def test_appends_without_repeating_header(tmp_path):
    path = tmp_path / "r.csv"
    append_rows([ROW], path, stamp=False)
    append_rows([{**ROW, "seed": 1}], path, stamp=False)

    assert len(read_rows(path)) == 2
    with open(path, encoding="utf-8") as fh:
        assert sum(1 for line in fh if line.startswith("config,")) == 1


def test_adds_timestamp_by_default(tmp_path):
    path = tmp_path / "r.csv"
    append_rows([ROW], path)
    assert "timestamp" in read_rows(path)[0]


def test_creates_missing_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "r.csv"
    append_rows([ROW], path, stamp=False)
    assert path.exists()


def test_empty_rows_raise(tmp_path):
    """F10: silently writing nothing is how an empty results file gets cited."""
    with pytest.raises(ValueError, match="empty result set"):
        append_rows([], tmp_path / "r.csv")


def test_ragged_rows_raise(tmp_path):
    with pytest.raises(ValueError, match="inconsistent keys"):
        append_rows([ROW, {"config": "x"}], tmp_path / "r.csv", stamp=False)


def test_schema_change_on_append_raises(tmp_path):
    """The old logger wrote the header once, then misaligned every later column."""
    path = tmp_path / "r.csv"
    append_rows([ROW], path, stamp=False)

    with pytest.raises(ValueError, match="would silently misalign"):
        append_rows([{**ROW, "new_column": 1.0}], path, stamp=False)


def test_no_silent_zero_fallback(tmp_path):
    """The core F10 regression.

    The previous implementation resolved a missing metric column to 0.0. A run
    that failed to record its RMSE therefore appeared in the results file as a
    perfect score. Here a missing key is a schema error instead.
    """
    path = tmp_path / "r.csv"
    append_rows([ROW], path, stamp=False)

    incomplete = {k: v for k, v in ROW.items() if k != "rmse"}
    with pytest.raises(ValueError):
        append_rows([incomplete], path, stamp=False)

    assert len(read_rows(path)) == 1  # nothing was appended


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no results file"):
        read_rows(tmp_path / "absent.csv")
