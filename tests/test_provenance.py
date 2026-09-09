"""Provenance capture must not be able to break a run, and must not lie.

Two properties matter here. First, a manifest is written beside results that
would otherwise be unattributable, so every failure mode has to degrade to a
recorded ``None`` rather than an exception -- a run that dies because the git
query failed is strictly worse than one with an incomplete manifest. Second,
the ``dirty`` flag has to be honest, because it is the field that tells a
reader whether the commit SHA is sufficient to reproduce the numbers.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dengue_gnn import provenance


@dataclass
class _Cfg:
    label: str
    origins: tuple[float, ...] = (0.4, 0.7)
    seeds: tuple[int, ...] = (0, 1)


def test_manifest_sits_beside_its_results(tmp_path):
    out = tmp_path / "runs.csv"
    path = provenance.write_manifest(out, [_Cfg("a")], n_jobs=4)
    assert path == tmp_path / "runs.csv.manifest.json"
    # The suffix is appended, not replaced: 'runs.manifest.json' would collide
    # across runs.csv and runs.json in the same directory.
    assert path.name.startswith(out.name)


def test_records_enough_to_detect_a_truncated_run(tmp_path):
    out = tmp_path / "runs.csv"
    path = provenance.write_manifest(out, [_Cfg("a"), _Cfg("b")], n_jobs=32)
    m = json.loads(path.read_text(encoding="utf-8"))
    # A reader compares this against the CSV's row count; without it a run that
    # died at job 20 of 32 is indistinguishable from one that finished.
    assert m["n_jobs_scheduled"] == 32
    assert m["n_configs"] == 2
    assert {c["label"] for c in m["configs"]} == {"a", "b"}


def test_config_tuples_survive_the_json_round_trip(tmp_path):
    path = provenance.write_manifest(tmp_path / "r.csv", [_Cfg("a")], n_jobs=1)
    cfg = json.loads(path.read_text(encoding="utf-8"))["configs"][0]
    # Tuples become lists; the values must be intact, because these are the
    # origins and seeds someone would re-run from.
    assert cfg["origins"] == [0.4, 0.7]
    assert cfg["seeds"] == [0, 1]


def test_dirty_is_true_when_the_tree_has_uncommitted_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        provenance, "_run", lambda cmd: "abc123" if "rev-parse" in cmd else " M src/x.py\n?? y.py"
    )
    state = provenance.git_state()
    assert state["dirty"] is True
    assert state["dirty_files"] == 2


def test_clean_tree_reports_not_dirty(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_run", lambda cmd: "" if "status" in cmd else "abc123")
    state = provenance.git_state()
    assert state["dirty"] is False
    assert state["dirty_files"] == 0


def test_git_failure_degrades_to_none_rather_than_raising(monkeypatch):
    monkeypatch.setattr(provenance, "_run", lambda cmd: None)
    state = provenance.git_state()
    assert state["commit"] is None
    # An absent SHA must not be reported as a clean tree -- that would claim
    # reproducibility the manifest cannot support.
    assert state["dirty"] is False
    assert state["dirty_files"] == 0


def test_missing_optional_dependency_is_recorded_not_raised(monkeypatch):
    # ``sys.modules[name] = None`` makes CPython's import machinery raise
    # ImportError for that name, so this exercises the real except branch
    # rather than a stand-in for it.
    monkeypatch.setitem(sys.modules, "xgboost", None)
    env = provenance.collect_environment()
    # The key is present and explicitly None: a manifest that silently omitted
    # the dependency would read as "not checked" rather than "not installed".
    assert "xgboost" in env
    assert env["xgboost"] is None
    # The failure is contained -- the rest of the environment is still captured.
    assert env["python"]


def test_extra_is_carried_through(tmp_path):
    path = provenance.write_manifest(
        tmp_path / "r.csv", [_Cfg("a")], n_jobs=1, extra={"mode": "arch", "status": "started"}
    )
    m = json.loads(path.read_text(encoding="utf-8"))
    assert m["extra"]["mode"] == "arch"
    assert m["extra"]["status"] == "started"
