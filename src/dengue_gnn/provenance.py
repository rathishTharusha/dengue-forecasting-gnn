"""Capture what is needed to reproduce a run, at the moment it is run.

Rule D2 requires every reported number to have a config, a commit SHA, seeds and
folds behind it. In practice that record was being written by hand into
``docs/EXPERIMENT_LOG.md`` after the fact, which is how the Phase-2 numbers ended
up citing scripts that no longer existed (review finding F1).

This module writes the record automatically, next to the results CSV, at the time
the run starts. It captures the things that silently change a result and are
invisible in the numbers themselves: the code version, whether the tree was dirty,
library versions, thread counts, and the full configuration of every arm.

A manifest is not a substitute for the experiment log -- the log carries the
question and the verdict, which no script can write. It is the mechanical half.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["collect_environment", "git_state", "write_manifest"]


def _run(cmd: list[str]) -> str | None:
    """Best-effort shell capture; returns None rather than raising."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def git_state(repo: Path | None = None) -> dict[str, Any]:
    """Commit SHA, branch, and whether the working tree is dirty.

    ``dirty`` matters more than the SHA: a run made against uncommitted changes
    cannot be reproduced from the SHA alone, and that is exactly the situation
    most of this project's results were produced in. Recording it means a reader
    knows the difference.
    """
    cwd = ["-C", str(repo)] if repo else []
    sha = _run(["git", *cwd, "rev-parse", "HEAD"])
    status = _run(["git", *cwd, "status", "--porcelain"])
    return {
        "commit": sha,
        "branch": _run(["git", *cwd, "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "dirty_files": len(status.splitlines()) if status else 0,
    }


def collect_environment() -> dict[str, Any]:
    """Versions and machine settings that change results without changing code."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or None,
    }
    for mod in ("numpy", "torch", "sklearn", "xgboost", "scipy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None

    try:
        import torch

        env["torch_threads"] = torch.get_num_threads()
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return env


def _serialise(cfg: Any) -> Any:
    """Configs are dataclasses; tuples become lists so JSON round-trips."""
    if is_dataclass(cfg) and not isinstance(cfg, type):
        cfg = asdict(cfg)
    if isinstance(cfg, dict):
        return {k: _serialise(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_serialise(v) for v in cfg]
    return cfg


def write_manifest(
    out_csv: Path,
    configs: list[Any],
    n_jobs: int,
    repo: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``<out_csv>.manifest.json`` describing the run that produced it.

    Args:
        out_csv: The results file this manifest documents. The manifest sits
            beside it so the two cannot be separated.
        configs: The configuration objects for every arm.
        n_jobs: Number of scheduled jobs, so a truncated run is detectable by
            comparing against the row count actually written.
        repo: Repository root for the git query.
        extra: Anything else worth recording.

    Returns:
        Path to the manifest.
    """
    out_csv = Path(out_csv)
    manifest = {
        "results_file": out_csv.name,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(repo),
        "environment": collect_environment(),
        "n_jobs_scheduled": n_jobs,
        "n_configs": len(configs),
        "configs": [_serialise(c) for c in configs],
    }
    if extra:
        manifest["extra"] = _serialise(extra)

    path = out_csv.with_suffix(out_csv.suffix + ".manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
