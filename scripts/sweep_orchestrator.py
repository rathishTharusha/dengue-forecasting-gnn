"""Comprehensive Kaggle sweep orchestrator.

Checks kernel statuses, downloads new outputs safely with UTF-8 monkeypatch,
and prints full progress across all waves.
"""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Fix Windows encoding for Kaggle API
_orig_open = builtins.open
def _utf8_open(file, mode="r", *args, **kwargs):
    if "b" not in mode and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _orig_open(file, mode, *args, **kwargs)
builtins.open = _utf8_open

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from kaggle.api.kaggle_api_extended import KaggleApi

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "analysis" / "results" / "beat_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRACKED_KERNELS = [
    # Wave 1
    "beat-floor-a3tgcn",
    "covariates-a3tgcn",
    "beat-floor-stgat",
    "covariates-stgat",
    "beat-floor-combination",
    # Wave 2
    "beat-floor-astgcn",
    "beat-floor-aagcn",
    "beat-floor-dcrnn",
    # Physics Wave
    "physics-sweep-a3tgcn",
    "physics-sweep-stgat",
    "physics-sweep-aagcn",
    "physics-sweep-astgcn",
    "physics-sweep-dcrnn",
    # Super-Ensemble Wave
    "beat-floor-super-ensemble",
]

def get_status(api: KaggleApi, name: str) -> dict:
    try:
        st = api.kernels_status(f"tharushaperera16/{name}")
        status_str = str(st.status).replace("KernelWorkerStatus.", "")
        msg = getattr(st, "failure_message", None) or getattr(st, "failureMessage", "")
        return {"status": status_str, "error": msg}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

def download_output(api: KaggleApi, name: str):
    marker = OUT_DIR / f".downloaded_{name}"
    # Always check or download if marker doesn't exist
    if not marker.exists():
        print(f"  Downloading output for {name}...")
        try:
            api.kernels_output(f"tharushaperera16/{name}", path=str(OUT_DIR))
            marker.write_text(f"completed at {time.time()}\n", encoding="utf-8")
            print(f"  Downloaded {name} successfully.")
        except Exception as e:
            print(f"  Error downloading {name}: {e}")

def main():
    api = KaggleApi()
    api.authenticate()

    print("=================================================================")
    print("                    KAGGLE SWEEP ORCHESTRATOR                   ")
    print("=================================================================")

    counts = {"RUNNING": 0, "COMPLETE": 0, "ERROR": 0, "QUEUED": 0, "OTHER": 0}
    status_map = {}

    for name in TRACKED_KERNELS:
        info = get_status(api, name)
        st = info["status"]
        status_map[name] = info
        if st in counts:
            counts[st] += 1
        else:
            counts["OTHER"] += 1

        err_str = f" [ERROR: {info['error']}]" if info.get("error") else ""
        print(f"  {name:30s} : {st:12s}{err_str}")

        if st == "COMPLETE":
            download_output(api, name)

    print("-" * 65)
    print(f"Summary: Complete={counts['COMPLETE']}, Running={counts['RUNNING']}, Error={counts['ERROR']}")
    print("=================================================================")

    # If any new outputs were downloaded, run merge_beat
    try:
        res = subprocess.run(
            [sys.executable, str(REPO / "analysis" / "_build" / "merge_beat.py"),
             "--dir", str(OUT_DIR),
             "--out", str(REPO / "analysis" / "results" / "beat_summary.md")],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        print(res.stdout)
        if res.stderr and "Error" in res.stderr:
            print(res.stderr)
    except Exception as e:
        print(f"Error running merge_beat: {e}")

if __name__ == "__main__":
    main()
