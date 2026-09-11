"""Detailed inspection of all tracked Kaggle sweep kernels.

Prints exact kernel worker status, error message, current version, last run time,
and checks if outputs exist locally or if output download is pending.
"""

import builtins
_orig_open = builtins.open
def _utf8_open(file, mode="r", *args, **kwargs):
    if "b" not in mode and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _orig_open(file, mode, *args, **kwargs)
builtins.open = _utf8_open

import os
import sys
import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from kaggle.api.kaggle_api_extended import KaggleApi

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "analysis" / "results" / "beat_baseline"

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

def main():
    api = KaggleApi()
    api.authenticate()

    print("Fetching kernel list from Kaggle...")
    all_kernels = api.kernels_list(user="tharushaperera16", page_size=100)
    kernel_meta = {k.ref.split("/")[-1]: k for k in all_kernels}

    print("=" * 100)
    print(f"{'Kernel Name':30s} | {'Status':12s} | {'Ver':3s} | {'Last Run Time':24s} | Local Artifacts")
    print("=" * 100)

    for name in TRACKED_KERNELS:
        k = kernel_meta.get(name)
        ver = str(getattr(k, "current_version_number", "-")) if k else "-"
        last_run = str(getattr(k, "last_run_time", "-")) if k else "-"
        try:
            st = api.kernels_status(f"tharushaperera16/{name}")
            status_str = str(getattr(st, "status", "UNKNOWN")).replace("KernelWorkerStatus.", "")
            msg = getattr(st, "failure_message", None) or getattr(st, "failureMessage", "")
            err = f" [FAIL: {msg}]" if msg else ""
        except Exception as e:
            status_str = "ERROR"
            err = f" [{e}]"

        marker = OUT_DIR / f".downloaded_{name}"
        local_files = []
        if name == "beat-floor-a3tgcn":
            local_files = [f.name for f in OUT_DIR.glob("beat_A3TGCN*.json")]
        elif name == "beat-floor-stgat":
            local_files = [f.name for f in OUT_DIR.glob("beat_STGAT*.json")]
        elif name == "beat-floor-aagcn":
            local_files = [f.name for f in OUT_DIR.glob("beat_AAGCN*.json")]
        elif name == "beat-floor-astgcn":
            local_files = [f.name for f in OUT_DIR.glob("beat_ASTGCN*.json")]
        elif name == "beat-floor-dcrnn":
            local_files = [f.name for f in OUT_DIR.glob("beat_DCRNN*.json")]
        elif name == "beat-floor-super-ensemble":
            local_files = [f.name for f in OUT_DIR.glob("beat_super*.json")]

        local_info = f"{len(local_files)} JSON(s)" if local_files else ("Downloaded" if marker.exists() else "None")
        print(f"{name:30s} | {status_str:12s} | {ver:3s} | {last_run[:24]:24s} | {local_info}{err}")

    print("=" * 100)

if __name__ == "__main__":
    main()
