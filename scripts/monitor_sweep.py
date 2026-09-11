"""Monitor active Kaggle sweep kernels, download completed results, and launch queued kernels."""

import os
import sys
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# Ensure UTF-8 stdout/stderr
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "analysis" / "results" / "beat_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_KERNELS = [
    # Wave 1 (currently running):
    "beat-floor-a3tgcn",
    "covariates-a3tgcn",
    "beat-floor-stgat",
    "covariates-stgat",
    "beat-floor-combination",
    # Wave 2 (queued):
    "beat-floor-astgcn",
    "beat-floor-aagcn",
    "beat-floor-dcrnn",
    # Wave 3 (physics):
    "physics-sweep-a3tgcn",
    "physics-sweep-stgat",
    "physics-sweep-astgcn",
    "physics-sweep-aagcn",
    "physics-sweep-dcrnn",
    # Super-Ensemble Wave:
    "beat-floor-super-ensemble",
]

def get_status(api, kernel_name):
    try:
        st = api.kernels_status(f"tharushaperera16/{kernel_name}")
        return str(st.status).replace("KernelWorkerStatus.", "")
    except Exception as e:
        return f"ERROR ({e})"

def main():
    api = KaggleApi()
    api.authenticate()

    print("=== Kaggle Sweep Monitor ===")
    running = []
    completed = []
    queued = []

    for k in ALL_KERNELS:
        status = get_status(api, k)
        print(f"  {k:28s} : {status}")
        if status == "RUNNING":
            running.append(k)
        elif status == "COMPLETE":
            completed.append(k)
        else:
            queued.append(k)

    print(f"\nRunning: {len(running)}/5 | Completed: {len(completed)} | Queued: {len(queued)}")
    
    # Check if slots are available (limit 5)
    slots = 5 - len(running)
    if slots > 0 and queued:
        to_launch = queued[:slots]
        print(f"\nAvailable slots: {slots}. Launching next {len(to_launch)} queued kernel(s):")
        for k in to_launch:
            kernel_dir = BASE_DIR / "reproduction" / "kaggle" / "kernels" / k
            if kernel_dir.exists():
                print(f"  Pushing {k}...")
                try:
                    import subprocess
                    res = subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], capture_output=True, text=True)
                    print(f"    {res.stdout.strip() or res.stderr.strip()}")
                except Exception as e:
                    print(f"    Failed: {e}")
                time.sleep(2)

if __name__ == "__main__":
    main()
