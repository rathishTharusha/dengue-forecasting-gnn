"""Push the search notebook to Kaggle, wait for it, and pull the results CSV.

**This script never handles your credential.** It shells out to the `kaggle` CLI,
which reads `~/.kaggle/kaggle.json` (or `KAGGLE_USERNAME`/`KAGGLE_KEY`) on its own.
Nothing here reads, prints, stores or transmits the key, and none of it should ever
be pasted into a file in this repository -- a credential committed to git, or typed
into a chat, is a credential that must be rotated.

Setup, once, by you:

    pip install kaggle
    # download kaggle.json from kaggle.com -> Settings -> API -> Create New Token
    mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json

Then:

    python scripts/kaggle_run.py --user <your-kaggle-username> data     # once
    python scripts/kaggle_run.py --user <your-kaggle-username> push
    python scripts/kaggle_run.py --user <your-kaggle-username> status
    python scripts/kaggle_run.py --user <your-kaggle-username> pull

A note on compute, measured rather than assumed
------------------------------------------------
Kaggle gives 4 CPU cores per notebook; this machine has 12. The workload is
overhead-bound (7k parameters on a (1, 25, 33) input), so a GPU does not help --
one torch thread beats ten locally. A single Kaggle kernel is therefore *slower*
than running locally.

Kaggle is still worth it for: long unattended runs (9h sessions), keeping the
laptop free, running several kernels concurrently, and the GAN in Stage 5, which
is genuinely compute-bound. It is not worth it for raw speed on one kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KAGGLE_DIR = REPO / "kaggle"
NOTEBOOK = REPO / "notebooks" / "04_kaggle_search.ipynb"
DATA_FILES = [
    REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy",
    REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json",
]
SLUG = "dengue-stage3-search"
DATA_SLUG = "dengue-sri-lanka-graph"


def have_credentials() -> bool:
    """True if the CLI will find a credential. The value is never read here."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def require_cli() -> None:
    if shutil.which("kaggle") is None:
        raise SystemExit(
            "kaggle CLI not found.\n"
            "  pip install kaggle\n"
            "then place kaggle.json at ~/.kaggle/kaggle.json (chmod 600)."
        )
    if not have_credentials():
        raise SystemExit(
            "No Kaggle credential found.\n"
            "  Download kaggle.json from kaggle.com -> Settings -> API -> Create New Token\n"
            "  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
            "  chmod 600 ~/.kaggle/kaggle.json\n"
            "Do not paste the key into a file in this repository."
        )


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode


def write_data_metadata(user: str) -> Path:
    """Stage the two data files plus a dataset spec for `kaggle datasets`."""
    out = KAGGLE_DIR / "data"
    out.mkdir(parents=True, exist_ok=True)
    for f in DATA_FILES:
        if not f.exists():
            raise SystemExit(f"missing data file: {f}")
        shutil.copy2(f, out / f.name)
    (out / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Sri Lanka dengue graph tensor",
                "id": f"{user}/{DATA_SLUG}",
                "licenses": [{"name": "unknown"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def write_kernel_metadata(user: str, gpu: bool) -> Path:
    """Stage the notebook plus a kernel spec for `kaggle kernels push`."""
    out = KAGGLE_DIR / "kernel"
    out.mkdir(parents=True, exist_ok=True)
    if not NOTEBOOK.exists():
        raise SystemExit(
            f"{NOTEBOOK.relative_to(REPO)} not found.\n"
            "  cd notebooks/_build && python gen_notebooks.py"
        )
    shutil.copy2(NOTEBOOK, out / NOTEBOOK.name)
    (out / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{user}/{SLUG}",
                "title": "Dengue Stage-3 search",
                "code_file": NOTEBOOK.name,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                # GPU off by default: measured, this workload is overhead-bound
                # and a GPU instance also gives fewer CPU cores.
                "enable_gpu": gpu,
                "enable_internet": True,  # the notebook clones the repo
                "dataset_sources": [f"{user}/{DATA_SLUG}"],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("data", "push", "status", "pull", "check"))
    ap.add_argument("--user", help="your Kaggle username (not the API key)")
    ap.add_argument(
        "--gpu", action="store_true", help="request a GPU (not recommended; see docstring)"
    )
    ap.add_argument("--out", type=Path, default=REPO / "results")
    args = ap.parse_args()

    if args.action == "check":
        print("kaggle CLI :", shutil.which("kaggle") or "NOT FOUND")
        print("credential :", "found" if have_credentials() else "NOT FOUND")
        print("notebook   :", "present" if NOTEBOOK.exists() else "NOT GENERATED")
        return 0 if (shutil.which("kaggle") and have_credentials() and NOTEBOOK.exists()) else 1

    require_cli()
    if not args.user:
        raise SystemExit("--user <your-kaggle-username> is required for this action")

    if args.action == "data":
        d = write_data_metadata(args.user)
        print(f"staged {d.relative_to(REPO)}; creating dataset (use --version to update)")
        rc = run(["kaggle", "datasets", "create", "-p", str(d), "--dir-mode", "zip"])
        if rc:
            print("\ncreate failed -- if the dataset already exists, publish a new version:")
            print(f'  kaggle datasets version -p {d} -m "update"')
        return rc

    if args.action == "push":
        k = write_kernel_metadata(args.user, args.gpu)
        return run(["kaggle", "kernels", "push", "-p", str(k)])

    if args.action == "status":
        return run(["kaggle", "kernels", "status", f"{args.user}/{SLUG}"])

    if args.action == "pull":
        args.out.mkdir(parents=True, exist_ok=True)
        rc = run(["kaggle", "kernels", "output", f"{args.user}/{SLUG}", "-p", str(args.out)])
        if rc == 0:
            csv_path = args.out / "stage3_search.csv"
            print(
                f"\n-> {csv_path.relative_to(REPO) if csv_path.exists() else 'no CSV found'}\n"
                "Next: python scripts/make_tables.py --runs results/stage3_search.csv "
                "--prefix search --no-tex\n"
                "Then log it in docs/EXPERIMENT_LOG.md (D2: a number in a Kaggle output "
                "pane does not exist)."
            )
        return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
