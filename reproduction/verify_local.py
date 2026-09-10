"""Prove the pinned stack runs the authors' code, before spending Kaggle hours.

Builds the environment the Kaggle notebooks build, clones the authors'
repository, and calls each of their five ``run_*`` functions with
``num_epochs = 1`` on a single segment. That is **not** a reproduction -- the
numbers it prints are meaningless -- it only answers "does their code execute at
all in this environment", which is the question worth answering before queueing a
multi-hour kernel.

The environment is built at a short path outside the project folder for two
reasons, both learned the hard way here:

* Google Drive sync corrupts partially-written packages (``numpy`` came back
  reporting version ``None``).
* torch's nested headers exceed Windows ``MAX_PATH`` under a deep temp
  directory, and pip fails with ``OSError: [Errno 2]``.

Usage::

    python reproduction/verify_local.py            # build env, clone, smoke test
    python reproduction/verify_local.py --env-only # just build the environment
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2.git"
COMMIT = "45f1c0878f002407633ed1237638734faa9ceb2b"
RUNNERS = ["run_stgat", "run_a3tgcn", "run_astgcn", "run_dcrnn", "run_aagcn"]

#: Short, local, non-synced. See the module docstring for why both matter.
BASE = Path("C:/rp") if platform.system() == "Windows" else Path.home() / ".cache" / "rp"
VENV = BASE / "v"
SRC = BASE / "mlos2"


def _python() -> Path:
    return VENV / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def run(*args: str, **kw) -> subprocess.CompletedProcess:
    """Run a command, echoing it, and abort on failure."""
    print("$", " ".join(str(a) for a in args))
    result = subprocess.run([str(a) for a in args], text=True, **kw)
    if result.returncode != 0:
        raise SystemExit(f"failed: {' '.join(str(a) for a in args)}")
    return result


def build_env() -> None:
    """Install exactly what the Kaggle notebooks install."""
    BASE.mkdir(parents=True, exist_ok=True)
    if not _python().exists():
        run(sys.executable, "-m", "venv", VENV)

    py = _python()
    pip = [py, "-m", "pip", "install", "-q"]
    run(*pip, "--upgrade", "pip")

    # The authors' pinned torch, plus the PyG companion wheels built against it.
    run(*pip, "torch==2.1.2", "--index-url", "https://download.pytorch.org/whl/cpu")
    run(
        *pip, "torch_scatter", "torch_sparse", "-f", "https://data.pyg.org/whl/torch-2.1.2+cpu.html"
    )

    # Deviation 2: the authors pin torch_geometric==2.5.3, but PGT 0.54.0 imports
    # torch_geometric.utils.to_dense_adj, which PyG removed in 2.4.
    run(
        *pip,
        "torch_geometric==2.4.0",
        "numpy~=1.26.2",
        "pandas~=2.2.0",
        "scikit_learn==1.4.0",
        "statsmodels==0.14.1",
        "decorator==4.4.2",
        "cython",
        "matplotlib",
        "tqdm",
    )

    # Deviation 1: PGT's own pandas<=1.3.5 pin contradicts the authors'
    # pandas~=2.2.0 and has no Python 3.11 wheel.
    run(*pip, "--no-deps", "torch_geometric_temporal==0.54.0")

    run(
        py,
        "-c",
        (
            "import torch, torch_geometric, pandas, numpy;"
            "from torch_geometric_temporal import A3TGCN, ASTGCN, AAGCN;"
            "from torch_geometric_temporal.nn.recurrent import DCRNN;"
            "print('torch', torch.__version__, '| pyg', torch_geometric.__version__,"
            "      '| pandas', pandas.__version__, '| numpy', numpy.__version__);"
            "print('all five architectures import OK')"
        ),
    )


def clone() -> None:
    """Clone the authors' repository at the pinned commit."""
    if not SRC.exists():
        run("git", "clone", REPO, SRC)
    run("git", "-C", SRC, "checkout", "--quiet", COMMIT)
    head = subprocess.run(
        ["git", "-C", str(SRC), "rev-parse", "HEAD"], text=True, capture_output=True
    ).stdout.strip()
    if head != COMMIT:
        raise SystemExit(f"expected {COMMIT}, got {head}")
    print("repository at", head)


def smoke() -> None:
    """Call every run_* with one epoch on one segment; report execution only."""
    script = (
        "import io, contextlib\n"
        "import evaluation\n"
        "evaluation.num_epochs = 1   # smoke test, NOT the reproduction\n"
        f"for name in {RUNNERS!r}:\n"
        "    buf = io.StringIO()\n"
        "    try:\n"
        "        with contextlib.redirect_stdout(buf):\n"
        "            getattr(evaluation, name)([0.6])\n"
        "        lines = [l for l in buf.getvalue().splitlines() if 'Average' in l]\n"
        "        print(f'{name:12s} OK   ' + (lines[-2] if len(lines) >= 2 else str(lines)))\n"
        "    except Exception as exc:\n"
        "        print(f'{name:12s} FAIL {type(exc).__name__}: {str(exc)[:200]}')\n"
    )
    env = dict(os.environ, MPLBACKEND="Agg")
    print("\n-- smoke test: 1 epoch, 1 segment. Numbers below are meaningless. --")
    run(_python(), "-u", "-c", script, cwd=SRC / "Models", env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-only", action="store_true", help="build the environment and stop")
    args = parser.parse_args()

    build_env()
    if args.env_only:
        return 0
    clone()
    smoke()
    print("\nIf all five report OK, the Kaggle kernels will run. Push them with:")
    print("  cd reproduction/kaggle/kernels/weng2024-stgat && kaggle kernels push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
