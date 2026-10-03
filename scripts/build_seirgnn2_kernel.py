"""Generate the Kaggle kernel that runs a seirgnn2 grid.

The kernel clones the pushed branch rather than carrying a copy of the code, so
there is exactly one source of truth and nothing can drift between what ran
locally and what ran on Kaggle. The commit SHA is recorded in the output so a
result can always be traced back to the code that produced it -- plan R7.

Usage::

    python scripts/build_seirgnn2_kernel.py --grid confirm
    kaggle kernels push -p reproduction/kaggle/kernels/seirgnn2-confirm

Then collect the output with::

    kaggle kernels output tharushaperera16/seirgnn2-confirm -p seirgnn2/results

The branch must be pushed first: the kernel clones it, so an uncommitted local
change is invisible to the run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNELS = REPO / "reproduction" / "kaggle" / "kernels"
USER = "tharushaperera16"


def _cell(kind: str, text: str) -> dict:
    lines = text.strip("\n").split("\n")
    source = [f"{ln}\n" for ln in lines[:-1]] + [lines[-1]]
    cell = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": source}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def notebook(grid: str, sha: str, workers: int, epochs: int, branch: str,
             keep: bool, ensembles: list[list[str]]) -> dict:
    setup = f"""
# Shell work goes through subprocess, not IPython magics: `tools/check_notebooks.py`
# parses every committed notebook as Python and runs in CI, so `!` and `%` cells
# fail the build.
import os, subprocess, sys
from pathlib import Path


def sh(*cmd, **kw):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], text=True, **kw)
    if r.returncode:
        raise SystemExit(f"failed ({{r.returncode}}): {{cmd}}")
    return r


REPO_URL = "https://github.com/rathishTharusha/dengue-forecasting-gnn.git"
# Pinned to the commit this kernel was generated from, so the result is traceable.
SHA = "{sha}"
if not Path("repo").exists():
    sh("git", "clone", "--quiet", "--branch", "{branch}", REPO_URL, "repo")
os.chdir("repo")
sh("git", "checkout", "--quiet", SHA)
sh("git", "log", "--oneline", "-1")
"""
    deps = """
# Two installs, and the order matters.
#
# torch_geometric is a hard requirement of the five architectures and Kaggle does
# not ship it, so it is installed WITH its dependencies (it is pure Python; no
# build). Leaving it out is what broke kernel version 1: --no-deps on
# torch-geometric-temporal also skipped torch_geometric, so every graph arm died
# in its worker while the LSTM arms -- which need no PyG -- ran fine, and the
# grid came back 63 rows instead of 144 with no obvious error.
#
# torch-geometric-temporal then goes in WITHOUT dependencies, because its
# declared torch-sparse / torch-scatter have no wheels here and would try a
# source build. It needs torch_sparse only for evolvegcno, which none of the five
# use; backbones.install_shim() supplies the symbol.
sh(sys.executable, "-m", "pip", "install", "--quiet", "torch-geometric")
sh(sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "torch-geometric-temporal")
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
import torch_geometric; print("pyg", torch_geometric.__version__)
"""
    check = """
# Gate: every encoder must build and run before the grid starts. Kernel v1 had no
# gate, so a broken install looked like a short results file rather than a failure.
out = subprocess.run([sys.executable, "seirgnn2/backbones.py"],
                     capture_output=True, text=True)
print(out.stdout or out.stderr)
assert "FAIL" not in out.stdout, "an encoder failed to build -- fix before running the grid"
"""
    keep_flag = ', "--keep"' if keep else ""
    run = f"""
# Workers, not GPU: every architecture here is small and the grid is many short
# runs, so process-level parallelism over CPU cores beats one GPU stream. Set
# enable_gpu in kernel-metadata.json if a future grid actually needs it.
sh(sys.executable, "seirgnn2/sweep.py", "{grid}", "--workers", "{workers}", "--epochs", "{epochs}"{keep_flag})
"""
    collect = f"""
import json, shutil
from pathlib import Path
for f in sorted(Path("seirgnn2/results").glob("{grid}*")):
    shutil.copy(f, Path("/kaggle/working") / f.name)
    print("saved", f.name, f.stat().st_size, "bytes")
"""
    ens_lines = "\n".join(
        f'sh(sys.executable, "seirgnn2/ensemble.py", "{grid}", ' + ", ".join(f'"{m}"' for m in e) + ")"
        for e in ensembles)
    stats_grid = f"{grid}+ens" if ensembles else grid
    stats = f"""
# Ensembles are pre-registered with equal weights (docs/REMEDIES_PLAN.md, R5):
# nothing about them is fitted, so they are computed here rather than chosen.
{ens_lines}
sh(sys.executable, "seirgnn2/stats.py", "{stats_grid}", "--ref", "B", "--metric", "val_RMSE")
sh(sys.executable, "seirgnn2/stats.py", "{stats_grid}", "--ref", "persistence", "--metric", "val_RMSE")
""" if ensembles else f"""
sh(sys.executable, "seirgnn2/stats.py", "{grid}")
"""
    cells = [
        _cell("markdown", f"# seirgnn2 — `{grid}`\n\n"
                          f"Branch `{branch}` at `{sha}`. Selection is on validation "
                          f"only (plan R6); test numbers are written but not compared "
                          f"until a config is frozen."),
        _cell("code", setup), _cell("code", deps),
        _cell("markdown", "## All six encoders build and run"), _cell("code", check),
        _cell("markdown", f"## Run the `{grid}` grid"), _cell("code", run),
        _cell("markdown", "## Paired tests"), _cell("code", stats),
        _cell("markdown", "## Save"), _cell("code", collect),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="confirm")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--gpu", action="store_true", help="request a GPU (off by default)")
    ap.add_argument("--branch", default=None, help="branch to clone (default: current)")
    ap.add_argument("--keep", action="store_true", help="save every run's forecasts")
    args = ap.parse_args()

    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True,  # noqa: E731
                                    text=True, check=True).stdout.strip()
    sha = git("rev-parse", "HEAD")
    branch = args.branch or git("rev-parse", "--abbrev-ref", "HEAD")
    import sys
    sys.path.insert(0, str(REPO / "seirgnn2"))
    import grids
    ensembles = grids.ENSEMBLES.get(args.grid, []) if args.keep else []
    slug = f"seirgnn2-{args.grid}"
    out = KERNELS / slug
    out.mkdir(parents=True, exist_ok=True)

    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{slug}",
        "title": f"seirgnn2 {args.grid}",
        "code_file": f"{slug.replace('-', '_')}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": args.gpu,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2) + "\n", encoding="utf-8")

    nb = notebook(args.grid, sha, args.workers, args.epochs, branch, args.keep, ensembles)
    (out / f"{slug.replace('-', '_')}.ipynb").write_text(
        json.dumps(nb, indent=1) + "\n", encoding="utf-8")

    print(f"wrote {out}")
    print(f"  branch {branch} @ {sha[:7]}  grid={args.grid} workers={args.workers} "
          f"epochs={args.epochs} gpu={args.gpu} keep={args.keep} ensembles={len(ensembles)}")
    print(f"\n  kaggle kernels push -p {out.relative_to(REPO)}")
    print(f"  kaggle kernels status {USER}/{slug}")
    print(f"  kaggle kernels output {USER}/{slug} -p seirgnn2/results")


if __name__ == "__main__":
    main()
