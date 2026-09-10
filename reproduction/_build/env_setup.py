"""Shared environment bootstrap for the Weng et al. reproduction kernels.

Kaggle's notebook image is **Python 3.12**. The authors pin ``torch==2.1.2``,
which has no cp312 wheel -- torch added Python 3.12 support in 2.2.0 -- so their
stack cannot be installed on the stock image::

    SystemExit: command failed: ('/usr/bin/python3', '-m', 'pip', 'install',
    '-q', 'torch==2.1.2', '--index-url', 'https://download.pytorch.org/whl/cpu')

Bumping torch would break fidelity to the paper. Instead these kernels use ``uv``
to fetch a standalone **Python 3.11**, build the authors' exact stack inside it,
and run their code through that interpreter as a subprocess. The notebook only
orchestrates; nothing of the authors' code is modified or re-implemented.

``SETUP`` and ``RUNNER_SCRIPT`` are shared verbatim by the probe kernel and by
all five model kernels, so the environment that is validated is byte-identical
to the one that produces the numbers.
"""

REPO = "https://github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2.git"
COMMIT = "45f1c0878f002407633ed1237638734faa9ceb2b"
#: Git blob SHAs of the two released inputs at COMMIT. Blob SHAs are used rather
#: than md5 of the working-tree bytes because git converts line endings on
#: checkout: a Windows clone of sri_lanka_adj_list.json hashes differently from a
#: Linux one, which failed a first Kaggle run with a spurious "checksum changed".
#: A blob SHA is the same everywhere and is bound to the pinned commit.
BLOB_NPY = "f7cfa6ec31a4058584fe256a1d6de6800e72a5b1"
BLOB_ADJ = "f3a3cb7f43998850410b0a494f16f331c3830a84"

#: Cell 1 -- imports, constants and a shell helper.
PREAMBLE = f'''
import subprocess, sys, os, json, time, hashlib, re
from pathlib import Path

REPO = "{REPO}"
COMMIT = "{COMMIT}"
BLOB_NPY = "{BLOB_NPY}"
BLOB_ADJ = "{BLOB_ADJ}"
SEGMENTS = [0.6, 0.7, 0.8, 0.9, 1.0]     # the authors' __main__ segment list

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
# Build outside /kaggle/working: anything left there becomes kernel output, and a
# 40 MB clone plus a venv makes `kaggle kernels output` unusably slow.
SCRATCH = Path("/tmp/repro") if Path("/tmp").exists() else WORK
SCRATCH.mkdir(parents=True, exist_ok=True)
SRC = SCRATCH / "mlos2"
VENV = SCRATCH / "venv311"
PY311 = VENV / "bin" / "python"

os.environ["MPLBACKEND"] = "Agg"          # their plotting helpers call plt.show()


def sh(*args, check=True, quiet=False, **kw):
    """Run a command, echoing it, and abort on a non-zero exit."""
    if not quiet:
        print("$", " ".join(str(a) for a in args))
    r = subprocess.run([str(a) for a in args], text=True, capture_output=True, **kw)
    if r.stdout.strip() and not quiet:
        print(r.stdout[-2000:])
    if r.returncode != 0:
        print(r.stderr[-4000:])
        if check:
            raise SystemExit("command failed: " + " ".join(str(a) for a in args))
    return r


print("kernel python:", sys.version.split()[0])
'''

#: Cell 2 -- build the authors' stack under a standalone Python 3.11.
SETUP = '''
# Kaggle runs Python 3.12; torch 2.1.2 has no cp312 wheel. Fetch a standalone
# 3.11 with uv rather than bumping the authors' pinned torch.
sh(sys.executable, "-m", "pip", "install", "-q", "uv")

UV = [sys.executable, "-m", "uv"]
sh(*UV, "python", "install", "3.11")
sh(*UV, "venv", "--python", "3.11", str(VENV))

PIP = [*UV, "pip", "install", "-q", "--python", str(PY311)]

# Exactly the versions in the authors' requirements.txt.
sh(*PIP, "torch==2.1.2", "--index-url", "https://download.pytorch.org/whl/cpu")
sh(*PIP, "torch_scatter", "torch_sparse", "-f",
   "https://data.pyg.org/whl/torch-2.1.2+cpu.html")

# Deviation 2: the authors pin torch_geometric==2.5.3, but PGT 0.54.0 imports
# torch_geometric.utils.to_dense_adj, which PyG removed in 2.4. 2.4.0 is the
# newest release where all five architectures import.
sh(*PIP, "torch_geometric==2.4.0", "numpy~=1.26.2", "pandas~=2.2.0",
   "scikit_learn==1.4.0", "statsmodels==0.14.1", "decorator==4.4.2",
   "cython", "matplotlib", "tqdm")

# Deviation 1: PGT's own pandas<=1.3.5 pin contradicts the authors'
# pandas~=2.2.0 and has no Python 3.11 wheel.
sh(*PIP, "--no-deps", "torch_geometric_temporal==0.54.0")
'''

#: Cell 3 -- assert the stack is what we intended, and that PGT imports.
VERIFY_ENV = '''
probe = sh(str(PY311), "-c", """
import json, torch, torch_geometric, pandas, numpy
from torch_geometric_temporal import A3TGCN, ASTGCN, AAGCN
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, temporal_signal_split
print(json.dumps({
    "python": ".".join(map(str, __import__("sys").version_info[:3])),
    "torch": torch.__version__,
    "torch_geometric": torch_geometric.__version__,
    "pandas": pandas.__version__,
    "numpy": numpy.__version__,
}))
""", quiet=True)

versions = json.loads(probe.stdout.strip().splitlines()[-1])
print(json.dumps(versions, indent=2))
assert versions["python"].startswith("3.11"), versions["python"]
assert versions["torch"].startswith("2.1.2"), versions["torch"]
assert versions["torch_geometric"] == "2.4.0", versions["torch_geometric"]
print("all five architectures import OK under Python 3.11")
'''

#: Cell 4 -- clone the authors' repository and verify the released inputs.
CLONE = '''
if not SRC.exists():
    sh("git", "clone", REPO, str(SRC))
sh("git", "-C", str(SRC), "checkout", "--quiet", COMMIT)

head = sh("git", "-C", str(SRC), "rev-parse", "HEAD", quiet=True).stdout.strip()
assert head == COMMIT, f"expected {COMMIT}, got {head}"
print("repository at", head)

# Verify the released inputs by git blob SHA. This is platform-independent --
# unlike an md5 of the checked-out bytes, which differs between a Windows and a
# Linux clone because git rewrites line endings in text files.
for rel, want in [("Data/Datasets/sri_lanka_2013-2022_shifted.npy", BLOB_NPY),
                  ("Models/sri_lanka_adj_list.json", BLOB_ADJ)]:
    got = sh("git", "-C", str(SRC), "rev-parse", f"HEAD:{rel}", quiet=True).stdout.strip()
    print(("OK  " if got == want else "MISMATCH ") + f"{rel}  {got}")
    assert got == want, f"{rel} blob changed: {got} != {want}"
'''


def runner_script(runner: str, epochs: str = "None") -> str:
    """A subprocess script that calls one of the authors' ``run_*`` functions.

    Args:
        runner: e.g. ``"run_stgat"``.
        epochs: ``"None"`` to leave the authors' ``num_epochs = 50`` untouched,
            or a literal integer for a smoke test.

    The script runs with ``cwd=Models`` because ``evaluation.py`` reads
    ``sri_lanka_adj_list.json`` at import time and resolves ``data_file``
    relative to that directory -- exactly as the authors' README instructs.
    """
    return f'''
import sys
sys.path.insert(0, ".")
import evaluation

override = {epochs}
if override is not None:
    evaluation.num_epochs = override      # smoke test only, NOT a reproduction

print("authors' hyperparameters, read from their module:")
for name in ("batch_size", "window_size", "in_channels", "out_channels",
             "num_nodes", "num_epochs", "lr", "decay", "dropout", "data_file"):
    print(f"  {{name:12s}} = {{getattr(evaluation, name)}}")

evaluation.{runner}({{SEGMENTS}})
'''
