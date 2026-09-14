"""Cell sources for the corrected-data benchmark kernels.

One kernel per architecture. Each runs the original array and both corrected
case series (``reordered``, ``rebuilt``) through the same training loop, so the
data is the only thing that differs between the three result sets. See
``docs/ARRAY_AUDIT.md`` for why.
"""

import env_setup
from gen_kernels import code, md

#: The branch holding the corrected datasets. Must be pushed before a kernel can clone it.
BRANCH = "feat/seir-gnn"

ARCHITECTURES = ("A3TGCN", "STGAT", "ASTGCN", "AAGCN", "DCRNN")
DATASETS = ("original", "reordered", "rebuilt")
SEEDS = 3

_SUMMARY = '''
import json
from pathlib import Path

import numpy as np
import pandas as pd

rows = []
for f in sorted(Path(WORK).glob("corrected_*.json")):
    rows.extend(json.loads(f.read_text(encoding="utf-8")))
df = pd.DataFrame(rows).drop_duplicates(["dataset", "arch", "increment", "origin", "seed"])

floor = (df[df.arch == "persistence"].groupby(["dataset", "origin"])["RMSE_clean"].mean())
model = df[df.arch != "persistence"]
table = (model.groupby(["dataset", "arch", "increment", "origin"])["RMSE_clean"].mean()
         .reset_index())
table["floor"] = [floor[(d, o)] for d, o in zip(table.dataset, table.origin)]
table["vs_floor"] = table.RMSE_clean - table.floor
summary = (table.groupby(["dataset", "arch", "increment"], sort=False)
           .agg(RMSE_clean=("RMSE_clean", "mean"), floor=("floor", "mean"),
                vs_floor=("vs_floor", "mean"), wins=("vs_floor", lambda s: int((s < 0).sum()))))
pd.set_option("display.width", 160)
print(summary.round(3).to_string())
'''


def build(arch: str) -> list:
    datasets = ", ".join(f'"{d}"' for d in DATASETS)
    return [
        md(
            f"""
# Corrected-data benchmark - {arch}

The processed array the benchmark used is out of date order: its first 48 rows
are 2023, placed before 2013, so every training split contained data from after
its own test period. It also drops 49 weekly reports and stores one week twice.
`docs/ARRAY_AUDIT.md` has the evidence.

This kernel re-runs **{arch}** on three versions of the case series with **one
training loop**, so the data is the only thing that changes:

| dataset | what it is |
|---|---|
| `original` | the array as the benchmark used it (459 rows) |
| `reordered` | the same rows in true date order, duplicate removed (451 weeks) |
| `rebuilt` | every source report on a regular weekly grid (559 weeks) |

Frozen protocol unchanged: 3 origins x {SEEDS} seeds, window 3 -> horizon 3,
pooled RMSE. The artifact is located by report, not by row 395, and windows
whose target is an interpolated week are excluded. Arms: `base` and `spatial`.

`original` should reproduce the earlier numbers; if it does not, stop and compare.
"""
        ),
        md("## 1. Environment"),
        code(env_setup.PREAMBLE),
        code(env_setup.SETUP),
        code(env_setup.VERIFY_ENV),
        md("## 2. Clone"),
        code(
            f"""
ARCH = "{arch}"
PROJECT = "https://github.com/rathishTharusha/dengue-forecasting-gnn.git"
BRANCH = "{BRANCH}"
PROJ = SCRATCH / "project"
if not PROJ.exists():
    sh("git", "clone", "--depth", "1", "--branch", BRANCH, PROJECT, str(PROJ))
RUNNER = PROJ / "analysis" / "_build" / "run_corrected_benchmark.py"
assert RUNNER.exists(), f"Runner not found at {{RUNNER}}"
assert (PROJ / "data" / "corrected" / "rebuilt_cases.npy").exists(), "corrected data missing"
print("Cloned", BRANCH)
"""
        ),
        md("## 3. Run all three datasets"),
        code(
            f"""
started = time.time()
proc = subprocess.Popen(
    [str(PY311), "-u", str(RUNNER), "--arch", ARCH, "--datasets", {datasets},
     "--seeds", "{SEEDS}", "--out-dir", str(WORK)],
    cwd=str(PROJ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
for line in proc.stdout:
    print(line, end="")
proc.wait()
assert proc.returncode == 0, f"runner failed with exit code {{proc.returncode}}"
print(f"\\nfinished in {{(time.time() - started) / 60:.1f}} min")
"""
        ),
        md("## 4. Summary"),
        code(_SUMMARY),
    ]
