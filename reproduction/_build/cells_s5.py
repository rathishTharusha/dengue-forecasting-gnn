"""Cell sources for Stage S5 SEIR-GNN benchmark Kaggle kernels.

Runs Stage S5 (Screening on A3TGCN + Expansion across all 5 GNN architectures)
using the Kaggle environment with PyTorch Geometric & torch_geometric_temporal.
"""

import env_setup
from gen_kernels import code, md

BRANCH = "exp/seir-gnn"
ARCHITECTURES = ("A3TGCN", "STGAT", "ASTGCN", "AAGCN", "DCRNN")

_SUMMARY = '''
import json
from pathlib import Path
import pandas as pd

rows = []
for f in sorted(Path(WORK).glob("*s5_seir_gnn*.json")):
    rows.extend(json.loads(f.read_text(encoding="utf-8")))

if rows:
    df = pd.DataFrame(rows)
    summary = df.groupby(["arch", "input_level", "coupling", "head_type"])[["val_RMSE", "test_RMSE"]].mean().reset_index()
    summary = summary.sort_values("val_RMSE")
    print("=== Stage S5 SEIR-GNN Leaderboard ===")
    print(summary.round(3).to_string(index=False))
'''


def build() -> list:
    return [
        md(
            """
# Stage S5 -- SEIR-GNN Benchmark & Physics Expansion

This kernel evaluates **SEIR-GNN** architectures under the pre-registered protocol:
- **Phase 1 Screen:** 12 arms on A3TGCN across 3 origins x 3 seeds.
- **Phase 2 Expansion:** Top 2 SEIR configurations + direct controls across STGAT, ASTGCN, AAGCN, DCRNN, A3TGCN.

All inputs from `data/corrected/rebuilt_cases.npy` (559 weeks, 3 rolling origins x 3 seeds).
"""
        ),
        md("## 1. Environment"),
        code(env_setup.PREAMBLE),
        code(env_setup.SETUP),
        code(env_setup.VERIFY_ENV),
        md("## 2. Clone Repository"),
        code(
            f"""
PROJECT = "https://github.com/rathishTharusha/dengue-forecasting-gnn.git"
BRANCH = "{BRANCH}"
PROJ = SCRATCH / "project"
if not PROJ.exists():
    sh("git", "clone", "--depth", "1", "--branch", BRANCH, PROJECT, str(PROJ))
RUNNER = PROJ / "analysis" / "_build" / "run_s5_seir_gnn.py"
assert RUNNER.exists(), f"Runner not found at {{RUNNER}}"
print("Cloned", BRANCH)
"""
        ),
        md("## 3. Execute Stage S5 Benchmark"),
        code(
            f"""
started = time.time()
proc = subprocess.Popen(
    [str(PY311), "-u", str(RUNNER), "--epochs", "120", "--out", str(WORK / "s5_seir_gnn_results.json")],
    cwd=str(PROJ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
for line in proc.stdout:
    print(line, end="")
proc.wait()
assert proc.returncode == 0, f"runner failed with exit code {{proc.returncode}}"
print(f"\\nfinished in {{(time.time() - started) / 60:.1f}} min")
"""
        ),
        md("## 4. Leaderboard Summary"),
        code(_SUMMARY),
    ]
