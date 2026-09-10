"""Cell sources for individual per-architecture physics sweep kernels on Kaggle.

Running each architecture in its own kernel prevents wall-clock exhaustion,
allows parallel execution on Kaggle, and isolates failures.

Architectures:
- STGAT: Attention-based spatio-temporal GNN
- AAGCN: Adaptive / learned graph convolutional network
- A3TGCN: Temporal gated convolution network
"""

import env_setup
from gen_kernels import code, md

BRANCH = "feat/physics-informed-loss"
ARCHITECTURES = ("STGAT", "AAGCN", "A3TGCN")


def build(arch: str) -> list:
    """Cells for one architecture's physics sweep kernel."""
    return [
        md(
            f"""
# Physics-Informed Loss Sweep — {arch}

This Kaggle kernel evaluates whether relaxed biological envelope and spatial
regularizers improve forecasting accuracy and resolve the lag-under-reaction
pathology for **{arch}**.

## Evaluated Arms:
1. `base`: Unconstrained GNN baseline.
2. `envelope`: SEIR-SEI upper biological growth ceiling ($r_{{\\text{{max}}}} = 2.3884$) and clearance bounds.
3. `spatial`: District-normalized spatial Dirichlet flux smoothness across connected districts.
4. `composite`: Combined biological envelope + spatial smoothness + non-negativity.
5. `outbreak_aware`: Composite regularizer + asymmetric under-prediction weighting ($w_{{\\text{{under}}}} = 2.5$).

Protocol: 3 chronological origins ($0.55, 0.70, 0.85$) $\\times$ 3 seeds ($0, 1, 2$) = 9 runs per arm.
Evaluated on all windows and artifact-free clean windows (excluding the 2019 backlog artifact).
"""
        ),
        md("## 1. Environment Bootstrap"),
        code(env_setup.PREAMBLE),
        code(env_setup.SETUP),
        code(env_setup.VERIFY_ENV),
        md("## 2. Clone Repository with Physics Framework"),
        code(
            f"""
ARCH = "{arch}"
PROJECT = "https://github.com/rathishTharusha/dengue-forecasting-gnn.git"
BRANCH = "{BRANCH}"
PROJ = SCRATCH / "project"

if not PROJ.exists():
    sh("git", "clone", "--depth", "1", "--branch", BRANCH, PROJECT, str(PROJ))
print("Cloned branch:", BRANCH)

RUNNER = PROJ / "analysis" / "_build" / "run_physics_experiments.py"
assert RUNNER.exists(), f"Runner script not found at {{RUNNER}}"
RESULTS_JSON = WORK / f"physics_envelope_{{ARCH}}.json"
"""
        ),
        md("## 3. Execute Physics Sweep for Architecture"),
        code(
            f"""
started = time.time()
print(f"Executing physics sweep for {{ARCH}} (3 seeds x 3 origins x 5 arms = 45 runs)...")

proc = subprocess.Popen(
    [
        str(PY311),
        "-u",
        str(RUNNER),
        "--arch",
        ARCH,
        "--seeds",
        "3",
        "--out-dir",
        str(WORK),
    ],
    cwd=str(PROJ),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
for line in proc.stdout:
    print(line, end="")
proc.wait()

assert proc.returncode == 0, f"Physics sweep runner failed with exit code {{proc.returncode}}"
print(f"\\nSweep for {{ARCH}} completed successfully in {{(time.time()-started)/60:.1f}} min")
"""
        ),
        md("## 4. Empirical Evaluation & Comparison Tables"),
        code(
            f"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

assert RESULTS_JSON.exists(), f"Results file {{RESULTS_JSON}} does not exist!"
records = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
df = pd.DataFrame(records)

print(f"=== SUMMARY RESULTS FOR {{ARCH}} (n=9 runs per arm) ===")
p_floor_clean = df[df["arch"] == "persistence"]["RMSE_clean"].mean()
p_floor_all = df[df["arch"] == "persistence"]["RMSE"].mean()

print(f"Persistence Floor: All RMSE = {{p_floor_all:.2f}}, Clean RMSE = {{p_floor_clean:.2f}}\\n")

model_df = df[df["arch"] == ARCH]
summary = model_df.groupby("increment").agg({{
    "RMSE": ["mean", "std"],
    "RMSE_clean": ["mean", "std"],
    "MAE_clean": ["mean", "std"],
    "growth_max": ["mean"],
    "growth_p99": ["mean"]
}})
print(summary)
"""
        ),
        code(
            f"""
print(f"\\n=== PAIRED DIFFERENCES AGAINST UNCONSTRAINED BASE ({{ARCH}}) ===")
base_clean = model_df[model_df["increment"] == "base"].set_index(["origin", "seed"])["RMSE_clean"]

for arm in ["envelope", "spatial", "composite", "outbreak_aware"]:
    sub = model_df[model_df["increment"] == arm].set_index(["origin", "seed"])["RMSE_clean"]
    common = sorted(set(base_clean.index) & set(sub.index))
    if common:
        diffs = sub.loc[common] - base_clean.loc[common]
        _, pval = stats.ttest_rel(sub.loc[common], base_clean.loc[common])
        print(f"  {{arm:15s}} dRMSE_clean: {{diffs.mean():+6.2f}} (better in {{(diffs < 0).sum()}}/{{len(common)}}, p={{pval:.4f}})")
"""
        ),
        code(
            f"""
# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

arms = ["base", "envelope", "spatial", "composite", "outbreak_aware"]
clean_means = [model_df[model_df["increment"] == a]["RMSE_clean"].mean() for a in arms]
gmax_means = [model_df[model_df["increment"] == a]["growth_max"].mean() for a in arms]

# Panel 1: Clean RMSE
bars = ax1.bar(arms, clean_means, color=["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"], alpha=0.85)
ax1.axhline(p_floor_clean, color="red", linestyle="--", label=f"Persistence Floor ({{p_floor_clean:.2f}})")
ax1.set_ylabel("Clean RMSE (Lower is better)")
ax1.set_title(f"{{ARCH}}: Artifact-Free Clean RMSE")
ax1.legend()
ax1.grid(axis="y", linestyle=":", alpha=0.6)
ax1.tick_params(axis="x", rotation=20)

# Panel 2: Max Growth Rate
ax2.bar(arms, gmax_means, color=["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"], alpha=0.85)
ax2.set_ylabel("Max Log-Growth Rate (Higher = more responsive)")
ax2.set_title(f"{{ARCH}}: Outbreak Growth Responsiveness")
ax2.grid(axis="y", linestyle=":", alpha=0.6)
ax2.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.show()
"""
        ),
    ]
