"""Cell sources for the physics-informed sweep kernel on Kaggle.

Evaluates whether relaxed biological envelope and spatial regularizers improve
over baseline and adaptive architectures across multiple seeds and origins.
Emits:
- reproduction/kaggle/kernels/physics-informed-sweep-dengue-gnn/physics_informed_sweep_dengue_gnn.ipynb
- reproduction/notebooks/physics_informed_sweep_dengue_gnn.ipynb
"""

import env_setup
from gen_kernels import code, md

BRANCH = "feat/physics-informed-loss"

SWEEP_DRIVER = r'''
import itertools, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

LIB = Path(sys.argv[1])
DATA = Path(sys.argv[2])
OUT = Path(sys.argv[3])
sys.path.insert(0, str(LIB))

import adaptive as base
import improved as imp
import reproduced as arch
import physics_loss as ploss
import physics_net as pnet

ARCHS = ["STGAT", "AAGCN"]
ARMS = ["base", "envelope", "spatial", "composite", "outbreak_aware"]
SEEDS = (0, 1, 2)
EPOCHS, WINDOW, HORIZON = 150, 3, 3

cases, adjacency, _ = base.load_dataset(
    DATA / "sri_lanka_2013-2022_shifted.npy", DATA / "sri_lanka_adj_list.json"
)
src, dst = np.nonzero(adjacency)
edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
adj_dense = torch.tensor(adjacency, dtype=torch.float32)
folds = base.build_folds(cases, WINDOW, HORIZON)

ARTIFACT = {f.origin: imp.artifact_windows(f.test_index, WINDOW, HORIZON) for f in folds}
for f in folds:
    print(f"origin {f.origin}: {int(ARTIFACT[f.origin].sum())}/{len(f.test_index)} "
          f"test windows touch week {imp.ARTIFACT_WEEK}", flush=True)

records = [
    dict(
        arch="persistence",
        increment="base",
        origin=f.origin,
        seed=-1,
        **base.persistence_scores(f, cases, HORIZON, ARTIFACT[f.origin])
    )
    for f in folds
]


def run_fold(arch_name, mode, fold, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    kwargs = {"adaptive": False, "channels": 8} if arch_name == "AAGCN" else {}
    backbone = arch.build(arch_name, 25, WINDOW, HORIZON, edge_index=edge_index, **kwargs)
    district_scales = torch.tensor(cases[fold.train_index].mean(axis=0), dtype=torch.float32)

    net = pnet.RelaxedPhysicsNet(
        backbone=backbone,
        edge_index=edge_index,
        adj_dense=adj_dense,
        district_scales=district_scales,
        mode=mode,
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)

    n = len(fold.x_train)
    best, best_val, waited = None, float("inf"), 0
    hist_train = torch.tensor(cases[fold.train_index], dtype=torch.float32)

    for epoch in range(EPOCHS):
        net.train()
        order = torch.randperm(n)
        for s in range(0, n, 32):
            idx = order[s : s + 32]
            opt.zero_grad()
            out = net(fold.x_train[idx])
            pred_y = out + fold.p_train[idx]
            y_curr = fold.x_train[idx, -1, :, 0]
            loss, _ = net.compute_loss(
                pred=out,
                target=fold.y_train[idx] - fold.p_train[idx],
                y_curr=y_curr,
                y_next=pred_y,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

        net.eval()
        with torch.no_grad():
            val_out = net(fold.x_val)
            val_loss = nn.functional.mse_loss(val_out, fold.y_val - fold.p_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= 30:
                break

    if best:
        net.load_state_dict(best)

    net.eval()
    with torch.no_grad():
        test_out = net(fold.x_test)
        pred_z = test_out + fold.p_test

    pred = fold.inverse(pred_z.numpy())
    truth = fold.inverse(fold.y_test.numpy())
    scores = base.pooled_scores(pred, truth, ARTIFACT[fold.origin])

    hist_test = cases[fold.test_index]
    log_h = np.log1p(np.maximum(hist_test[:, -1:, :], 0.0))
    log_p = np.log1p(np.maximum(pred, 0.0))
    log_traj = np.concatenate([log_h, log_p], axis=-1)
    dlog = np.diff(log_traj, axis=-1)
    scores["growth_max"] = float(np.max(dlog))
    scores["growth_p99"] = float(np.percentile(dlog, 99))
    scores["growth_min"] = float(np.min(dlog))
    return scores


started = time.time()
total_runs = len(ARCHS) * len(ARMS) * len(folds) * len(SEEDS)
curr = 0

for arch_name in ARCHS:
    for arm, fold, seed in itertools.product(ARMS, folds, SEEDS):
        curr += 1
        t0 = time.time()
        try:
            sc = run_fold(arch_name, arm, fold, seed)
        except Exception as exc:
            print(f"[{curr}/{total_runs}] {arch_name:7s} {arm:14s} o{fold.origin} s{seed} FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        records.append(dict(arch=arch_name, increment=arm, origin=fold.origin, seed=seed, **sc))
        print(
            f"[{curr}/{total_runs}] {arch_name:7s} {arm:14s} o{fold.origin} s{seed} "
            f"RMSE {sc['RMSE']:7.2f}  clean {sc['RMSE_clean']:6.2f}  gmax {sc['growth_max']:5.2f}  ({time.time()-t0:.0f}s)",
            flush=True
        )
        OUT.write_text(json.dumps(records, indent=1), encoding="utf-8")

csv_out = OUT.with_suffix(".csv")
pd.DataFrame(records).to_csv(csv_out, index=False)
print(f"\n{time.time()-started:.0f}s -> {OUT} and {csv_out} ({len(records)} records)")
'''


def build() -> list:
    """Cells for the physics-informed sweep kernel."""
    return [
        md(
            """
# Physics-Informed Loss vs. Adaptive GNNs — Full Scale Sweep

This Kaggle kernel settles the question from Contribution (c) of the proposal:
**Can physics-informed regularizers (relaxed SEIR-SEI envelope and spatial Dirichlet energy) improve spatial epidemic forecasting over unconstrained and adaptive GNNs?**

## Experimental Design
- **Architectures**: STGAT and AAGCN.
- **Arms under test**:
  1. `base`: Unconstrained baseline.
  2. `envelope`: Biological growth ceiling ($r_{\\text{max}} = 2.3884$) and host clearance lower bounds.
  3. `spatial`: District-normalized spatial flux smoothness.
  4. `composite`: Envelope + spatial flux + mass non-negativity.
  5. `outbreak_aware`: Composite + asymmetric under-prediction weighting ($w_{\\text{under}} = 2.5$).
- **Protocol**: Frozen rolling-origin evaluation over 3 origins ($0.55, 0.70, 0.85$) $\\times$ 3 seeds ($0, 1, 2$) = 9 runs per arm.
- **Evaluation**: Both all-windows RMSE and artifact-free clean RMSE (dropping the 6 week-395 backlog windows).
"""
        ),
        md("## 1. Environment Bootstrap"),
        code(env_setup.PREAMBLE),
        code(env_setup.SETUP),
        code(env_setup.VERIFY_ENV),
        md("## 2. Clone Repository with Physics Modules"),
        code(
            f"""
PROJECT = "https://github.com/rathishTharusha/dengue-forecasting-gnn.git"
BRANCH = "{BRANCH}"
PROJ = SCRATCH / "project"

if not PROJ.exists():
    sh("git", "clone", "--depth", "1", "--branch", BRANCH, PROJECT, str(PROJ))
print("branch:", BRANCH)

DATA = PROJ / "notebooks" / "baseline"
LIB = PROJ / "analysis" / "lib"
print("data:", sorted(p.name for p in DATA.iterdir() if p.suffix in {{".npy", ".json"}}))
print("lib :", sorted(p.name for p in LIB.iterdir() if p.suffix == ".py"))

RESULTS_JSON = WORK / "physics_sweep_results.json"
"""
        ),
        md("## 3. Run the Physics-Informed Sweep"),
        code(
            f'''
script = SCRATCH / "run_physics_sweep.py"
script.write_text(r"""{SWEEP_DRIVER}""", encoding="utf-8")
print(f"wrote {{len(script.read_text().splitlines())}} lines to {{script.name}}")
'''
        ),
        code(
            """
started = time.time()
proc = subprocess.Popen(
    [str(PY311), "-u", str(script), str(LIB), str(DATA), str(RESULTS_JSON)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
for line in proc.stdout:
    print(line, end="")
proc.wait()
print(f"\\nsweep exited {proc.returncode} after {(time.time()-started)/60:.1f} min")
"""
        ),
        md("## 4. Analysis and Pooled Tables"),
        code(
            """
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

records = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
df = pd.DataFrame(records)

print("=== POOLED SUMMARY ACROSS ALL RUNS ===")
p_floor = df[df["arch"] == "persistence"]["RMSE_clean"].mean()
print(f"Persistence Floor (Clean RMSE): {p_floor:.2f}\\n")

summary = df[df["arch"] != "persistence"].groupby(["arch", "increment"]).agg({
    "RMSE": ["mean", "std"],
    "RMSE_clean": ["mean", "std"],
    "MAE_clean": ["mean", "std"],
    "growth_max": ["mean"]
})
print(summary)
"""
        ),
        code(
            """
print("\\n=== PAIRED DIFFERENCES AGAINST UNCONSTRAINED BASE ===")
for arch_name in ["STGAT", "AAGCN"]:
    print(f"\\n--- {arch_name} ---")
    sub = df[df["arch"] == arch_name]
    base_rows = sub[sub["increment"] == "base"].set_index(["origin", "seed"])["RMSE_clean"]
    for arm in ["spatial", "composite", "outbreak_aware"]:
        arm_rows = sub[sub["increment"] == arm].set_index(["origin", "seed"])["RMSE_clean"]
        common = sorted(set(base_rows.index) & set(arm_rows.index))
        if common:
            diffs = arm_rows.loc[common] - base_rows.loc[common]
            _, pval = stats.ttest_rel(arm_rows.loc[common], base_rows.loc[common])
            print(f"  {arm:15s} dRMSE_clean: {diffs.mean():+6.2f} (better in {(diffs < 0).sum()}/{len(common)}, p={pval:.4f})")
"""
        ),
        code(
            """
# Bar comparison visualization
fig, ax = plt.subplots(figsize=(10, 5))
summary_clean = df[df["arch"] == "STGAT"].groupby("increment")["RMSE_clean"].mean()
summary_clean.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"], alpha=0.85)
ax.axhline(p_floor, color="red", linestyle="--", label=f"Persistence Floor ({p_floor:.2f})")
ax.set_ylabel("Clean RMSE (Lower is better)")
ax.set_title("STGAT Physics Arms vs. Persistence Floor")
ax.legend()
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
"""
        ),
    ]
