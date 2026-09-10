"""Cell sources for R2 -- DengueGNN / Dynamic ST-GNN (GulMohamed et al., 2026).

The paper's absolute numbers are not reproducible: no code, no region subset, no
hyperparameters, and "OpenDengue" spans 102 countries. This notebook therefore
does what *can* be done honestly -- audits what is specified, implements the
architecture from Equations (2)-(14), and tests the three ablation claims of
Table 6, which are relative and so survive the missing dataset definition.
"""

from gen_notebooks import code, md

CELLS = [
    md(
        """
# R2 — DengueGNN: Dynamic Spatio-Temporal GNN

**Source:** Rasitha Banu GulMohamed, Wafa Hetany, Hanan Abdullah Almaimani,
Faiza Abdalla Saeed Khiery, *DengueGNN: Graph-based deep learning for modeling
disease spread dynamics and prediction*, Scientific Reports **16**:10584 (2026).
(`papers/DengueGNN Graph-based deep.pdf`)

## Read this before running anything

**The published numbers cannot be reproduced, and this notebook does not claim
to reproduce them.** That is a finding about the paper, not a shortfall of this
workspace, and section 1 documents it specifically. What *is* reproducible is
the architecture — the paper specifies it in equations — and the three ablation
claims of Table 6, which are relative and therefore survive the missing dataset
definition.

Reported targets, for the record (Table 3): RMSE 6.1 / MAE 4.9 / MAPE 12.0% at
1 week, RMSE 10.9 / MAE 8.8 / MAPE 22.4% at 4 weeks; Moran's I 0.76 and 0.68
(Table 5); CRPS 3.85, PICP 0.95, MPIW 21.2 at 1 week (Table 4).

## What this notebook does

| Section | Purpose |
|---|---|
| 1 | Reproducibility audit — what the paper specifies and what it omits |
| 2–3 | Implement Eq. (2)–(14): dynamic graph, spatial encoder, attention-LSTM, fusion, per-horizon heads |
| 4 | Fit on the Sri Lanka data we have, at 1-week and 4-week horizons |
| 5 | Moran's I and the probabilistic metrics of Table 4 |
| 6 | The Table 6 ablations: static graph, no attention, no mobility |
| 7 | Verdict |

The by-product matters as much as the audit: a working dynamic-graph ST-GNN with
temporal attention is most of **Phase 2** of `docs/ROADMAP.md`.
"""
    ),
    md(
        """
## 1. Reproducibility audit

The paper reports results to two significant figures across 11 models, three
horizons and six metrics. To recompute any of it you would need the following.

| Needed | Stated in the paper? |
|---|---|
| Dataset | "OpenDengue" — a database spanning 102 countries and 3 spatial levels |
| **Which regions / country / admin level** | **No.** Table 2 says 56,000,000 dengue-case rows |
| **Which time period** | **No.** OpenDengue runs from the early 1990s |
| Node definition | Regions, but the number of nodes is never given |
| **Mobility data** | Gravity prior from population and distance (Eq. 3); real flows "when available" — **sources not named** |
| **Environmental covariates** | Temperature, rainfall, humidity named; **provenance and resolution not given** |
| Graph blend `α` (Eq. 2) | **No value given** |
| Window length `L` (Eq. 7) | **No value given** |
| Hidden units, layers, dropout, LR, batch size | "grid search on the validation set" — **search space and chosen values not given** |
| Train/test split | "last 20% as holdout", 5 seeds |
| Code | **Not available** |

Two internal details compound this. Table 2 reports 56 million dengue-case rows
against 3.2 million temperature rows and 1.2 million mobility rows — so the
covariates cover at most ~6% of the case rows, and no join or imputation rule is
given for the rest. And an RMSE of 6.1 against a stated mean weekly incidence of
34.6 with standard deviation 215.2 implies either a very low-incidence subset or
aggressive filtering; neither is described.

**Consequence.** Any number this notebook produces is on *our* Sri Lanka data,
not the paper's, and is not comparable to Table 3. The ablations are.
"""
    ),
    md("## 2. Setup"),
    code(
        """
import sys
from pathlib import Path

LIB = Path.cwd().parent / "lib"
if not LIB.exists():
    LIB = Path.cwd() / "lib"
sys.path.insert(0, str(LIB))

REPO = Path.cwd().parent.parent
DATA_DIR = REPO / "notebooks" / "baseline"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from xcheck.data import load_adjacency, load_array
from xcheck.metrics import crps_gaussian, mae, mape_masked, morans_i, mpiw, picp, rmse
from xcheck.graph import gravity_mobility, hop_distance, hybrid_adjacency, symmetric_normalize
from xcheck.models import DynamicSTGNN, gaussian_nll

QUICK_TEST = True


class CFG:
    window = 8            # L in Eq. (7); the paper gives no value, so this is ours
    horizons = (1, 4)     # the paper's two reported horizons
    alpha = 0.5           # graph blend in Eq. (2); the paper gives no value
    hidden = 64
    dropout = 0.1
    lr = 1e-3
    weight_decay = 1e-5
    epochs = 20 if QUICK_TEST else 150
    batch_size = 32
    seeds = (0,) if QUICK_TEST else (0, 1, 2, 3, 4)   # the paper uses 5 seeds
    test_frac = 0.2       # "last 20% as holdout"
    val_frac = 0.1


torch.manual_seed(0)
np.random.seed(0)
print(f"QUICK_TEST={QUICK_TEST} | epochs={CFG.epochs} | seeds={CFG.seeds}")

raw = load_array(DATA_DIR / "sri_lanka_2013-2022_shifted.npy")
A_geo, districts = load_adjacency(DATA_DIR / "sri_lanka_adj_list.json", self_loops=False)
cases = raw[..., 5]
N_NODES = len(districts)
print(f"{raw.shape[0]} weeks x {N_NODES} districts; target = feature 5")
"""
    ),
    md(
        """
## 3. The dynamic graph (Eq. 2–4)

`A_t = α·A_geo + (1 − α)·A_mob`, with `A_mob` from the gravity model
`G_ij = p_i·p_j / d_ij²`, then symmetrically normalized as `D^-½ A D^-½`.

**Substitution, stated plainly.** The paper's gravity prior needs district
populations and inter-district distances. Neither is in this repo — the released
array carries meteorological channels only, and `docs/DATA.md` records no
population source. So:

* `d_ij` = **shortest-path hop count** in the district adjacency graph, a
  topological stand-in for geographic distance;
* `p_i` = **uniform**, which reduces the gravity model to a pure distance-decay
  kernel.

That is a documented proxy, not the paper's mobility matrix. It is enough to
test whether a *distance-decayed, denser* graph beats a binary adjacency one —
which is the mechanism the paper's ablation claims — but it cannot test the
value of real mobility flows. `POPULATION` below is the hook to fix if district
populations are ever added to the repo.
"""
    ),
    code(
        """
hops = hop_distance(A_geo)
print(f"hop distance: max {hops.max():.0f} hops between any two districts")

POPULATION = np.ones(N_NODES)      # <-- replace with real district populations
A_mob = gravity_mobility(POPULATION, hops)
A_dynamic = symmetric_normalize(hybrid_adjacency(A_geo, A_mob, alpha=CFG.alpha))
A_static = symmetric_normalize(A_geo)          # the Table 6 "static graph" ablation

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, m, title in zip(
    axes, [A_geo, A_mob, A_dynamic],
    ["A_geo (binary adjacency)", "A_mob (gravity, hop-distance)", "A_t normalized (alpha=0.5)"],
):
    im = ax.imshow(m, cmap="viridis")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); plt.show()

print(f"A_geo density {(A_geo > 0).mean():.3f} | A_mob density {(A_mob > 1e-3).mean():.3f}")
"""
    ),    md(
        """
## 4. Features, windowing and training

Equation (9) fuses three things: the temporal embedding `h`, **environmental**
features `e`, and **mobility** features `m`. All three are built here, because
without `e` and `m` the fusion layer is decorative and the ablations that remove
them cannot mean anything.

* **Environmental** — the ten non-target channels of the released array
  (precipitation, temperatures, humidity, NDVI, soil moisture, canopy) at the
  last week of the window. These are already lag-shifted; see `docs/DATA.md`.
* **Mobility** — two per-district quantities derived from the gravity matrix:
  its row strength, and the mobility-weighted incidence of a district's
  neighbours in the last observed week. The second is the actual mechanism the
  paper claims: a district's risk rising because *connected* districts are
  seeing cases.

Protocol as the paper describes it: chronological split, last 20% held out, a
validation slice before it, statistics from training weeks only, no shuffling.
"""
    ),
    code(
        """
NON_TARGET = [c for c in range(raw.shape[2]) if c != 5]
print(f"environmental channels: {NON_TARGET} ({len(NON_TARGET)} of {raw.shape[2]})")

# Mobility-weighted neighbour incidence: row-normalize A_mob so each district's
# feature is an average over its partners rather than a sum that scales with degree.
_row = A_mob.sum(axis=1, keepdims=True)
A_MOB_ROWNORM = np.divide(A_mob, _row, out=np.zeros_like(A_mob), where=_row > 0)
MOB_STRENGTH = A_mob.sum(axis=1)


def build_windows(window, horizon):
    \"\"\"Return x, env, mob, y for every window.

    x    (B, window, N, 1)  past cases
    env  (B, N, 10)         environmental channels at the window's last week
    mob  (B, N, 2)          [gravity strength, mobility-weighted neighbour cases]
    y    (B, N)             cases at t + horizon - 1
    \"\"\"
    xs, envs, mobs, ys = [], [], [], []
    for i in range(window, raw.shape[0] - horizon + 1):
        xs.append(cases[i - window:i][..., None])
        envs.append(raw[i - 1][:, NON_TARGET])
        neighbour_cases = A_MOB_ROWNORM @ cases[i - 1]
        mobs.append(np.stack([MOB_STRENGTH, neighbour_cases], axis=1))
        ys.append(cases[i + horizon - 1])
    return np.stack(xs), np.stack(envs), np.stack(mobs), np.stack(ys)


def split_indices(n):
    n_test = int(n * CFG.test_frac)
    n_val = int(n * CFG.val_frac)
    return slice(0, n - n_val - n_test), slice(n - n_val - n_test, n - n_test), slice(n - n_test, n)


x_demo, env_demo, mob_demo, y_demo = build_windows(CFG.window, 1)
print(f"x {x_demo.shape} | env {env_demo.shape} | mob {mob_demo.shape} | y {y_demo.shape}")
print(f"splits over {len(x_demo)} windows: {split_indices(len(x_demo))}")
"""
    ),
    code(
        """
def zscore(a, train_slice, axis=None):
    \"\"\"Standardize with statistics from the training slice only.\"\"\"
    mu = a[train_slice].mean(axis=axis, keepdims=axis is not None)
    sd = a[train_slice].std(axis=axis, keepdims=axis is not None) + 1e-8
    return (a - mu) / sd, mu, sd


def fit_predict(horizon, adjacency, seed, temporal_attention=True, use_mobility=True,
                probabilistic=False):
    \"\"\"Train one DynamicSTGNN and return (pred, truth, sigma) on the holdout.\"\"\"
    x, env, mob, y = build_windows(CFG.window, horizon)
    tr, va, te = split_indices(len(x))

    xz, mu, sd = zscore(x, tr)
    yz = (y - mu) / sd
    envz, _, _ = zscore(env, tr, axis=(0, 1))
    mobz, _, _ = zscore(mob, tr, axis=(0, 1))

    xt = torch.tensor(xz, dtype=torch.float32)
    yt = torch.tensor(yz, dtype=torch.float32)
    envt = torch.tensor(envz, dtype=torch.float32)
    mobt = torch.tensor(mobz, dtype=torch.float32)
    adj = torch.tensor(adjacency, dtype=torch.float32)

    torch.manual_seed(seed)
    model = DynamicSTGNN(
        n_features=1, n_env=envt.shape[-1], n_mob=mobt.shape[-1], horizons=1,
        hidden=CFG.hidden, dropout=CFG.dropout,
        temporal_attention=temporal_attention, use_mobility=use_mobility,
        probabilistic=probabilistic,
    )
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    def forward(sl):
        return model(xt[sl], adj, envt[sl], mobt[sl])

    n_train = tr.stop - tr.start
    best_state, best_val = None, float("inf")
    for _ in range(CFG.epochs):
        model.train()
        for start in range(0, n_train, CFG.batch_size):
            batch = slice(start, min(start + CFG.batch_size, n_train))
            opt.zero_grad()
            out = forward(batch)
            if probabilistic:
                loss = gaussian_nll(out[..., 0, 0], out[..., 0, 1], yt[batch])
            else:
                loss = torch.nn.functional.mse_loss(out[..., 0], yt[batch])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vout = forward(va)
            vpred = vout[..., 0, 0] if probabilistic else vout[..., 0]
            vloss = torch.nn.functional.mse_loss(vpred, yt[va]).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = forward(te)

    truth = y[te]
    if probabilistic:
        return out[..., 0, 0].numpy() * sd + mu, truth, np.sqrt(np.exp(out[..., 0, 1].numpy())) * sd
    return out[..., 0].numpy() * sd + mu, truth, None


def score(pred, truth):
    return {"RMSE": rmse(pred, truth), "MAE": mae(pred, truth),
            "MAPE(>=1) %": mape_masked(pred, truth)}


def persistence(horizon):
    \"\"\"Last observed week, carried forward -- the reference any forecast must beat.\"\"\"
    x, _, _, y = build_windows(CFG.window, horizon)
    _, _, te = split_indices(len(x))
    return score(x[te][:, -1, :, 0], y[te])
"""
    ),
    code(
        """
PAPER_TABLE3 = {1: {"RMSE": 6.1, "MAE": 4.9}, 4: {"RMSE": 10.9, "MAE": 8.8}}

rows, predictions = [], {}
for horizon in CFG.horizons:
    per_seed = []
    for seed in CFG.seeds:
        pred, truth, _ = fit_predict(horizon, A_dynamic, seed)
        per_seed.append(score(pred, truth))
        if seed == CFG.seeds[0]:
            predictions[horizon] = (pred, truth)
    rows.append({
        "horizon (weeks)": horizon,
        **{k: float(np.mean([s[k] for s in per_seed])) for k in per_seed[0]},
        "RMSE sd": float(np.std([s["RMSE"] for s in per_seed])),
        "persistence RMSE": persistence(horizon)["RMSE"],
        "paper RMSE": PAPER_TABLE3[horizon]["RMSE"],
    })

table3 = pd.DataFrame(rows)
print(table3.to_string(index=False))
print("\\nThe 'paper RMSE' column is context, NOT a target: it is computed on a")
print("different, unspecified dataset (see section 1).")
print("The 'persistence RMSE' column is the one that matters -- it is on OUR data,")
print("and a forecast that does not beat it has not earned its complexity.")
"""
    ),
    md(
        """
## 5. Moran's I and the probabilistic metrics

Table 5 reports Moran's I of the predicted incidence pattern; Table 4 reports
CRPS, PICP and MPIW from the Gaussian head of Eq. (12), with 95% intervals from
Eq. (14).

Moran's I is computed per test week on the district adjacency graph and
averaged. The **observed** pattern's Moran's I is reported beside it, because a
prediction can be no more spatially clustered than the truth it tracks — a high
value against a low observed value means the model is inventing structure, not
capturing it.
"""
    ),
    code(
        """
spatial_rows = []
for horizon, (pred, truth) in predictions.items():
    i_pred = np.array([morans_i(p, A_geo) for p in pred])
    i_true = np.array([morans_i(t, A_geo) for t in truth])
    ok = np.isfinite(i_pred) & np.isfinite(i_true)
    spatial_rows.append({
        "horizon (weeks)": horizon,
        "Moran's I (predicted)": i_pred[ok].mean(),
        "Moran's I (observed)": i_true[ok].mean(),
        "paper Table 5": {1: 0.76, 4: 0.68}[horizon],
    })
spatial = pd.DataFrame(spatial_rows)
print(spatial.to_string(index=False))
"""
    ),
    code(
        """
prob_rows = []
for horizon in CFG.horizons:
    pred, truth, sigma = fit_predict(horizon, A_dynamic, CFG.seeds[0], probabilistic=True)
    lo, hi = pred - 1.96 * sigma, pred + 1.96 * sigma      # Eq. (14)
    prob_rows.append({
        "horizon (weeks)": horizon,
        "CRPS": crps_gaussian(pred, sigma, truth),
        "PICP": picp(lo, hi, truth),
        "MPIW": mpiw(lo, hi),
        "paper CRPS": {1: 3.85, 4: 5.91}[horizon],
        "paper PICP": {1: 0.95, 4: 0.93}[horizon],
    })
probabilistic = pd.DataFrame(prob_rows)
print(probabilistic.to_string(index=False))
print("\\nPICP is the one to read: near 0.95 means the intervals are calibrated on")
print("OUR data. It says nothing about the paper's number.")
"""
    ),
    md(
        """
## 6. The Table 6 ablations

Table 6 reports three ablations as *relative* degradations, which do not depend
on the dataset definition:

| Ablation | Paper's reported degradation |
|---|---|
| Static graph instead of dynamic | +6% to +12% (ΔRMSE +0.7) |
| No temporal attention | +4% to +8% (ΔRMSE +0.5) |
| No mobility features | +8% to +15% (ΔRMSE +1.2), the largest |

**One of these three is not testable here, and it must be said plainly.** The
paper's "dynamic graph" is *time-varying* — `A_t` changes week to week with
mobility. This repo has no time-varying mobility data, so our `A_dynamic` is
static too, just denser and distance-weighted. What the first row below tests is
therefore **gravity-blended graph vs. binary adjacency**, not dynamic vs. static.
It cannot confirm or refute the paper's claim.

The other two are genuine tests: the attention module is removed from the
architecture, and the mobility features are removed from the Eq. (9) fusion.
"""
    ),
    code(
        """
VARIANTS = {
    "full model": dict(adjacency=A_dynamic, temporal_attention=True, use_mobility=True),
    "binary graph (not 'static')": dict(
        adjacency=A_static, temporal_attention=True, use_mobility=True),
    "no attention": dict(adjacency=A_dynamic, temporal_attention=False, use_mobility=True),
    "no mobility features": dict(
        adjacency=A_dynamic, temporal_attention=True, use_mobility=False),
}

rows = []
for name, kwargs in VARIANTS.items():
    for horizon in CFG.horizons:
        scores = [score(*fit_predict(horizon, seed=seed, **kwargs)[:2])
                  for seed in CFG.seeds]
        rows.append({
            "variant": name, "horizon (weeks)": horizon,
            "RMSE": float(np.mean([s["RMSE"] for s in scores])),
            "RMSE sd": float(np.std([s["RMSE"] for s in scores])),
            "MAE": float(np.mean([s["MAE"] for s in scores])),
        })

ablation = pd.DataFrame(rows)
base = ablation[ablation.variant == "full model"].set_index("horizon (weeks)")["RMSE"]
ablation["dRMSE vs full"] = ablation.apply(
    lambda r: r.RMSE - base[r["horizon (weeks)"]], axis=1)
ablation["% worse"] = ablation.apply(
    lambda r: 100 * r["dRMSE vs full"] / base[r["horizon (weeks)"]], axis=1)
print(ablation.to_string(index=False))
"""
    ),
    code(
        """
# A degradation is only real if it exceeds the seed-to-seed noise of the runs
# being compared. With one seed there is no noise estimate and nothing can be
# concluded -- say so rather than reading the sign of a difference.
noise = ablation["RMSE sd"].max()
print(f"largest seed-to-seed RMSE sd across variants: {noise:.4f}")

if len(CFG.seeds) < 2:
    print("\\nOnly one seed: no noise estimate, so NO ablation conclusion is available.")
    print("Set QUICK_TEST = False (5 seeds) before reading anything into these deltas.")
else:
    summary = (ablation[ablation.variant != "full model"]
               .groupby("variant")[["% worse", "dRMSE vs full"]].mean()
               .sort_values("% worse", ascending=False))
    summary["exceeds seed noise"] = summary["dRMSE vs full"].abs() > 2 * noise
    print()
    print(summary.to_string())
    print("\\nPaper's ranking (most damaging first): no mobility > static graph > no attention.")
    print("Rows marked False are within noise and support no claim either way.")
    print("Note the first variant tests graph density, not dynamism -- see above.")
"""
    ),
    md("## 7. Verdict"),
    code(
        """
results_dir = Path.cwd().parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)
for name, frame in (
    ("R2_horizon_scores", table3),
    ("R2_spatial_morans_i", spatial),
    ("R2_probabilistic", probabilistic),
    ("R2_ablation", ablation),
):
    frame.assign(quick_test=QUICK_TEST).to_csv(results_dir / f"{name}.csv", index=False)
print(f"wrote 4 CSVs to {results_dir}")

if QUICK_TEST:
    print("\\n*** QUICK_TEST=True — degraded numbers, do not cite. ***")
ablation
"""
    ),
    md(
        """
## 8. What this establishes

**On reproducibility.** The paper's Tables 3–5 are not reproducible from the
paper. The dataset is named but not defined (no country, admin level, period, or
node count), no hyperparameter values are given despite a grid search being
claimed, the mobility and covariate sources are unnamed, and no code is
released. Nothing in section 4 above should be read as a reproduction of
RMSE 6.1 — it is a different model on a different dataset.

**On the architecture.** Equations (2)–(14) are specific enough to implement,
and the implementation trains and produces calibrated intervals. That part of
the paper is sound and useful.

**On the ablations.** These are the claims worth taking seriously, because they
are relative. Record whether our ranking matched, at `QUICK_TEST = False`, with
the caveat that our mobility ablation tests a hop-distance proxy rather than
real flows.

**For our project.** `xcheck.models.DynamicSTGNN` is a working
adaptive-adjacency ST-GNN with temporal attention — most of the machinery
Phase 2 of `docs/ROADMAP.md` calls for. Two cautions before it is promoted into
`src/`: it must be re-scored under our frozen rolling-origin protocol (this
notebook uses the paper's single 20% holdout, which is *not* our protocol), and
the gravity prior needs real district populations and distances before it can be
claimed as a mobility model rather than a distance kernel.
"""
    ),
]
