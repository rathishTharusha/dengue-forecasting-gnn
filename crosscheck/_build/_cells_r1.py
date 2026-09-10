"""Cell sources for R1 -- Weng et al. (2024), the benchmark this project reproduces.

Reproduction target: Table I, which is also reproduced verbatim in the authors'
own ``reference_repo/Models/results.txt``. The notebook reproduces the published
protocol exactly, then re-runs the same model under a corrected protocol so the
difference is measured rather than argued.
"""

from gen_notebooks import code, md

CELLS = [
    md(
        """
# R1 — Graph Representation Learning for Dengue Forecasting

**Source:** Jiaqi Weng, David Qiu, Ethan Cruz, Malik Magdon-Ismail, Thilanka
Munasinghe, Jennifer C. Wei, Ashan Pathirana, *Graph Representation Learning for
Dengue Forecasting*, IEEE BigData 2024.
(`papers/GraphRepresentation_Dengue_IEEEbigData2024.pdf`)

**Why this one matters most.** This is the paper our Phase-1 baseline reproduces
and whose dataset we consume as given. If its Table I is not what it appears to
be, the number our own baseline is measured against moves. That is the question
this notebook exists to settle.

We are unusually well placed to answer it: the authors' implementation is
available at `reference_repo/`, and `reference_repo/Models/results.txt` contains
the raw output that became Table I. So the target is not a guess.

## Reproduction targets — Table I, shifted dataset

| Model | CV MAE | CV RMSE | Full MAE | Full RMSE |
|---|---|---|---|---|
| ARIMA | 168.37 ± 187.86 | 189.22 ± 181.52 | 133.41 | 142.22 |
| Random Forest | 48.91 ± 11.95 | 84.66 ± 19.22 | 34.52 | 64.42 |
| XGBoost | 53.89 ± 10.76 | 95.22 ± 17.22 | 40.77 | 75.44 |
| ARNN | 82.46 ± 16.67 | 106.83 ± 26.14 | 56.38 | 70.16 |
| LSTM | 81.19 ± 17.95 | 131.36 ± 30.81 | 49.19 | 81.88 |
| DCRNN | 45.98 ± 2.34 | 71.61 ± 2.56 | 26.20 | 36.30 |
| **STGAT** | **25.38 ± 1.37** | **44.78 ± 2.26** | 22.28 | 42.65 |
| AAGCN | 41.93 ± 1.11 | 55.83 ± 1.98 | 39.27 | 49.20 |
| A3TGCN | 34.03 ± 1.10 | 58.52 ± 1.87 | 18.65 | 30.55 |
| ASTGCN | 33.68 ± 19.85 | 47.72 ± 18.12 | 20.62 | 32.41 |

**Headline claim under test:** "GNNs consistently outperform the baseline in
this dataset in both cross validation and no validation", average RMSE 38.22 for
GNNs against 70.65 for baselines on the shifted data.

## What this notebook does

1. Audits the released implementation against the protocol the paper describes —
   five specific checks, each one a cell you can read and re-run.
2. Reproduces STGAT under the published protocol and compares to 25.38 / 44.78.
3. Re-runs the identical model under a corrected protocol.
4. Reports both, side by side, with the persistence floor for scale.
"""
    ),
    md("## 1. Setup and configuration"),
    code(
        """
import sys
from pathlib import Path

LIB = Path.cwd().parent / "lib"
if not LIB.exists():
    LIB = Path.cwd() / "lib"
sys.path.insert(0, str(LIB))

REPO = Path.cwd().parent.parent          # crosscheck/notebooks -> repo root
DATA_DIR = REPO / "notebooks" / "baseline"
NPY = DATA_DIR / "sri_lanka_2013-2022_shifted.npy"
ADJ = DATA_DIR / "sri_lanka_adj_list.json"

import numpy as np
import pandas as pd
import torch

from xcheck.data import (
    build_edge_index,
    load_adjacency,
    load_array,
    make_windows,
    segment_split,
)
from xcheck.metrics import mae, mape_weng, rmse
from xcheck.protocol import WENG_SEGMENTS, FoldResult, batched_mean_metric, summarize

# Run QUICK_TEST=True first to confirm the pipeline executes. Its numbers are
# intentionally degraded (fewer epochs, fewer segments) and must never be cited.
QUICK_TEST = True


class CFG:
    \"\"\"Reference implementation settings, from reference_repo/Models/evaluation.py.\"\"\"
    window = 3
    horizon = 3
    n_nodes = 25
    self_loops = True
    use_disease_only = True    # the reference runs its GNNs on the case channel alone
    batch_size = 1
    epochs = 10 if QUICK_TEST else 50
    segments = WENG_SEGMENTS[:2] if QUICK_TEST else WENG_SEGMENTS
    # NOTE the discrepancy audited in section 3: the paper states lr=1e-5,
    # weight_decay=1e-6; the released code uses these values instead.
    lr = 1e-4
    weight_decay = 5e-5
    dropout = 0.1
    heads = 8
    hidden = 64
    seed = 0


torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
print(f"QUICK_TEST={QUICK_TEST} | epochs={CFG.epochs} | segments={CFG.segments}")
print("torch", torch.__version__)
"""
    ),
    code(
        """
raw = load_array(NPY)
adjacency, districts = load_adjacency(ADJ, self_loops=CFG.self_loops)
edge_index = torch.tensor(build_edge_index(adjacency))

print(f"array   : {raw.shape}  (weeks, districts, features)")
print(f"graph   : {len(districts)} nodes, {edge_index.shape[1]} directed edges (self-loops "
      f"{CFG.self_loops})")

cases = raw[..., 5]
print(f"\\ntarget (feature 5): median {np.median(cases):.0f}, max {cases.max():.0f}, "
      f"{100 * (cases == 0).mean():.1f}% zeros")
print(f"lag-1 autocorrelation: "
      f"{np.mean([np.corrcoef(cases[:-1, n], cases[1:, n])[0, 1] for n in range(cases.shape[1]) if cases[:, n].std() > 0]):.3f}")
"""
    ),
    md(
        """
### A note on the feature count

The paper describes "20 features at each time-step" (§VII, Modeling Challenges)
and lists precipitation, humidity, land surface temperature, temperature and
canopy cover saved as mean/min/max. The released array has **11**. This does not
affect the GNN reproduction — the released code sets `use_disease_only=True` for
STGAT, A3TGCN, DCRNN and AAGCN, so those models never see the covariates at all —
but it does mean the published feature ablation cannot be reproduced from the
released data.
"""
    ),
    md(
        """
## 2. Audit A — what "Cross Validated" is measured on

The paper (§VI) describes the protocol as: five nested segments, each split
70/30 into train and test, model reinitialized per segment.

The released code does the split, evaluates the test set — and then reports a
different number. From `reference_repo/Models/evaluation.py`, in every one of
`run_stgat`, `run_a3tgcn`, `run_astgcn`, `run_dcrnn`, `run_aagcn`:

```python
y_pred, y_truth, _, _  = infer(model, "cpu", test, m, s, "Test")   # discarded
y_pred, y_truth, ma, rm = infer(model, "cpu", full, m, s, "Full")  # appended
maes += [ma]
rmses += [rm]
```

`full` is a `DataLoader` over `dataset` — every window in the segment, the 70%
the model trained on included. So the "Cross Validated" column of Table I is a
**training-inclusive** error, not a held-out one.

The cell below reproduces the index arithmetic to show how much of the reported
evaluation set the model was fitted on.
"""
    ),
    code(
        """
windows_all = make_windows(
    raw, window=CFG.window, horizon=CFG.horizon,
    use_all_features=not CFG.use_disease_only, normalize="global",
)
n_windows = len(windows_all)


def segment_weeks(frac):
    \"\"\"Weeks and windows in one segment, matching the reference exactly.

    load_data truncates the array to int(T * subset) WEEKS and only then
    windows it, so a segment holds int(T*frac) - window - horizon windows,
    not int(n_windows * frac).
    \"\"\"
    n_weeks = int(raw.shape[0] * frac)
    return n_weeks, n_weeks - CFG.window - CFG.horizon


rows = []
for frac in WENG_SEGMENTS:
    n_weeks, n_seg = segment_weeks(frac)
    train_slice, _, test_slice = segment_split(n_seg, 0.7)
    n_train = train_slice.stop - train_slice.start
    rows.append({
        "segment": frac,
        "weeks": n_weeks,
        "windows in segment": n_seg,
        "trained on": n_train,
        "reported ('full')": n_seg,
        "held-out ('test')": test_slice.stop - test_slice.start,
        "% of reported set seen in training": 100 * n_train / n_seg,
    })

audit_a = pd.DataFrame(rows)
print(audit_a.to_string(index=False))
print(f"\\nAcross all five segments, {audit_a['% of reported set seen in training'].mean():.0f}% "
      "of every reported evaluation set was training data.")
"""
    ),
    md(
        """
## 3. Audit B — the baselines solve a different problem from the GNNs

This is the more consequential finding, because the paper's central claim is a
comparison between the two groups.

From `reference_repo/Models/random_forest.py` (and `arima.py`, `lstm.py`, all
identical in structure):

```python
data = pd.read_csv("../Data/Datasets/MLSO2_Final.csv")
state_data = data[data["region"] == "Kalutara"].reset_index(drop=True)
yc = state_data["cases"]
Xc = state_data.drop([...], axis=1)
...
rf_model.fit(train_X, train_y)
predictions = rf_model.predict(test_X)
```

Three differences from the GNN setup, all in the same direction:

| | Classical baselines | GNNs |
|---|---|---|
| Spatial scope | **Kalutara only** (1 district) | all 25 districts |
| Task | cases at week *t* from covariates at week *t* | 3-step-ahead forecast |
| Input | same-week covariates | 3-week window of past cases |

The baselines are not forecasting. They are fitting current cases from current
covariates in one district, which is a *regression* problem, and their errors
are computed over a single district's case distribution rather than over all 25.

`MLSO2_Final.csv` is not in this repo, so the cell below reconstructs both
framings from the released array — same district, same 70/30 segment scheme —
to show that the framing alone moves the error by more than the gap the paper
attributes to graph structure.
"""
    ),
    code(
        """
from sklearn.ensemble import RandomForestRegressor

KALUTARA = districts.index("Kalutara")
print(f"Kalutara is node {KALUTARA} of {len(districts)}")

cases_all = raw[..., 5]
feats_all = raw                                  # 11 channels

def rf_nowcast_one_district(node, segments):
    \"\"\"Reference framing: cases[t] from covariates[t], one district.\"\"\"
    out = []
    y_full = cases_all[:, node]
    x_full = np.delete(feats_all[:, node, :], 5, axis=1)   # drop the target channel
    for frac in segments:
        n = int(len(y_full) * frac)
        y, x = y_full[:n], x_full[:n]
        cut = int(n * 0.7)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x[:cut], y[:cut])
        pred = model.predict(x[cut:])
        out.append((mae(pred, y[cut:]), rmse(pred, y[cut:])))
    return np.array(out)


def rf_forecast_all_districts(segments):
    \"\"\"GNN framing: 3-step-ahead forecast from a 3-week window, all 25 districts.\"\"\"
    out = []
    w = CFG.window
    h = CFG.horizon
    for frac in segments:
        n_weeks = int(raw.shape[0] * frac)
        idx = np.arange(w, n_weeks - h)
        x = np.stack([cases_all[i - w:i].T.ravel() for i in idx])       # 25*3 window
        y = np.stack([cases_all[i:i + h].T.ravel() for i in idx])       # 25*3 horizon
        cut = int(len(idx) * 0.7)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x[:cut], y[:cut])
        pred = model.predict(x[cut:])
        out.append((mae(pred, y[cut:]), rmse(pred, y[cut:])))
    return np.array(out)


seg = list(CFG.segments)
nowcast = rf_nowcast_one_district(KALUTARA, seg)
forecast = rf_forecast_all_districts(seg)

audit_b = pd.DataFrame([
    {
        "framing": "reference baselines: Kalutara, same-week covariates",
        "MAE": nowcast[:, 0].mean(), "MAE sd": nowcast[:, 0].std(),
        "RMSE": nowcast[:, 1].mean(), "RMSE sd": nowcast[:, 1].std(),
    },
    {
        "framing": "GNN task: 25 districts, 3-step-ahead forecast",
        "MAE": forecast[:, 0].mean(), "MAE sd": forecast[:, 0].std(),
        "RMSE": forecast[:, 1].mean(), "RMSE sd": forecast[:, 1].std(),
    },
])
print(audit_b.to_string(index=False))
print("\\nSame estimator (RandomForestRegressor, 100 trees, seed 42), same segments,")
print("same data. The only change is the task definition.")
"""
    ),
    md(
        """
## 4. Audit C — normalization statistics include the test period

`load_data` in the reference computes the z-score over the whole array before
any split:

```python
x = np.nan_to_num(np.load(data_file, allow_pickle=True))
mean, std = np.mean(x[..., -6]), np.std(x[..., -6])   # whole series
x = z_norm(torch.tensor(x), mean, std)
```

`inverse_z_norm` then uses the same statistics to map predictions back to case
counts at scoring time. On a heavy-tailed series where the largest outbreak sits
in the later years, the location and scale of the test period are therefore known
during training.
"""
    ),
    code(
        """
n_weeks = raw.shape[0]
train_end = int(n_weeks * 0.7)

glob = make_windows(raw, use_all_features=False, normalize="global")
trn = make_windows(raw, use_all_features=False, normalize="train", train_end=train_end)

audit_c = pd.DataFrame([
    {"statistics from": "whole series (reference)", "mean": glob.mean, "std": glob.std},
    {"statistics from": f"training weeks only (first {train_end})", "mean": trn.mean, "std": trn.std},
])
print(audit_c.to_string(index=False))
print(f"\\nscale differs by {100 * abs(glob.std - trn.std) / trn.std:.1f}%, "
      f"location by {100 * abs(glob.mean - trn.mean) / trn.mean:.1f}%")
print("Predictions are inverse-transformed with these values, so the error itself is")
print("expressed in a scale the test period helped define.")
"""
    ),
    md(
        """
## 5. Audit D — stated hyperparameters differ from the released ones

The paper (§II.D, Training Details) states: "trained ... using the Adam
optimizer with a learning rate of 0.00001 and a weight decay of 0.000001".

`reference_repo/Models/evaluation.py` line 26:

```python
lr, decay, dropout = 1e-4, 5e-5, 0.1
```

A 10× learning rate and a 50× weight decay. Reproducing "the paper" and
reproducing "the code" are therefore different experiments; this notebook
follows the code, because the code is what produced Table I.
"""
    ),
    code(
        """
audit_d = pd.DataFrame([
    {"setting": "learning rate", "paper (§II.D)": 1e-5, "released code": 1e-4, "ratio": 10.0},
    {"setting": "weight decay", "paper (§II.D)": 1e-6, "released code": 5e-5, "ratio": 50.0},
])
print(audit_d.to_string(index=False))
"""
    ),
    md(
        """
## 6. Audit E — per-batch metric averaging

The reference accumulates the metric inside the loader loop and divides by the
number of batches:

```python
rmse += RMSE(truth, pred)
...
rmse /= n
```

With `batch_size = 1` that is the mean of per-window RMSEs, not the RMSE of the
pooled predictions. By Jensen's inequality the former is never larger, and the
gap widens the more uneven the error is across weeks — which, on a series whose
maximum is 2631 and median 13, it very much is.

The cell below quantifies the gap using persistence as a stand-in predictor, so
no training is needed to see the effect.
"""
    ),
    code(
        """
idx = windows_all.index
persist_pred = np.stack([
    np.repeat(cases_all[i - 1][:, None], CFG.horizon, axis=1) for i in idx
])
persist_true = np.stack([cases_all[i:i + CFG.horizon].T for i in idx])

pooled_rmse = rmse(persist_pred, persist_true)
perbatch_rmse = batched_mean_metric(persist_pred, persist_true, rmse, batch_size=1)
pooled_mae = mae(persist_pred, persist_true)
perbatch_mae = batched_mean_metric(persist_pred, persist_true, mae, batch_size=1)

audit_e = pd.DataFrame([
    {"metric": "RMSE", "pooled": pooled_rmse, "per-window mean (reference)": perbatch_rmse,
     "reference is lower by %": 100 * (pooled_rmse - perbatch_rmse) / pooled_rmse},
    {"metric": "MAE", "pooled": pooled_mae, "per-window mean (reference)": perbatch_mae,
     "reference is lower by %": 100 * (pooled_mae - perbatch_mae) / pooled_mae},
])
print(audit_e.to_string(index=False))
print("\\nMAE is a mean of means and is unaffected. RMSE is not.")
"""
    ),
    md(
        """
## 7. Reproducing STGAT

STGAT is the paper's best cross-validated model and — usefully — the only one of
its five that needs no `torch-geometric-temporal`: the reference builds it from
plain `torch_geometric.nn.GATConv` plus two LSTM layers. `xcheck.models.STGAT`
is transcribed from `reference_repo/Models/gnn_models.py`.

The other four (ASTGCN, A3TGCN, DCRNN, AAGCN) come from
`torch-geometric-temporal`. If that package is installed they are run too; if
not, they are reported as unavailable rather than replaced by a lookalike.
"""
    ),
    code(
        """
try:
    import torch_geometric_temporal  # noqa: F401
    HAS_PGT = True
except ImportError:
    HAS_PGT = False

print(f"torch-geometric-temporal available: {HAS_PGT}")
if not HAS_PGT:
    print("  -> STGAT reproduces here; ASTGCN / A3TGCN / DCRNN / AAGCN are skipped.")
    print("  -> To include them:  pip install torch-geometric-temporal")
    print("     (see docs/decisions/0002 for why the main project avoids it)")
"""
    ),
    code(
        """
from xcheck.models import STGAT


def train_stgat(x_train, y_train, epochs, seed=0):
    \"\"\"Reference training loop: Adam, MSE, batch_size=1, no early stopping.\"\"\"
    torch.manual_seed(seed)
    model = STGAT(
        in_channels=CFG.window, n_pred=CFG.horizon, n_nodes=CFG.n_nodes,
        heads=CFG.heads, hidden=CFG.hidden, dropout=CFG.dropout,
    )
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for i in range(0, len(x_train), CFG.batch_size):
            opt.zero_grad()
            xb = x_train[i:i + CFG.batch_size]
            yb = y_train[i:i + CFG.batch_size]
            loss = loss_fn(model(xb, edge_index), yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model, x):
    model.eval()
    return model(x, edge_index).numpy()


def as_tensors(w, sl):
    \"\"\"Slice a Windows bundle into (x, y) tensors shaped for STGAT.\"\"\"
    x = torch.tensor(w.x[sl], dtype=torch.float32).squeeze(2)   # (B, N, window)
    y = torch.tensor(w.y[sl], dtype=torch.float32)              # (B, N, horizon)
    return x, y
"""
    ),
    code(
        """
import time

def run_stgat(report_on, normalize, label):
    \"\"\"One full segment-CV sweep. report_on='full' reproduces Table I.

    Ordering follows the reference: truncate to the segment's weeks, z-score,
    window, then split 70/30.
    \"\"\"
    results = []
    for frac in CFG.segments:
        n_weeks, n_seg = segment_weeks(frac)
        raw_seg = raw[:n_weeks]
        train_slice, _, test_slice = segment_split(n_seg, 0.7)
        eval_slice = slice(0, n_seg) if report_on == "full" else test_slice

        if normalize == "global":
            # The reference z-scores the WHOLE array and only then truncates to
            # the segment, so the statistics are the same for every segment.
            # Windowing is prefix-stable, so the segment's windows are exactly
            # the first n_seg windows of the full array.
            w = windows_all
        else:
            # Fold-local statistics: this segment's training weeks only.
            w = make_windows(
                raw_seg, use_all_features=False, normalize="train",
                train_end=CFG.window + train_slice.stop,
            )
        x_tr, y_tr = as_tensors(w, train_slice)
        x_ev, y_ev = as_tensors(w, eval_slice)

        t0 = time.time()
        model = train_stgat(x_tr, y_tr, CFG.epochs, seed=CFG.seed)
        pred = w.inverse(predict(model, x_ev))
        truth = w.inverse(y_ev.numpy())

        results.append(FoldResult(
            label=frac,
            n_train=len(x_tr),
            n_eval=len(x_ev),
            metrics={
                "MAE": batched_mean_metric(pred, truth, mae, CFG.batch_size),
                "RMSE": batched_mean_metric(pred, truth, rmse, CFG.batch_size),
                "MAE_pooled": mae(pred, truth),
                "RMSE_pooled": rmse(pred, truth),
            },
            extra={"seconds": round(time.time() - t0, 1)},
        ))
        last = results[-1]
        print(f"  [{label}] segment {frac}: train {last.n_train:4d} eval {last.n_eval:4d} "
              f"MAE {last.metrics['MAE']:7.2f} RMSE {last.metrics['RMSE']:7.2f} "
              f"({last.extra['seconds']}s)")
    return results


print("Published protocol (report on the full segment, global normalization):")
published = run_stgat(report_on="full", normalize="global", label="published")
"""
    ),
    code(
        """
print("Corrected protocol (report on held-out test only, fold-local normalization):")
corrected = run_stgat(report_on="test", normalize="train", label="corrected")
"""
    ),
    md("## 8. Results side by side"),
    code(
        """
PAPER_STGAT = {"MAE": (25.38, 1.37), "RMSE": (44.78, 2.26)}


def persistence_on(report_on):
    \"\"\"Persistence scored on exactly the slices a protocol reports on.

    A floor computed over different windows than the model is not a floor. This
    walks the same segments and the same eval slice as run_stgat, so the
    comparison is like for like.
    \"\"\"
    per_seg = []
    for frac in CFG.segments:
        _, n_seg = segment_weeks(frac)
        _, _, test_slice = segment_split(n_seg, 0.7)
        eval_slice = slice(0, n_seg) if report_on == "full" else test_slice
        idx = [CFG.window + k for k in range(eval_slice.start, eval_slice.stop)]
        pred = np.stack([np.repeat(cases_all[i - 1][:, None], CFG.horizon, axis=1) for i in idx])
        truth = np.stack([cases_all[i:i + CFG.horizon].T for i in idx])
        per_seg.append({
            "MAE": batched_mean_metric(pred, truth, mae, CFG.batch_size),
            "RMSE": batched_mean_metric(pred, truth, rmse, CFG.batch_size),
            "RMSE_pooled": rmse(pred, truth),
        })
    return {k: (float(np.mean([s[k] for s in per_seg])), float(np.std([s[k] for s in per_seg])))
            for k in per_seg[0]}


pub, cor = summarize(published), summarize(corrected)
pub_persist, cor_persist = persistence_on("full"), persistence_on("test")


def row(label, stats):
    out = {"protocol": label}
    for k in ("MAE", "RMSE"):
        out[k], out[k + " sd"] = stats[k]
    return out


comparison = pd.DataFrame([
    {"protocol": "Weng et al., Table I (published)",
     "MAE": PAPER_STGAT["MAE"][0], "MAE sd": PAPER_STGAT["MAE"][1],
     "RMSE": PAPER_STGAT["RMSE"][0], "RMSE sd": PAPER_STGAT["RMSE"][1]},
    row("STGAT, published protocol", pub),
    row("  persistence, same slices", pub_persist),
    row("STGAT, corrected protocol", cor),
    row("  persistence, same slices", cor_persist),
])
print(comparison.to_string(index=False))

print("\\nSame runs, RMSE pooled instead of averaged per window:")
pooled_view = pd.DataFrame([
    {"series": "STGAT, published protocol", "RMSE (pooled)": pub["RMSE_pooled"][0]},
    {"series": "STGAT, corrected protocol", "RMSE (pooled)": cor["RMSE_pooled"][0]},
    {"series": "persistence, corrected slices", "RMSE (pooled)": cor_persist["RMSE_pooled"][0]},
])
print(pooled_view.to_string(index=False))

if QUICK_TEST:
    print("\\n*** QUICK_TEST=True -- these numbers are degraded and must not be cited. ***")
"""
    ),
    code(
        """
results_dir = Path.cwd().parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

rows = []
for name, folds in (("published", published), ("corrected", corrected)):
    for f in folds:
        rows.append({
            "protocol": name, "segment": f.label, "n_train": f.n_train, "n_eval": f.n_eval,
            **f.metrics, **f.extra, "quick_test": QUICK_TEST,
        })
per_fold = pd.DataFrame(rows)
per_fold.to_csv(results_dir / "R1_stgat_per_segment.csv", index=False)
comparison.to_csv(results_dir / "R1_protocol_comparison.csv", index=False)

for name, frame in (("R1_audit_a_full_vs_test", audit_a), ("R1_audit_b_task_framing", audit_b),
                    ("R1_audit_c_normalization", audit_c), ("R1_audit_d_hyperparameters", audit_d),
                    ("R1_audit_e_metric_averaging", audit_e)):
    frame.to_csv(results_dir / f"{name}.csv", index=False)

print(f"wrote {len(list(results_dir.glob('R1_*.csv')))} CSVs to {results_dir}")
per_fold
"""
    ),
    md(
        """
## 9. What this establishes

Numbers below are from a full `QUICK_TEST = False` sweep (5 segments, 50 epochs),
logged as EXP-003 in `docs/EXPERIMENT_LOG.md`. Re-run before citing anything that
has changed.

**Finding 0 — Table I reproduces, and persistence beats it on the cross-validated
column.**

Our STGAT scores MAE 24.01 ± 1.32 / RMSE 42.31 ± 2.47 against the published
25.38 ± 1.37 / 44.78 ± 2.26 — within ~5%, with matching spread. The published
numbers are real and this reimplementation is faithful.

Persistence, on identical slices with the identical metric convention, scores
**MAE 18.58 / RMSE 34.67** — better than all ten models in Table I's *Cross
Validated* column. Be precise about the other column: on *Full Dataset*,
persistence (17.87 / 33.48) still beats every model on MAE, but A3TGCN (30.55)
and ASTGCN (32.41) beat it on RMSE. So the accurate claim is that persistence
wins outright on the cross-validated column, and on MAE everywhere — not that it
beats everything in the table.

The paper reports no naive baseline, and persistence gains nothing from the
training-inclusive evaluation of finding 1, because it does no training.

**Findings that do not depend on the training run** — they follow from the
released code and are demonstrated in sections 2–6:

1. **Table I's "Cross Validated" column is training-inclusive.** The held-out
   test score is computed and discarded; the reported number comes from a loader
   over the whole segment, ~70% of which is training data.
2. **The classical baselines and the GNNs solve different problems.** ARIMA, RF
   and LSTM run on Kalutara alone, predicting current cases from current
   covariates. The GNNs forecast 3 weeks ahead across 25 districts. The paper's
   headline claim — GNN RMSE 38.22 against baseline 70.65 — compares these
   directly.
3. **Normalization statistics span the whole series**, test period included, and
   are reused to inverse-transform predictions at scoring time.
4. **Stated and released hyperparameters differ** by 10× (lr) and 50× (weight
   decay).
5. **RMSE is averaged per window rather than pooled**, which reports a
   systematically smaller number on a heavy-tailed target.

**What this means for our project.** Our Phase-1 baseline "matching the
persistence floor" (RMSE 45.3 vs 44.8) has been read as underperforming against
Weng's STGAT at 44.78. It is not. 44.78 is a per-window-averaged RMSE (finding 5)
computed largely on training data (finding 1) under a normalization that saw the
test period (finding 3) — and persistence beats it anyway (finding 0).

**A spatio-temporal GNN matching rather than beating persistence on this dataset
is what the benchmark paper's own numbers show, once a naive baseline is placed
beside them.** Our Phase-1 result is not a shortfall against the literature; it
is the same result, reported honestly — which is the framing
`docs/decisions/0001` already argues for.

So: cite our own persistence floor in the ablation table, and cite Table I as
prior work with a stated protocol difference rather than as a target. Read
positively, this is an argument *for* the three contributions — the headroom over
persistence is genuinely open, and no published result on this dataset has closed
it.

None of this makes the paper's qualitative direction wrong — spatial structure
may well help, the graph framing is a real contribution, and the dataset remains
the right one to use. If any of this reaches the report, make the narrow claims
demonstrated above, not a broad verdict on the paper.
"""
    ),
]
