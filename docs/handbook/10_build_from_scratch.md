# Chapter 10 — Build It From Scratch

*Prerequisites: all previous chapters.*

A staged plan for rebuilding this project from an empty directory, using only
official documentation.

Each stage has a **checkpoint**: a concrete, verifiable claim. Do not move on
until the checkpoint passes. Most of the mistakes in Chapter 9 would have been
caught by a checkpoint one stage earlier.

**Build the evaluation before the model.** Stages 1–4 contain no neural network at
all. This is deliberate and it is the single most important structural advice in
the handbook.

---

## Stage 0 — Environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib pytest ruff
```

Add torch only at Stage 5. Everything before then is NumPy.

**Pin your versions** in `requirements.txt` with `==`. Chapter 9 showed what
unpinned dependencies cost when you try to reproduce something a year later.

### Project layout

```
dengue/
├── src/dengue/
│   ├── __init__.py
│   ├── data.py          # loading, windowing, folds
│   ├── metrics.py       # scoring
│   ├── baselines.py     # persistence
│   └── models.py        # torch — Stage 5 onward
├── tests/
├── notebooks/
└── requirements.txt
```

**Checkpoint 0:** `python -c "import numpy; print(numpy.__version__)"` runs inside
the venv.

---

## Stage 1 — Load and verify the data

**Goal:** load the array and know exactly what is in it.

```python
# src/dengue/data.py
import json
from pathlib import Path
import numpy as np

CASES_IDX = 5

def load_raw(npy_path):
    """Return (weeks, districts, features) as float64."""
    return np.nan_to_num(np.load(Path(npy_path), allow_pickle=True)).astype(np.float64)

def load_districts(adj_path):
    """District names in the canonical order: sorted JSON keys."""
    return sorted(json.loads(Path(adj_path).read_text(encoding="utf-8")))
```

Then verify, do not assume:

```python
N = load_raw(path)
assert N.shape == (459, 25, 11)

cases = N[..., CASES_IDX]
print("median", np.median(cases))          # 13
print("max",    cases.max())               # 2631
print("zeros",  (cases == 0).mean())       # 0.097
```

**Checkpoint 1:** you can state each channel's identity and lag, and the target's
median, max and zero fraction. If the numbers do not match Chapter 3, you have
the wrong index or the wrong file.

**Do not skip this.** Chapter 3 D1 (4% zero-filled weather) and D2 (the week-395
artifact) are both visible at this stage, and both are invisible later.

---

## Stage 2 — Metrics

**Goal:** a scoring module you trust.

```python
# src/dengue/metrics.py
import numpy as np

def _pair(pred, truth):
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs truth {y.shape}")
    if p.size == 0:
        raise ValueError("cannot score empty arrays")
    return p, y

def rmse(pred, truth):
    p, y = _pair(pred, truth)
    return float(np.sqrt(np.mean((p - y) ** 2)))

def mae(pred, truth):
    p, y = _pair(pred, truth)
    return float(np.mean(np.abs(p - y)))

def smape(pred, truth, eps=1e-6):
    p, y = _pair(pred, truth)
    return float(np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + eps)) * 100)

def mape_masked(pred, truth, floor=1.0):
    p, y = _pair(pred, truth)
    m = y >= floor
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(p[m] - y[m]) / y[m]) * 100)
```

Write the `_pair` guard **first**. It is three lines and it eliminates the entire
class of silent-broadcasting bugs.

Tests with hand-computable answers:

```python
def test_rmse_perfect():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0

def test_rmse_known():
    # errors 1, 1, 1  ->  sqrt(mean([1,1,1])) == 1
    assert rmse([2, 3, 4], [1, 2, 3]) == 1.0

def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        rmse(np.zeros((3, 2)), np.zeros((2, 3)))

def test_smape_bounded():
    assert smape([1000], [1]) <= 200.0
```

**Checkpoint 2:** `pytest` passes, and you can compute each metric by hand on a
3-element example and get the same answer.

---

## Stage 3 — Windowing and folds

**Goal:** turn the series into training examples without leaking.

```python
def build_windows(series, window, horizon):
    """series: (weeks, districts). Returns (x, y, p, index)."""
    n_weeks = series.shape[0]
    ids = list(range(window, n_weeks - horizon))
    x = np.stack([series[i - window : i].T for i in ids])
    y = np.stack([series[i : i + horizon].T for i in ids])
    p = np.stack([np.repeat(series[i - 1][:, None], horizon, axis=1) for i in ids])
    return x, y, p, np.asarray(ids)
```

Verify the shapes and the boundaries yourself:

```python
x, y, p, idx = build_windows(cases, 3, 3)
assert x.shape == (453, 25, 3)
assert y.shape == (453, 25, 3)
assert idx[0] == 3 and idx[-1] == 455
```

Then the folds. **The normalisation statistics go inside the loop.**

```python
def build_folds(cases, window=3, horizon=3,
                origins=(0.55, 0.70, 0.85), test_frac=0.15, val_weeks=30):
    n_weeks = cases.shape[0]
    ids = list(range(window, n_weeks - horizon))
    folds = []
    for origin in origins:
        cut = int(origin * len(ids))
        end = int(min(origin + test_frac, 1.0) * len(ids))
        if end <= cut:
            continue
        train_ids = ids[: cut - val_weeks]
        val_ids   = ids[cut - val_weeks : cut]
        test_ids  = ids[cut:end]

        # TRAINING WEEKS ONLY
        train_weeks = np.log1p(cases[: train_ids[-1] + 1])
        mean, std = float(train_weeks.mean()), float(train_weeks.std() + 1e-8)
        z = (np.log1p(cases) - mean) / std
        ...
    return folds
```

**Checkpoint 3, three assertions:**

```python
# 1. Chronological: no test index precedes any training index
assert max(train_ids) < min(test_ids)

# 2. Fold-specific statistics: they must differ across folds
assert folds[0].mean != folds[1].mean

# 3. Inversion round-trips
assert np.allclose(fold.inverse((np.log1p(cases) - mean) / std), cases)
```

The third one catches an entire family of transform bugs in one line. Write it.

---

## Stage 4 — Persistence

**Goal:** the number everything else must beat.

```python
def persistence(cases, test_index, horizon=3):
    pred  = np.stack([np.repeat(cases[i - 1][:, None], horizon, axis=1) for i in test_index])
    truth = np.stack([cases[i : i + horizon].T for i in test_index])
    return pred, truth
```

Score it on every fold. Record the numbers.

**Checkpoint 4:** you have a persistence RMSE and MAE per fold, on raw counts,
and you have written them down.

**This is the most important checkpoint in the chapter.** From here on, every
model is compared against these numbers, on these exact windows.

While you are here, check for the artifact:

```python
sq_err = ((pred - truth) ** 2).sum(axis=(1, 2))
share  = np.sort(sq_err)[::-1][:6].sum() / sq_err.sum()
print(f"top 6 windows carry {share:.0%} of squared error")   # ~90% on origin 0.85
```

If a handful of windows dominate, you have found your equivalent of week 395 and
you must report scores both ways from now on.

---

## Stage 5 — The simplest possible model

Now add torch.

```bash
pip install torch==2.1.2
```

Start with **no graph at all**:

```python
import torch
from torch import nn

class Simple(nn.Module):
    """No graph. 25 independent series. The control."""
    def __init__(self, window=3, horizon=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, horizon),
        )
    def forward(self, x):          # (B, N, W) -> (B, N, H)
        return self.net(x)
```

The training loop, with everything from Chapter 4:

```python
def train(model, fold, epochs=150, lr=1e-3, weight_decay=5e-4,
          patience=30, batch_size=32, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    n = fold.x_train.shape[0]
    best_state, best_val, waited = None, float("inf"), 0

    for _ in range(epochs):
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            opt.zero_grad()
            pred   = model(fold.x_train[idx])
            target = fold.y_train[idx] - fold.p_train[idx]     # RESIDUAL
            loss   = loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        val = evaluate(model, fold, "val")["RMSE"]
        if val < best_val - 1e-6:
            best_val, waited = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
```

And evaluation, which must add persistence back and invert the transform:

```python
@torch.no_grad()
def evaluate(model, fold, split="test"):
    model.eval()                                   # turns dropout OFF
    x = getattr(fold, f"x_{split}")
    y = getattr(fold, f"y_{split}")
    p = getattr(fold, f"p_{split}")
    pred = model(x) + p                            # residual -> absolute
    return {"RMSE": rmse(fold.inverse(pred.numpy()), fold.inverse(y.numpy()))}
```

**Checkpoint 5:** this model is **within ~2 RMSE of persistence**.

If it is far worse, something is wrong — most likely you forgot to add `p` back,
or you are scoring in log space. If it is far better, you have leaked; go back to
Checkpoint 3.

### The three-step ablation

Before adding a graph, verify the two decisions that actually matter:

| variant | expected |
|---|---|
| absolute target, raw counts | RMSE ≈ 66 |
| absolute target, log1p | **worse** than raw |
| **residual + log1p** | RMSE ≈ 45 |

Reproducing this ordering — including log1p alone being *harmful* — confirms your
pipeline behaves like the real one, and teaches you that these two techniques
interact.

---

## Stage 6 — Add the graph

```python
def load_adjacency(adj_path):
    adj = json.loads(Path(adj_path).read_text(encoding="utf-8"))
    names = sorted(adj)
    index = {n: i for i, n in enumerate(names)}
    a = np.eye(len(names))
    for district, neighbours in adj.items():
        for neighbour in neighbours:
            a[index[district], index[neighbour]] = 1.0
    a = np.maximum(a, a.T)                       # symmetrise (D4)
    deg = a.sum(1)
    d = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    return d @ a @ d, names
```

Verify before using:

```python
assert np.allclose(A, A.T), "adjacency must be symmetric"
assert (np.diag(A) > 0).all(), "self-loops missing"
assert A.shape == (25, 25)
```

Then the graph model:

```python
class STGNN(nn.Module):
    def __init__(self, n_nodes, window=3, horizon=3, hidden=64, dropout=0.1):
        super().__init__()
        self.lin_in = nn.Linear(window, hidden)
        self.gc1 = nn.Linear(hidden, hidden)
        self.gc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x, a):
        h = torch.relu(self.lin_in(x))
        h = torch.relu(a @ self.gc1(h))
        h = self.drop(h)
        h = torch.relu(a @ self.gc2(h))
        return self.head(h)
```

**Checkpoint 6:** run this against Stage 5's no-graph model, **same seeds, same
folds**, and report the paired difference.

Expect the difference to be **small and not significant**. That is the correct
result on this dataset. If you get a large improvement, look for leakage before
celebrating.

**Keep the no-graph arm forever.** It is the control that makes every later graph
claim meaningful.

---

## Stage 7 — The adaptive graph

```python
self.emb_src = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
self.emb_dst = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)

def adjacency(self, fixed):
    if self.graph_mode == "none":
        return torch.eye(self.n_nodes, device=fixed.device)
    if self.graph_mode == "fixed":
        return fixed
    learned = torch.softmax(torch.relu(self.emb_src @ self.emb_dst.T), dim=-1)
    if self.graph_mode == "adaptive":
        return learned
    return self.alpha * fixed + (1.0 - self.alpha) * learned
```

**Allocate the embeddings in every arm**, even where unused, so parameter
initialisation consumes the same RNG draws (Chapter 4 §4.8). Otherwise your arms
differ in the graph *and* in every initial weight.

**Checkpoint 7:** all four arms run, and their differences are reported as
**paired** comparisons on matched (fold, seed) pairs with a p-value.

Expect a null result. Chapter 6 §6.5 has the numbers.

---

## Stage 8 — Protocol discipline

Formalise what you have been doing:

```
3 origins × 3 seeds = 9 runs per configuration
W = 3, H = 3
metrics per horizon and overall: RMSE, MAE, SMAPE, masked MAPE
pooled, on genuinely held-out windows, with and without artifact windows
```

Write one row per **(fold, seed, horizon)** to CSV. Never pre-aggregate — you
cannot recover a distribution from a mean, and you will want the paired test
later.

Record with each run: config, commit SHA, seeds, fold boundaries, timestamp.

**Checkpoint 8:** you can regenerate any reported number from its CSV and its
commit SHA, without re-deriving anything by hand.

---

## Stage 9 — Extensions, in order of expected value

Ranked by what Chapters 3–9 measured, not by what sounds impressive:

**1. Fix the covariates.** They are 4% zero-filled and currently unusable
(Chapter 3 D1). Mask or interpolate, normalise over observed values only. This is
the only change that adds genuinely new information to the model.

**2. Probabilistic forecasts.** Predict `(mu, log_var)`, train on Gaussian NLL,
report CRPS/PICP/MPIW. A calibrated interval is operationally more useful than a
point estimate at the same RMSE — and nothing in the benchmark provides one.

**3. Focus on the extrapolation fold.** Every model handles interpolation folds
comfortably and fails on the fold whose test window exceeds its training maximum.
That fold is the whole problem, and it is where a real improvement would show.

**4. Per-horizon heads.** Cheap, and justified: autocorrelation falls 0.92 → 0.82
across the three horizons, so a shared head averages over three different
problems.

**5. Huber loss.** Skewness 8.6; squared error is dominated by a handful of
windows.

### What not to do

| idea | why not |
|---|---|
| week-of-year feature | No annual cycle exists (D6, retracted finding, EXP-012) |
| wider windows | EXP-012, 8 folds × 3 seeds, no improvement |
| deeper/wider models | Every architecture lands within ±2 RMSE of the floor |
| a learned graph as *the* contribution | Null result, sign flips between implementations |

**The ceiling is in the data, not the architecture.** Spending Stage 9 on
architecture is spending it in the one place Chapters 3–9 measured no headroom.

---

## The checkpoint summary

| Stage | Checkpoint |
|---|---|
| 0 | venv works, versions pinned |
| 1 | every channel identified; target statistics match |
| 2 | metrics tested against hand-computed values |
| 3 | chronological split, per-fold statistics, inversion round-trips |
| 4 | **persistence scored and written down** |
| 5 | no-graph model within ~2 RMSE of persistence |
| 6 | graph vs no-graph, paired, small and non-significant |
| 7 | four arms, paired comparison with p-values |
| 8 | any number regenerable from CSV + SHA |
| 9 | extensions ranked by measured headroom |

---

## Three closing rules

**1. Build the evaluation first.** Stages 1–4 have no neural network. Nearly every
error in Chapter 9 — ours and the published work's — was in measurement.

**2. Never delete a baseline.** Persistence and the no-graph arm stay in every
table forever. They are what make the other rows mean something.

**3. When a result surprises you, suspect the measurement first.** Every large
apparent improvement in this project's history turned out to be leakage, a metric
convention, or a bug. Every one.

---

*Next: [Chapter 11 — Glossary and References](11_glossary.md)*
