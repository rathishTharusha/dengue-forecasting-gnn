"""Cell sources for the graph-mode sweep: does the learned adjacency earn its place?

EXP-022 found the first positive signal this project has produced, at n=9 and 60
epochs: with the self-path restored, ``hybrid`` scored 28.55 artifact-free against
a 29.52 persistence floor, ahead of ``fixed`` (28.97) and ``none`` (28.90), while
``adaptive`` alone stayed worst (29.16). This kernel settles it at full scale.

Three things it adds over EXP-022
---------------------------------
1. **150 epochs and 10 seeds**, giving n=30 per cell instead of 9, with paired
   tests against ``fixed`` -- the authors' hand-built graph is the control that
   matters here, not an abstract baseline.
2. **The blend weight as an arm.** ``hybrid`` is ``alpha * fixed + (1-alpha) *
   learned``. If alpha=0.75 ties with alpha=0.5, the learned half is contributing
   little and the effect is really about softening the prior. Graph WaveNet's own
   choice is an even blend, so 0.5 is the pre-registered one and the others are
   there to say whether it matters.
3. **The self-path fix as an arm.** Running ``self_path=False`` reproduces the
   EXP-018/EXP-022 numbers in the same job, so the claim "the earlier result was
   an implementation defect" is measured rather than asserted.

What this kernel does not cover
-------------------------------
One encoder -- ``analysis/lib/adaptive.py``'s ``STGNN`` -- not the five reproduced
architectures. That is deliberate: it is the only place where the adjacency is a
free variable with everything else held identical, which is what makes a graph
comparison a graph comparison. STGAT's attention, ASTGCN's Chebyshev filters and
AAGCN's constructor-baked adjacency each interact with the graph differently, so
swapping the adjacency there changes more than one thing at a time.
"""

import env_setup
from gen_kernels import code, md

#: Branch the kernel clones. The default branch carries the dataset but not
#: ``analysis/lib``, so this cannot be left implicit.
BRANCH = "feat/reproduction-and-eda"


def build() -> list:
    return [
        md(
            """
# Graph-mode sweep — does a learned adjacency earn its place?

## The question

`docs/ROADMAP.md` lists a learned adaptive graph as contribution (c). Three
earlier tests said it does not help:

| test | result |
|---|---|
| E2 controlled encoder (n=9) | `none` 42.82 < `fixed` 43.36 < `hybrid` 43.55 < `adaptive` 43.81 |
| AAGCN adaptive flag, artifact-free (n=9, paired) | adaptive **+2.76 RMSE worse**, 7/9 folds worse |
| EXP-018 vs the head-to-head benchmark | effect flips sign between implementations |

The first of those contained a clue rather than a conclusion: **identity beat a
real district adjacency.** That is not a fact about Sri Lankan geography.

## The defect it turned out to be

Propagation was `relu(A @ W h)` — no separate route for a node's own state.
`none` and `fixed` both carry a diagonal (`fixed` is built with self-loops), so
they keep it. `adaptive` is `softmax(ReLU(E1 E2ᵀ))`, which at small initialisation
is close to **uniform over 25 districts** — an averaging matrix that erases the
node's own signal exactly where lag-1 autocorrelation (r = 0.92) carries almost
all of the information.

Graph WaveNet does not have this problem, because its diffusion convolution
includes a `k = 0` identity term. `STGNN` now does the same through an explicit
`self_path`, present in **every** mode so the comparison stays controlled.

At n=9 and 60 epochs (EXP-022), artifact-free:

| mode | `self_path=False` | `self_path=True` |
|---|---|---|
| `none` | 29.16 | 28.90 |
| `fixed` | 29.03 | 28.97 |
| `adaptive` | 29.29 | 29.16 |
| `hybrid` | **28.79** | **28.55** |

Persistence floor 29.52. `hybrid` is the first arm in this project to sit clearly
ahead of the floor — by 3.3%. At n=9 over a 0.42 spread that is suggestive and
nothing more, which is what this kernel is for.

## What runs here

- 10 arms: `none`, `fixed`, `adaptive`, `hybrid` at α ∈ {0.25, 0.5, 0.75} with the
  self-path on, plus the four original modes with it off to reproduce EXP-018.
- 3 origins × **10 seeds** × 150 epochs, early stopping on a genuine validation
  split. n = 30 per cell.
- Paired against `fixed`, because the question is whether *learning* the graph
  beats the graph the authors drew by hand.
- Every fold scored twice, with and without the six week-395 artifact windows,
  models and persistence floor alike.

## What would count as a result

The seed-to-seed spread sets the bar. `hybrid` needs to beat `fixed` on matched
(origin, seed) pairs by more than that spread, on the **artifact-free** column,
to be worth writing up. A win that appears only on the all-windows column is a
win at predicting a reporting backlog.
"""
        ),
        md("## 1. Environment — the authors' pins under a standalone Python 3.11"),
        code(env_setup.PREAMBLE),
        code(env_setup.SETUP),
        code(env_setup.VERIFY_ENV),
        md("## 2. This repository, for the protocol and the encoder"),
        code(
            f'''
PROJECT = "https://github.com/rathishTharusha/dengue-forecasting-gnn.git"
# Pinned explicitly: the default branch carries the dataset but not
# `analysis/lib`, so an unpinned clone finds no encoder and fails below.
BRANCH = "{BRANCH}"
PROJ = SCRATCH / "project"
if not PROJ.exists():
    sh("git", "clone", "--depth", "1", "--branch", BRANCH, PROJECT, str(PROJ))
print("branch:", BRANCH)

DATA = PROJ / "notebooks" / "baseline"
LIB = PROJ / "analysis" / "lib"
print("data:", sorted(p.name for p in DATA.iterdir() if p.suffix in {{".npy", ".json"}}))
print("lib :", sorted(p.name for p in LIB.iterdir() if p.suffix == ".py"))

# The self-path is what this sweep is about, so fail loudly rather than silently
# running the pre-fix encoder if the branch is behind.
src = (LIB / "adaptive.py").read_text(encoding="utf-8")
assert "self_path" in src, (
    "adaptive.py on this branch has no self_path -- push the commit that adds it "
    "before running this kernel, or the sweep measures the old defect"
)
print("self_path present: OK")
'''
        ),
        md(
            """
## 3. The sweep

Runs in the Python 3.11 environment as a subprocess, where the pinned stack
lives. Results are written after every run, so a kernel that hits the wall clock
still leaves behind everything it finished.
"""
        ),
        code(
            '''
SWEEP = r"""
__SWEEP__
"""

script = SCRATCH / "graph_sweep.py"
script.write_text(SWEEP, encoding="utf-8")
RESULTS_JSON = WORK / "graph_sweep.json"
print("wrote", script)
'''.replace("__SWEEP__", '''
import itertools, json, sys, time
from pathlib import Path

import numpy as np
import torch

LIB = Path(sys.argv[1])
DATA = Path(sys.argv[2])
OUT = Path(sys.argv[3])
sys.path.insert(0, str(LIB))

import adaptive as base
import improved as imp

# (label, graph_mode, alpha, self_path). The four self_path=False arms exist to
# reproduce EXP-018/EXP-022 inside this job, so "the earlier result was a defect"
# is a measurement rather than a claim.
ARMS = [
    ("none",          "none",     0.5,  True),
    ("fixed",         "fixed",    0.5,  True),
    ("adaptive",      "adaptive", 0.5,  True),
    ("hybrid_0.25",   "hybrid",   0.25, True),
    ("hybrid_0.50",   "hybrid",   0.50, True),
    ("hybrid_0.75",   "hybrid",   0.75, True),
    ("none_noself",   "none",     0.5,  False),
    ("fixed_noself",  "fixed",    0.5,  False),
    ("adapt_noself",  "adaptive", 0.5,  False),
    ("hybrid_noself", "hybrid",   0.50, False),
]
SEEDS = tuple(range(10))
EPOCHS, WINDOW, HORIZON = 150, 3, 3

cases, adjacency, _ = base.load_dataset(
    DATA / "sri_lanka_2013-2022_shifted.npy", DATA / "sri_lanka_adj_list.json")
fixed = torch.tensor(adjacency, dtype=torch.float32)
folds = base.build_folds(cases, WINDOW, HORIZON)
ARTIFACT = {f.origin: imp.artifact_windows(f.test_index, WINDOW, HORIZON) for f in folds}

records = [dict(arm="persistence", origin=f.origin, seed=-1,
                **base.persistence_scores(f, cases, HORIZON, ARTIFACT[f.origin]))
           for f in folds]

started = time.time()
for (label, mode, alpha, self_path), fold, seed in itertools.product(ARMS, folds, SEEDS):
    t0 = time.time()
    try:
        model, _ = base.train_one(fold, fixed, mode, seed=seed, epochs=EPOCHS,
                                  alpha=alpha, self_path=self_path)
        model.eval()
        with torch.no_grad():
            pred = fold.inverse((model(fold.x_test, fixed) + fold.p_test).numpy())
        sc = base.pooled_scores(pred, fold.inverse(fold.y_test.numpy()),
                                ARTIFACT[fold.origin])
    except Exception as exc:
        print(f"{label:14s} o{fold.origin} s{seed} FAILED "
              f"{type(exc).__name__}: {str(exc)[:70]}", flush=True)
        continue
    records.append(dict(arm=label, graph_mode=mode, alpha=alpha, self_path=self_path,
                        origin=fold.origin, seed=seed, **sc))
    print(f"{label:14s} o{fold.origin} s{seed} RMSE {sc['RMSE']:7.2f} "
          f"clean {sc['RMSE_clean']:6.2f} ({time.time()-t0:.0f}s)", flush=True)
    OUT.write_text(json.dumps(records, indent=1), encoding="utf-8")

print(f"\\n{time.time()-started:.0f}s -> {OUT}  ({len(records)} records)")
'''.strip())
        ),
        code(
            '''
started = time.time()
proc = subprocess.Popen(
    [str(PY311), "-u", str(script), str(LIB), str(DATA), str(RESULTS_JSON)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
for line in proc.stdout:
    print(line, end="")
proc.wait()
print(f"\\nsweep exited {proc.returncode} after {(time.time()-started)/60:.1f} min")
'''
        ),
        md("## 4. Results"),
        code(
            '''
import numpy as np

records = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
model = [r for r in records if r["arm"] != "persistence"]
order = ["none", "fixed", "adaptive", "hybrid_0.25", "hybrid_0.50", "hybrid_0.75",
         "none_noself", "fixed_noself", "adapt_noself", "hybrid_noself"]
arms = [a for a in order if any(r["arm"] == a for r in model)]

for field, caption in [("RMSE", "all windows"), ("RMSE_clean", "artifact-free")]:
    floor = np.mean([r[field] for r in records if r["arm"] == "persistence"])
    print(f"pooled RMSE, {caption}   persistence floor {floor:.2f}")
    print(f"{'arm':16s}{'mean':>9s}{'sd':>8s}{'n':>5s}{'vs floor':>10s}")
    for a in arms:
        v = np.array([r[field] for r in model if r["arm"] == a])
        print(f"{a:16s}{v.mean():9.2f}{v.std(ddof=1):8.2f}{len(v):5d}"
              f"{100 * (v.mean() / floor - 1):+9.1f}%")
    print()
'''
        ),
        code(
            '''
from scipy import stats

# Paired against `fixed`: the question is whether LEARNING the graph beats the
# graph the authors drew by hand, on identical folds and seeds.
key = lambda r: (r["origin"], r["seed"])

for field, caption in [("RMSE", "all windows"), ("RMSE_clean", "artifact-free")]:
    basis = {key(r): r[field] for r in model if r["arm"] == "fixed"}
    print(f"paired against `fixed` -- same origin and seed -- on {caption}\\n")
    print(f"{'arm':16s}{'mean dRMSE':>12s}{'sd':>8s}{'better':>10s}{'p':>9s}")
    for a in arms:
        if a == "fixed":
            continue
        pairs = [(r[field], basis[key(r)]) for r in model
                 if r["arm"] == a and key(r) in basis]
        if len(pairs) < 3:
            continue
        d = np.array([x - y for x, y in pairs])
        _, p = stats.ttest_rel([x for x, _ in pairs], [y for _, y in pairs])
        print(f"{a:16s}{d.mean():+12.3f}{d.std(ddof=1):8.3f}"
              f"{int((d < 0).sum()):7d}/{len(d):<3d}{p:9.4f}")
    print()

# Did the self-path fix actually matter? Paired, same mode, same fold and seed.
print("effect of the self-path, paired within each mode (negative = fix helps)\\n")
for on, off in [("none", "none_noself"), ("fixed", "fixed_noself"),
                ("adaptive", "adapt_noself"), ("hybrid_0.50", "hybrid_noself")]:
    a = {key(r): r["RMSE_clean"] for r in model if r["arm"] == on}
    b = {key(r): r["RMSE_clean"] for r in model if r["arm"] == off}
    shared = sorted(set(a) & set(b))
    if not shared:
        continue
    d = np.array([a[k] - b[k] for k in shared])
    _, p = stats.ttest_rel([a[k] for k in shared], [b[k] for k in shared])
    print(f"  {on:14s}{d.mean():+8.3f}  {int((d < 0).sum())}/{len(d)} better  p={p:.4f}")
'''
        ),
        md(
            """
## 5. How to read this

**The artifact-free column is the one about forecasting.** Roughly 90% of the
origin-0.85 fold's squared error comes from six windows of backlogged reporting
in week 395. An arm that wins only on the all-windows column has learned to
predict a data-entry event.

**Paired against `fixed`, not against the floor.** Everything here already sits
within a couple of RMSE of persistence; the interesting comparison is the graph,
and pairing removes fold difficulty, which is larger than any effect being
measured.

**Three outcomes worth naming in advance:**

- `hybrid_0.50` beats `fixed` on artifact-free pairs with p below the seed noise
  → contribution (c) is real, and it works the way Graph WaveNet says it does:
  the learned graph augments the prior, it does not replace it.
- `hybrid_0.75` matches `hybrid_0.50` → the gain is from softening the hand-built
  adjacency, not from anything learned. Worth saying plainly.
- Nothing separates from `fixed` → EXP-022's signal was n=9 noise, contribution
  (c) is a negative result, and the write-up says so.

**The self-path table settles a separate question:** whether EXP-018's verdict
was about adaptive graphs or about a missing identity term. If the fix moves
every mode, the earlier comparison was measuring an implementation defect.
"""
        ),
    ]
