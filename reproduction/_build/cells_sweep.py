"""Cell sources for the improved-architecture sweep, run on Kaggle.

Two things at once: establish a baseline from the five *verified* architectures
under this project's frozen protocol, and test the increments that the papers and
our own EDA justify.

One kernel per architecture. DCRNN measures ~9 s/epoch on the largest fold, so
all five in one kernel would run 11-14 h against Kaggle's 12 h CPU limit. Five
kernels run in parallel instead, each writing the same record schema;
``analysis/_build/merge_sweep.py`` joins them into the cross-architecture table.
The alternative -- a GPU kernel -- would mean a CUDA build of ``torch==2.1.2`` and
a different PyG wheel index, i.e. a fourth deviation from the authors'
environment, for a sweep whose whole claim is that it runs in theirs.
"""

import env_setup
from gen_kernels import code, md

#: Branch the kernels clone. The default branch carries the dataset but not
#: ``analysis/lib``, so this cannot be left implicit.
BRANCH = "feat/reproduction-and-eda"

#: The architectures this module generates a kernel for, in Table I order.
ARCHITECTURES = ("STGAT", "A3TGCN", "ASTGCN", "DCRNN", "AAGCN")

#: Increment -> (citation, what the evidence says).
INCREMENTS = [
    ("per_horizon_heads",
     "GulMohamed et al. Eq. (11)",
     "one output layer per step; EDA F6 shows lag-1 r=0.92 falls to 0.82 at lag 3, "
     "so the three horizons are not the same problem"),
    ("temporal_attention",
     "GulMohamed et al. Eq. (6)-(8)",
     "additive attention over the window; their Table 6 attributes +0.5 RMSE to "
     "removing it, the smallest of their three ablations"),
    ("huber",
     "EDA distribution",
     "skewness 8.6 and the top 1% of district-weeks carrying 18.3% of cases make "
     "squared error a few-window average"),
    ("probabilistic",
     "GulMohamed et al. Eq. (12)-(14)",
     "Gaussian head; docs/ROADMAP.md requires CRPS/PICP/MPIW from Phase 3 and a "
     "point forecast cannot supply them"),
]

#: Generic sweep driver. The architecture arrives as argv[4] so this text is
#: identical in all five kernels -- a per-kernel copy would be five things to keep
#: in step, and they would not stay in step.
SWEEP = r"""
import itertools, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn

LIB = Path(sys.argv[1])
DATA = Path(sys.argv[2])
OUT = Path(sys.argv[3])
ARCH = sys.argv[4]
sys.path.insert(0, str(LIB))

import adaptive as base
import improved as imp
import reproduced as arch

INCREMENTS = [
    {},
    {"per_horizon_heads": True},
    {"temporal_attention": True},
    {"huber": True},
    {"probabilistic": True},
]
SEEDS = (0, 1, 2)
EPOCHS, WINDOW, HORIZON = 150, 3, 3

cases, adjacency, _ = base.load_dataset(
    DATA / "sri_lanka_2013-2022_shifted.npy", DATA / "sri_lanka_adj_list.json")
src, dst = np.nonzero(adjacency)
edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
folds = base.build_folds(cases)

# Which test windows touch week 395. Under this protocol only the origin-0.85
# fold has any -- 6 of its 68 -- and none of them are in any training split.
ARTIFACT = {f.origin: imp.artifact_windows(f.test_index, WINDOW, HORIZON) for f in folds}
for f in folds:
    print(f"origin {f.origin}: {int(ARTIFACT[f.origin].sum())}/{len(f.test_index)} "
          f"test windows touch week {imp.ARTIFACT_WEEK}", flush=True)

records = [dict(arch="persistence", increment="base", origin=f.origin, seed=-1,
                **base.persistence_scores(f, cases, HORIZON, ARTIFACT[f.origin]))
           for f in folds]


# Backbone plus the optional attention front-end, as ONE checkpointable unit.
# Attention carries parameters, so a checkpoint holding only the backbone would
# restore the best backbone beside the *last* epoch's attention weights.
class Net(nn.Module):

    def __init__(self, inc):
        super().__init__()
        kwargs = {"adaptive": False, "channels": 8} if ARCH == "AAGCN" else {}
        self.core = arch.build(ARCH, 25, WINDOW, HORIZON,
                               edge_index=edge_index, inc=inc, **kwargs)
        self.attn = imp.TemporalAttention(WINDOW) if inc.temporal_attention else None

    def forward(self, x):
        return self.core(self.attn(x) if self.attn is not None else x, edge_index)


def run(inc, fold, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(inc)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    point_loss = nn.HuberLoss(delta=1.0) if inc.huber else nn.MSELoss()

    def loss_fn(out, target):
        if not inc.probabilistic:
            return point_loss(out, target)
        return imp.gaussian_nll(out[..., 0], out[..., 1], target)

    def predict(split):
        # Residual over persistence; the Gaussian head is scored on its mean so
        # the number stays comparable to the point arms.
        out = net(getattr(fold, "x_" + split))
        mu = out[..., 0] if inc.probabilistic else out
        return mu + getattr(fold, "p_" + split), out

    n = len(fold.x_train)
    best, best_val, waited = None, float("inf"), 0
    for _ in range(EPOCHS):
        net.train()
        order = torch.randperm(n)
        for s in range(0, n, 32):
            i = order[s:s + 32]
            opt.zero_grad()
            loss = loss_fn(net(fold.x_train[i]), fold.y_train[i] - fold.p_train[i])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = base.rmse(fold.inverse(predict("val")[0].numpy()),
                          fold.inverse(fold.y_val.numpy()))
        if v < best_val - 1e-6:
            best_val, waited = v, 0
            best = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            waited += 1
            if waited >= 30:
                break
    if best:
        net.load_state_dict(best)

    net.eval()
    with torch.no_grad():
        pred_z, out = predict("test")
    pred = fold.inverse(pred_z.numpy())
    truth = fold.inverse(fold.y_test.numpy())
    scores = base.pooled_scores(pred, truth, ARTIFACT[fold.origin])

    if inc.probabilistic:
        # Eq. (14): mu +- 1.96 sigma. fold.inverse() is monotone, so transforming
        # the bounds is the same as transforming the interval.
        sigma = torch.exp(0.5 * out[..., 1].clamp(-10.0, 10.0)).numpy()
        lo = fold.inverse(pred_z.numpy() - 1.96 * sigma)
        hi = fold.inverse(pred_z.numpy() + 1.96 * sigma)
        scores["PICP"] = float(np.mean((truth >= lo) & (truth <= hi)))
        scores["MPIW"] = float(np.mean(hi - lo))
    return scores


started = time.time()
for inc_kwargs, fold, seed in itertools.product(INCREMENTS, folds, SEEDS):
    inc = imp.Increments(**inc_kwargs)
    t0 = time.time()
    try:
        sc = run(inc, fold, seed)
    except Exception as exc:
        print(f"{ARCH:8s} {inc.label():20s} o{fold.origin} s{seed} FAILED "
              f"{type(exc).__name__}: {str(exc)[:70]}", flush=True)
        continue
    records.append(dict(arch=ARCH, increment=inc.label(),
                        origin=fold.origin, seed=seed, **sc))
    print(f"{ARCH:8s} {inc.label():20s} o{fold.origin} s{seed} "
          f"RMSE {sc['RMSE']:7.2f}  clean {sc['RMSE_clean']:6.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    OUT.write_text(json.dumps(records, indent=1), encoding="utf-8")

print(f"\n{time.time()-started:.0f}s -> {OUT}  ({len(records)} records)")
"""


def build(arch: str) -> list:
    """Cells for one architecture's sweep kernel."""
    return [
        md(
            f"""
# Improved-architecture sweep — {arch}

## What this runs, and why on Kaggle

Two jobs in one sweep, of which this kernel is the **{arch}** slice:

1. **A credible baseline.** The five architectures from Weng et al. — STGAT,
   A3TGCN, ASTGCN, DCRNN, AAGCN — are the only ones in this project that have
   been *verified*: `reproduction/` runs their released code and recovers their
   published numbers to within 7.3% across twenty comparisons, two to four
   decimals. Earlier baselines here were hand-rolled.

2. **Increments that the evidence justifies.** Each one below traces to a paper
   or to a measurement in `analysis/notebooks/E1_dataset_eda.ipynb`.

It runs on Kaggle because the stack is Weng et al.'s pinned one — `torch==2.1.2`,
which has no Python 3.12 wheel — so the kernel bootstraps a standalone Python
3.11. See `reproduction/REPRODUCIBILITY_MATRIX.md` for the three forced
deviations.

**One architecture per kernel.** DCRNN measures ~9 s/epoch on the largest fold,
so all five together would run 11–14 h against Kaggle's 12 h CPU limit. The five
kernels are identical apart from this one name and run in parallel;
`analysis/_build/merge_sweep.py` joins their outputs into the cross-architecture
table. Running them on GPU instead would mean a CUDA build of `torch==2.1.2` and
a different PyG wheel index — a fourth deviation from the authors' environment,
in a sweep whose claim is that it runs inside theirs.

## The evaluation is ours, not theirs

Architectures from them; protocol from `docs/ROADMAP.md`. Rolling-origin over 3
origins × 3 seeds, window 3 → horizon 3, normalisation from training weeks only,
early stopping on a genuine validation split, **pooled** RMSE on held-out
windows.

Their own evaluation is not used, and the reasons are measured: the column they
report is training-inclusive, their RMSE is averaged per window rather than
pooled (~39% below the pooled value on this target), and their loop has no early
stopping — the final epoch's weights are scored. Numbers here are therefore
**not** comparable to Table I, and **are** comparable to every other row of this
project's ablation table.

## Increments under test

| increment | source | rationale |
|---|---|---|
{chr(10).join(f'| `{n}` | {c} | {r} |' for n, c, r in INCREMENTS)}

All five architectures take all four increments, because they share one head seam
(`analysis/lib/reproduced.py`). Before that seam, `per_horizon_heads` and
`probabilistic` could only have been wired into three of the five — and an
ablation row covering an unstated subset of architectures is worse than no row.

## Every fold is scored twice

EDA F4 found week 395: a 19× reporting spike across 18 of 25 districts, a backlog
rather than an epidemic. Under this protocol it lands in the **test** split of the
origin-0.85 fold and in no training split at all. Measured on that fold:

| test windows | persistence RMSE |
|---|---|
| all 68 | 68.62 |
| the 6 touching week 395 | 219.07 |
| the other 62 | 22.80 |

Six windows — 9% of the fold — carry ~90% of its squared error, and that fold
dominates the pooled average. So every arm below is reported **with and without**
those windows. Neither number alone is honest: including them measures a
reporting backlog, and dropping them silently would flatter every arm equally
while hiding that the fold's difficulty is one bad week.

They are never removed from training, because they are in no training set. An
earlier version of this kernel carried a `mask_artifact` arm that masked
*training* windows; it was inert by construction and has been deleted.

## Deliberately excluded, with reasons

- **week-of-year feature.** EDA F9 originally recommended this and **was
  retracted**: the seasonal shape does not repeat once amplitude is normalised
  (r = −0.065, chance), no district shows a week-of-year effect at p < 0.05, and
  it explains R² = 0.03 against 0.86 from the previous week. EXP-012 reached the
  same conclusion first.
- **wider window.** EXP-012 tested W ∈ {{3, 8, 16, 26, 39}} at 8 folds × 3 seeds:
  no improvement, after retracting an interim claim that there was one.
- **learned adjacency.** EXP-018 and the head-to-head benchmark both tested it.
  The effect flips sign between two implementations on identical folds and seeds
  (−0.47 in the repo's model, +0.45 in ours) — which is what a null looks like.

## What to expect

Every model measured on this dataset so far lands within ±2 RMSE of the
persistence floor — the repo's DenseGCN, our STGNN, and all five of Weng et al.'s
architectures. EDA F6 explains why: cases at *t−1* explain r² = 0.85 while the
best covariate explains 0.02. An increment that moves RMSE by less than the
seed-to-seed spread is not a result.
"""
        ),
        md("## 1. Environment — the authors' pins under a standalone Python 3.11"),
        code(env_setup.PREAMBLE),
        code(env_setup.SETUP),
        code(env_setup.VERIFY_ENV),
        md("## 2. This repository, for the protocol and architecture wrappers"),
        code(
            f'''
ARCH = "{arch}"

PROJECT = "https://github.com/rathishTharusha/dengue-forecasting-gnn.git"
# Pinned explicitly. The reproduction kernels clone the default branch, which
# carries the dataset but not `analysis/lib` -- so an unpinned clone here finds
# no protocol module and fails at the listing below. Change this in
# `reproduction/_build/cells_sweep.py` when the work merges.
BRANCH = "{BRANCH}"
PROJ = SCRATCH / "project"
if not PROJ.exists():
    sh("git", "clone", "--depth", "1", "--branch", BRANCH, PROJECT, str(PROJ))
print("branch:", BRANCH)

DATA = PROJ / "notebooks" / "baseline"
print("data:", sorted(p.name for p in DATA.iterdir() if p.suffix in {{".npy", ".json"}}))

# analysis/lib carries the protocol (adaptive.py), the architecture wrappers
# (reproduced.py) and the increments (improved.py).
LIB = PROJ / "analysis" / "lib"
print("lib :", sorted(p.name for p in LIB.iterdir() if p.suffix == ".py"))
'''
        ),
        md(
            """
## 3. The sweep

Runs in the Python 3.11 environment as a subprocess, because that is where the
pinned stack lives. Each arm is the baseline architecture plus one increment, so
a difference is attributable to that increment alone.

Four details in the code below are worth reading, because three of them were bugs
before they were features:

* **The attention module is inside the checkpoint.** It has parameters, so a
  checkpoint saving only the backbone would restore the best backbone next to the
  *last* epoch's attention weights.
* **The probabilistic arm is scored on its mean.** Its loss is the Gaussian NLL of
  Eq. (12), but RMSE comes from μ, so the number stays comparable to the point
  arms. PICP and MPIW from Eq. (14) are reported separately — they are the only
  thing that arm can actually win on.
* **The test split is scored twice**, all windows and artifact-free, for the models
  and for the persistence floor alike. Scoring the floor on a different window set
  than the models would compare two different problems.
* **Results are written after every run**, not at the end, so a kernel that hits
  the wall clock still leaves behind everything it finished.
"""
        ),
        # SWEEP lands inside a raw triple-quoted string in the notebook cell, so
        # its backslashes must not be escaped on the way in -- doing so turned
        # the script's newlines into literal backslash-n on an earlier attempt.
        code(
            '''
SWEEP = r"""
__SWEEP__
"""

script = SCRATCH / "sweep.py"
script.write_text(SWEEP, encoding="utf-8")
RESULTS_JSON = WORK / f"improved_sweep_{ARCH}.json"
print("wrote", script)
'''.replace("__SWEEP__", SWEEP.strip())
        ),
        code(
            '''
started = time.time()
proc = subprocess.Popen(
    [str(PY311), "-u", str(script), str(LIB), str(DATA), str(RESULTS_JSON), ARCH],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
for line in proc.stdout:
    print(line, end="")
proc.wait()
print(f"\\nsweep exited {proc.returncode} after {(time.time()-started)/60:.1f} min")
'''
        ),
        md(
            f"""
## 4. Results for {arch}

Two tables over the same runs. `all windows` is what the frozen protocol scores;
`artifact-free` drops the six origin-0.85 windows touching week 395 from the
**evaluation** of every arm and of the floor alike.

The cross-architecture table needs all five kernels; build it with
`analysis/_build/merge_sweep.py` once their outputs are downloaded.
"""
        ),
        code(
            '''
import numpy as np

records = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
incs = ["base"] + sorted({r["increment"] for r in records
                          if r["arch"] != "persistence" and r["increment"] != "base"})


def table(field, caption):
    floor = np.mean([r[field] for r in records if r["arch"] == "persistence"])
    print(f"{caption}   persistence floor {floor:.2f}")
    print(f"{'origin':>8s}" + "".join(f"{i[:16]:>18s}" for i in incs))
    for origin in sorted({r["origin"] for r in records if r["arch"] == ARCH}):
        row = ""
        for i in incs:
            v = [r[field] for r in records if r["increment"] == i
                 and r["arch"] == ARCH and r["origin"] == origin]
            row += f"{np.mean(v):18.2f}" if v else f"{'--':>18s}"
        print(f"{origin:8.2f}{row}")
    row = ""
    for i in incs:
        v = [r[field] for r in records if r["increment"] == i and r["arch"] == ARCH]
        row += f"{np.mean(v):18.2f}" if v else f"{'--':>18s}"
    print(f"{'all':>8s}{row}\\n")


table("RMSE", "pooled RMSE, all windows   ")
table("RMSE_clean", "pooled RMSE, artifact-free ")
'''
        ),
        code(
            '''
from scipy import stats

key = lambda r: (r["origin"], r["seed"])
model = [r for r in records if r["arch"] == ARCH]

for field, caption in [("RMSE", "all windows"), ("RMSE_clean", "artifact-free")]:
    basis = {key(r): r[field] for r in model if r["increment"] == "base"}
    print(f"paired against `base` -- same origin and seed -- on {caption}\\n")
    print(f"{'increment':22s}{'mean dRMSE':>12s}{'sd':>8s}{'better':>10s}{'p':>9s}")
    for i in incs:
        if i == "base":
            continue
        pairs = [(r[field], basis[key(r)]) for r in model
                 if r["increment"] == i and key(r) in basis]
        if len(pairs) < 3:
            continue
        d = np.array([a - b for a, b in pairs])
        _, p = stats.ttest_rel([a for a, _ in pairs], [b for _, b in pairs])
        print(f"{i:22s}{d.mean():+12.3f}{d.std(ddof=1):8.3f}"
              f"{int((d < 0).sum()):7d}/{len(d):<3d}{p:9.3f}")
    print()

print("Seed-to-seed spread sets the noise floor; anything inside it is not a result.")
'''
        ),
        code(
            '''
prob = [r for r in model if r["increment"] == "probabilistic" and "PICP" in r]
if prob:
    print(f"Eq. (14) intervals for {ARCH} -- nominal coverage 0.95\\n")
    print(f"{'origin':>8s}{'PICP':>8s}{'MPIW':>10s}")
    for origin in sorted({r["origin"] for r in prob}):
        rows = [r for r in prob if r["origin"] == origin]
        print(f"{origin:8.2f}{np.mean([r['PICP'] for r in rows]):8.3f}"
              f"{np.mean([r['MPIW'] for r in rows]):10.1f}")
    print(f"{'all':>8s}{np.mean([r['PICP'] for r in prob]):8.3f}"
          f"{np.mean([r['MPIW'] for r in prob]):10.1f}")
    print("\\nPICP well below 0.95 means the intervals are too narrow to use;"
          "\\nwell above, with a wide MPIW, means they are honest but uninformative.")
else:
    print("no probabilistic runs recorded")
'''
        ),
        md(
            """
## 5. How to read this

**Against the floor first.** Persistence is printed above every table. An
architecture or increment that does not beat it has not earned its complexity —
`crosscheck/FINDINGS.md` shows persistence beats every model in Weng et al.'s
cross-validated column.

**Paired, not pooled.** The three origins differ in difficulty by a factor of
two, so a standard deviation taken across them is fold difficulty, not model
variance. The paired table removes it.

**The artifact-free table is the one about forecasting.** The all-windows table is
what the frozen protocol reports and is kept for comparability with every earlier
row, but roughly 90% of the origin-0.85 fold's squared error comes from six
windows of backlogged reporting. An increment significant in one table and not the
other is telling you which of the two it acted on, and a gain that appears *only*
in the all-windows table is a gain at predicting a data-entry event.

Note which way the fold ordering flips between the two tables. On all windows,
origin 0.85 is by far the hardest fold; artifact-free, it is the easiest. Any
statement about "the hard fold" has to say which table it came from.

**Expect small numbers.** Every model measured on this dataset lands within ±2
RMSE of persistence. If the increments land there too, that is the finding, and it
points where EDA F6 already points: the ceiling is temporal, and the remaining
headroom is in outbreak extrapolation rather than in architecture.

## 6. What this sweep still does not test

- **The adaptive graph on all five.** AAGCN is the only one of the reproduced set
  that exposes an adaptive switch. `analysis/notebooks/E3_adaptive_graph.ipynb`
  tests a learned adjacency on a controlled encoder instead; the effect there
  flips sign between implementations.
- **Physics-informed loss and GAN augmentation**, the other two contributions in
  `docs/ROADMAP.md`. Both need this baseline settled first, because both are
  measured as a delta against it.
- **Outbreak-period performance specifically.** Pooled RMSE over all district-weeks
  is dominated by the ~90% of weeks that are quiet. EDA F6 puts the remaining
  headroom in the tail, which this table cannot resolve.
"""
        ),
    ]
