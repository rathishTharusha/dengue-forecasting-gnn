"""Cell sources for the adaptive-graph experiment.

A controlled comparison in which the adjacency is the only variable: four graph
modes share one encoder, one training loop, one normalisation and one protocol.
"""

from gen_analysis import code, md

CELLS = [
    md(
        """
# Does a learned graph help? — implementing an adaptive adjacency

**Answer, up front: no.** On this dataset, under a clean held-out protocol, a
learned adjacency does not beat the hand-built one — and *no graph at all* is
statistically indistinguishable from either. The detail is below; the headline is
that this is a negative result, and it is reported as one.

## Why this is its own notebook, and not a flag on the authors' model

The first attempt was to flip `adaptive=False -> True` in Weng et al.'s AAGCN.
That turns out to be impossible, for a reason worth recording. PGT computes

```python
self.inter_c = out_channels // coff_embedding      # coff_embedding defaults to 4
```

and the authors instantiate `AdaptiveGCN(1, 1, num_nodes, edge_index)`, so
`out_channels = 1` and `inter_c = 0`, giving a convolution with weight shape
`[0, 1, 1, 1]`:

```
RuntimeError: Given groups=1, expected weight to be at least 1 at dimension 0,
but got weight of size [0, 1, 1, 1] instead
```

`AAGCN.__init__` does not expose `coff_embedding`, so the only fix is
`out_channels >= 4` — but `out_channels = 1` is load-bearing downstream, where
`train_AAGCN` reshapes predictions to match a `(25, 3)` target:

| out_channels | adaptive | reshaped | matches target |
|---|---|---|---|
| 1 | False | (25, 3) | yes |
| 1 | True | — | crashes |
| 4 | False | (25, 12) | no |
| 4 | True | (25, 12) | no |

So `adaptive=False` in that repository was **forced by their output dimension**,
not chosen on the merits. Testing the idea properly means implementing it
directly — which is what `docs/ROADMAP.md` calls Contribution (c) anyway.

## The experiment

Four graph modes sharing one encoder, one training loop, one normalisation and
one protocol. The adjacency is the only thing that changes.

| mode | adjacency |
|---|---|
| `none` | identity — 25 independent series, no message passing |
| `fixed` | the hand-built district graph, symmetrically normalised |
| `adaptive` | `softmax(ReLU(E1 @ E2.T))`, with `E1`/`E2` learned node embeddings |
| `hybrid` | `0.5 * fixed + 0.5 * adaptive` — how Graph WaveNet actually uses it |

`none` is the control that matters most: it asks whether a graph earns its keep
at all. Without it, "adaptive beats fixed" could just mean "both are noise".

Everything else follows `docs/decisions/0001`: residual over persistence, log1p
target space, normalisation from training weeks only.

**Protocol:** `docs/ROADMAP.md`'s frozen rolling-origin CV — 3 origins x 3 seeds,
window 3 -> horizon 3, scored on held-out windows with **pooled** RMSE. Not Weng
et al.'s segment CV, which is training-inclusive and per-window averaged. Results
here are directly usable as a row of this project's ablation table.

**Two graph defects are corrected in both arms** (EDA F7): the released adjacency
has two one-way edges, and a non-symmetric adjacency relation is a data defect
rather than a modelling choice.

**Prediction, recorded before running:** EDA F6 says cases at *t-1* explain
r^2 = 0.85 while the best covariate explains 0.02; F8 says adjacency adds only
+0.07 correlation over a national baseline of 0.55. So: a small effect, with
`adaptive` and `hybrid` above `fixed` if the graph matters at all.
"""
    ),
    md("## 1. Setup"),
    code(
        """
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path.cwd().parent.parent if (Path.cwd().parent.parent / "notebooks").exists() else Path.cwd()
sys.path.insert(0, str(REPO / "analysis" / "lib"))
RESULTS = REPO / "analysis" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

from adaptive import (  # noqa: E402
    GRAPH_MODES,
    build_folds,
    load_dataset,
    persistence_scores,
    train_one,
)

cases, A, districts = load_dataset(
    REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy",
    REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json",
)
fixed = torch.tensor(A, dtype=torch.float32)
folds = build_folds(cases)

print(f"cases {cases.shape} | adjacency {A.shape} symmetric={np.allclose(A, A.T)}")
for f in folds:
    print(f"  origin {f.origin}: train {f.x_train.shape[0]}, "
          f"val {f.x_val.shape[0]}, test {f.x_test.shape[0]} windows")
"""
    ),
    md(
        """
## 2. What the four modes compute

`STGNN.adjacency` is the whole experiment. The node embeddings are allocated in
**every** arm, used or not, so parameter initialisation consumes identical RNG
draws — otherwise the arms would differ by more than the graph.
"""
    ),
    code(
        """
import inspect

from adaptive import STGNN

print(inspect.getsource(STGNN.adjacency))
"""
    ),
    md("## 3. Run — 4 modes x 3 origins x 3 seeds"),
    code(
        """
import time

RUNS = RESULTS / "adaptive_graph_runs.json"

if RUNS.exists():
    records = json.loads(RUNS.read_text(encoding="utf-8"))
    print(f"loaded {len(records)} cached runs from {RUNS.name}")
else:
    records, started = [], time.time()
    for mode in GRAPH_MODES:
        for fold in folds:
            for seed in (0, 1, 2):
                _, scores = train_one(fold, fixed, mode, seed=seed, epochs=150)
                records.append(dict(mode=mode, origin=fold.origin, seed=seed, **scores))
                print(f"{mode:9s} origin {fold.origin} seed {seed}  "
                      f"RMSE {scores['RMSE']:7.2f}  MAE {scores['MAE']:6.2f}", flush=True)
    for fold in folds:
        records.append(dict(mode="persistence", origin=fold.origin, seed=-1,
                            **persistence_scores(fold, cases)))
    RUNS.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"{time.time() - started:.0f}s -> {RUNS}")
"""
    ),
    md("## 4. Pooled results"),
    code(
        """
ORDER = ["persistence", "none", "fixed", "adaptive", "hybrid"]


def summarise(mode):
    rows = [r for r in records if r["mode"] == mode]
    rmse = np.array([r["RMSE"] for r in rows])
    mae = np.array([r["MAE"] for r in rows])
    return rmse.mean(), rmse.std(), mae.mean(), mae.std(), len(rows)


base = summarise("fixed")
table = {}
print(f"{'mode':13s}{'RMSE':>9s}{'sd':>7s}{'MAE':>9s}{'sd':>7s}{'n':>4s}{'RMSE vs fixed':>16s}")
for mode in ORDER:
    m_rmse, s_rmse, m_mae, s_mae, n = summarise(mode)
    table[mode] = (m_rmse, s_rmse, m_mae, s_mae)
    mark = "" if mode == "fixed" else f"{100 * (m_rmse - base[0]) / base[0]:+15.2f}%"
    print(f"{mode:13s}{m_rmse:9.2f}{s_rmse:7.2f}{m_mae:9.2f}{s_mae:7.2f}{n:4d}{mark:>16s}")
"""
    ),
    md(
        """
Those standard deviations are ~16 RMSE against differences of ~0.5, so the
pooled table cannot resolve anything. The spread is **fold difficulty**, not
model variance: the three origins score around 27, 38 and 65 RMSE respectively.

The right analysis pairs each arm against `fixed` at the **same origin and the
same seed**, which removes fold difficulty entirely.
"""
    ),
    code(
        """
from scipy import stats

by_key = {m: {(r["origin"], r["seed"]): r["RMSE"] for r in records if r["mode"] == m}
          for m in GRAPH_MODES}
ref = by_key["fixed"]

print("paired against `fixed` — same origin, same seed")
print(f"{'mode':10s}{'mean dRMSE':>12s}{'sd':>8s}{'better in':>11s}{'paired t p':>12s}")
for mode in ("none", "adaptive", "hybrid"):
    keys = sorted(set(ref) & set(by_key[mode]))
    diff = np.array([by_key[mode][k] - ref[k] for k in keys])
    _, pval = stats.ttest_rel([by_key[mode][k] for k in keys], [ref[k] for k in keys])
    print(f"{mode:10s}{diff.mean():+12.3f}{diff.std(ddof=1):8.3f}"
          f"{int((diff < 0).sum()):8d}/{len(keys):<3d}{pval:12.3f}")

print()
print("seed-only noise floor (spread across seeds within one origin, mode=fixed)")
for o in sorted({r['origin'] for r in records if r['mode'] == 'fixed'}):
    v = [r["RMSE"] for r in records if r["mode"] == "fixed" and r["origin"] == o]
    print(f"  origin {o}: {np.mean(v):6.2f} +/- {np.std(v, ddof=1):.2f} "
          f"({100 * np.std(v, ddof=1) / np.mean(v):.1f}%)")
"""
    ),
    md("## 5. Per origin, and against the persistence floor"),
    code(
        """
origins = sorted({r["origin"] for r in records if r["mode"] != "persistence"})
print("held-out RMSE per origin (mean over seeds)")
print(f"{'mode':13s}" + "".join(f"{o:>10.2f}" for o in origins))
for mode in ORDER:
    cells = []
    for o in origins:
        vals = [r["RMSE"] for r in records if r["mode"] == mode and r["origin"] == o]
        cells.append(np.mean(vals) if vals else float("nan"))
    print(f"{mode:13s}" + "".join(f"{c:10.2f}" for c in cells))

print()
persist = {r["origin"]: r["RMSE"] for r in records if r["mode"] == "persistence"}
print("versus persistence, paired by origin")
for mode in ("none", "fixed", "adaptive", "hybrid"):
    deltas = [np.mean([r["RMSE"] for r in records
                       if r["mode"] == mode and r["origin"] == o]) - persist[o]
              for o in persist]
    print(f"  {mode:10s} mean dRMSE {np.mean(deltas):+7.2f}   "
          f"beats persistence on {sum(1 for d in deltas if d < 0)}/{len(deltas)} origins")
"""
    ),
    code(
        """
import csv

out = RESULTS / "adaptive_graph_summary.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["mode", "rmse_mean", "rmse_sd", "mae_mean", "mae_sd", "rmse_rel_vs_fixed_pct"])
    for mode in ORDER:
        m_rmse, s_rmse, m_mae, s_mae = table[mode]
        w.writerow([mode, f"{m_rmse:.4f}", f"{s_rmse:.4f}", f"{m_mae:.4f}", f"{s_mae:.4f}",
                    f"{100 * (m_rmse - base[0]) / base[0]:.3f}"])
print("wrote", out)
"""
    ),
    md(
        """
## 6. Verdict

**The adaptive graph does not help.** Paired against the fixed graph on identical
folds and seeds, it is *worse* by +0.46 RMSE (p = 0.09) — not significant, but
certainly not an improvement. `hybrid` is +0.19 (p = 0.37), i.e. indistinguishable.

**Neither does the graph itself.** Removing message passing entirely (`none`) is
−0.54 RMSE against `fixed`, p = 0.49. Twenty-five independent time series do as
well as the spatio-temporal graph model. The seed-only noise floor is 0.6–2.7%,
and every effect here sits at or below it.

This is exactly what EDA F6 and F8 predicted: cases at *t−1* explain r² = 0.85,
adjacency adds only +0.07 correlation over a national baseline of 0.55, and the
Most inter-district co-movement is a shared national trend rather than spatial
transmission — so there is little for any graph, learned or given, to extract.

(An earlier draft attributed that shared movement to seasonality, citing EDA F9.
F9 has since been retracted: there is no usable annual cycle in this dataset. The
shared movement is real; calling it *seasonal* was not supported.)

**One positive result worth keeping.** Under this clean protocol all four model
variants do beat persistence, by 1–2 RMSE, with `adaptive` and `hybrid` beating
it on 3/3 origins. That is a genuine if small improvement — and it contrasts with
`crosscheck/FINDINGS.md`, where persistence beats every model in Weng et al.'s
cross-validated column. The difference is the protocol, not the models.

### What this redirects, and where the evidence points next

Contribution (c) as written — "replace the fixed adjacency with a learned one" —
is not supported by this evidence. Three avenues remain, in the order the
evidence supports them:

1. ~~**A seasonal feature.**~~ **Ruled out.** An earlier draft made this the top
   recommendation on the strength of EDA F9. F9 has been retracted: the seasonal
   shape does not repeat once amplitude is normalised (r = −0.065, chance), no
   district shows a week-of-year effect, and week-of-year explains R² = 0.03
   against 0.86 from the previous week. EXP-012 had already concluded this.
2. **The physics-informed loss.** If features carry no signal and persistence
   captures the autocorrelation, extra accuracy must come from structure rather
   than data. A compartmental prior supplies structure. Note EXP-014's warning:
   with a correctly derived ceiling the constraint became inert, so the term needs
   a mechanism, not just a weight.
3. **Outbreak augmentation.** The failure mode is extrapolation to unseen regimes.
   But EDA F3 and F4 say clean first — a generator trained on this array will
   learn to reproduce a 19× reporting artifact.

### Limits of this result

Three seeds and three origins is thin, and p = 0.09 on `adaptive` is the kind of
number that moves with more seeds. The claim supported is "no evidence of
improvement", not "proof of no effect". A larger seed count would tighten it, and
would be cheap — the whole experiment runs in about seven minutes.
"""
    ),
]
