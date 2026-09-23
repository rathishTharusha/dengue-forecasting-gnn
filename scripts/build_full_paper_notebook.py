"""Generate ``full_paper/reproduce_full_paper.ipynb``.

The notebook it replaces trained models inline and reported a leaderboard from
them. That leaderboard is withdrawn, for the four defects recorded in EXP-032
plus one the S5 script did not have: it normalised with

    mean_c = float(np.mean(cases_matrix)); std_c = float(np.std(cases_matrix))

over the **entire** series, test weeks included, so every split was standardised
with statistics computed from its own targets. It also took one full-batch
gradient step per epoch for 25 epochs on a single 70/85 split rather than the
protocol's rolling origins.

The replacement does not train. It renders the committed confirmatory result --
nine origins with disjoint test spans, produced by the leakage-controlled
``seirgnn2`` harness -- and recomputes the paired tests inline so a reader can
check the inference rather than take it on trust. Retraining is a documented
command, not an inline cell, because a notebook cannot carry the frozen protocol
and a reader cannot tell a leaky split from a clean one by looking.

Self-contained by design: numpy, pandas and matplotlib only, reading
``data/*.csv`` and ``outputs/confirm.json`` relative to ``full_paper/``. No torch,
no repo-root imports.

Run::

    python scripts/build_full_paper_notebook.py
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "full_paper"
OUT = PKG / "reproduce_full_paper.ipynb"


def md(text: str) -> dict:
    return _cell("markdown", text)


def code(text: str) -> dict:
    return _cell("code", text)


def _cell(kind: str, text: str) -> dict:
    lines = text.strip("\n").split("\n")
    source = [f"{ln}\n" for ln in lines[:-1]] + [lines[-1]]
    cell = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": source}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


INTRO = """
# SEIR-GNN — reproducing the reported results

Physics-informed graph neural networks for district-level weekly dengue
forecasting in Sri Lanka, 25 districts. This notebook reproduces every SEIR-GNN
number the package reports, and reports them as what they are.

## What this notebook no longer does

Earlier versions trained models inline and printed a leaderboard from them. That
leaderboard is **withdrawn**. The code behind it took one full-batch gradient
step per epoch, minimised SMAPE while being scored by RMSE, held the force of
infection constant across the forecast horizon, collapsed every covariate to a
scalar before the graph saw it — and standardised the whole series, test weeks
included, with statistics computed from its own targets:

```python
mean_c = float(np.mean(cases_matrix))   # over ALL T, including test
cases_norm = (cases_matrix - mean_c) / std_c
```

A model selected under that normalisation has seen its own evaluation data.
See `docs/EXPERIMENT_LOG.md`, EXP-032 and EXP-037.

## What it does instead

It renders the **confirmatory result**: nine forecast origins whose test spans do
not overlap, five arms frozen before the run, three seeds each, selection on
validation only, produced by the leakage-controlled `seirgnn2` harness. The
paired tests are recomputed here from the raw per-run rows, so you can check the
inference rather than take it on trust.

Retraining is a command, not a cell — `python seirgnn2/sweep.py confirm` from the
repository root. A notebook cannot carry the frozen protocol, and a reader cannot
tell a leaky split from a clean one by looking at one.

**Requires** numpy, pandas, matplotlib. No torch. Runs in seconds.
"""

SETUP = """
import itertools, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd() / "full_paper"
RESULTS = HERE / "outputs" / "confirm.json"

rows = json.loads(RESULTS.read_text(encoding="utf-8"))
df = pd.DataFrame(rows)
print(f"{len(df)} runs | arms: {df.name.nunique()} | origins: {sorted(df.origin.unique())}")
print(f"seeds: {sorted(s for s in df.seed.unique() if s >= 0)}  (persistence is deterministic, seed=-1)")
"""

DATA = """
cases = pd.read_csv(HERE / "data" / "dengue_cases_raw.csv")
print(f"case series: {cases.shape[0]} rows x {cases.shape[1]} columns")
cases.head(3)
"""

LEADER = """
lead = (df.groupby("name")
          .agg(val_RMSE=("val_RMSE", "mean"), test_RMSE=("RMSE", "mean"), n=("name", "size"))
          .sort_values("val_RMSE"))
print(lead.round(2).to_string())
print()
print("Selection is on validation only. Read the next cell before using these means:")
print("they are not a usable summary of this grid.")
"""

OUTLIER = """
# Why the means above must not be read on their own.
piv = df.pivot_table(index="origin", columns="name", values="val_RMSE", aggfunc="mean")
order = lead.index.tolist()
print(piv[order].round(2).to_string())
print()
worst = piv.mean(axis=1).idxmax()
print(f"Origin {worst} is an outlier: every arm fails there, by roughly an order of")
print("magnitude, and it dominates every mean and standard deviation in the table.")
print("A variance claim computed across these folds is a claim about that one fold.")
print("Read win counts and per-origin values instead.")
"""

STATS = """
def paired(arm: str, ref: str, metric: str) -> dict:
    \"\"\"Difference per origin (seeds averaged), and an exact sign-flip p-value.

    The origin is the only thing resampled from the data, so it is the unit the
    plan clusters on. Nine origins put the smallest attainable two-sided p at
    2/2**9 = 0.004; three origins would cap it at 0.25.
    \"\"\"
    a = df[df.name == arm].groupby("origin")[metric].mean()
    b = df[df.name == ref].groupby("origin")[metric].mean()
    d = (a - b).dropna().to_numpy()
    flips = np.array(list(itertools.product((-1, 1), repeat=len(d))), dtype=float)
    return {"delta": d.mean(), "wins": f"{int((d < 0).sum())}/{len(d)}",
            "p": float((np.abs(flips @ d / len(d)) >= abs(d.mean()) - 1e-12).mean())}


def bh(p, m):
    \"\"\"Benjamini-Hochberg over a family of size m (m may exceed len(p)).\"\"\"
    p = np.asarray(p, dtype=float); adj = np.empty(len(p)); run = 1.0
    for rank, i in reversed(list(enumerate(np.argsort(p), start=1))):
        run = min(run, p[i] * m / rank); adj[i] = min(run, 1.0)
    return adj


# The family is fixed by docs/SEIR_GNN_EXPERIMENT_PLAN.md S9 and is NOT chosen here.
# It names five tests; the fifth, the early-warning AUC comparison, was never validly
# computed and is withdrawn. The family size stays 5 regardless -- dropping a member
# after the fact shrinks every other member's adjusted p, and choosing m to cross a
# threshold is exactly the practice this project exists to catch. Both are shown.
S_STAR, B_STAR = "ASTGCN+foi_res", "AAGCN+direct"
FAMILY = [("S* vs B*", S_STAR, B_STAR),
          ("S* vs persistence", S_STAR, "persistence"),
          ("S* vs direct control", S_STAR, "ASTGCN+direct"),
          ("S* vs SEIR-LSTM", S_STAR, "LSTM+foi_res")]

for metric, label in (("RMSE", "TEST -- the pre-registered endpoint metric"),
                      ("val_RMSE", "VALIDATION -- the selection metric, exploratory only")):
    res = pd.DataFrame([{"comparison": n, **paired(a, b, metric)} for n, a, b in FAMILY])
    res["p_adj (m=5)"] = bh(res.p, 5)
    res["p_adj (m=4)"] = bh(res.p, 4)
    print(f"=== {label} ===")
    print(res.round(3).to_string(index=False))
    print()
"""


ENDPOINT = """
## The pre-registered endpoint

`docs/SEIR_GNN_EXPERIMENT_PLAN.md` S9 fixed all of this before any SEIR-GNN
result existed — the comparison, the metric, the family and the decision rule:

> **Primary endpoint:** S* against B*, **test RMSE**, 9 disjoint origins x 3 seeds,
> paired sign-flip permutation clustered by origin.
> **A "beats the baseline" claim needs all three:** BH-adjusted p < 0.05; a win on
> >= 6/9 origins; the same direction in the frozen 3-origin table.

| criterion | required | observed | met |
|---|---|---|---|
| BH-adjusted p | < 0.05 | **0.684** | no |
| origins won | >= 6/9 | **5/9** | no |
| same direction as 3-origin table | yes | yes (-2.23) | yes |

**The endpoint is not met**, on two of the three criteria, and not narrowly.

### The validation result, and why it is not the endpoint

On **validation**, S* vs SEIR-LSTM is -0.75 on 8/9 origins with raw p = 0.012 —
the most consistent margin anywhere in this work. It is not the endpoint, for two
reasons that both matter:

1. **Validation is the selection metric** (plan R6). Every arm here was chosen on
   it. A margin measured on the quantity used for selection is not held-out
   evidence.
2. **The plan names test RMSE.** Switching metric after seeing both is the
   deviation the pre-registration exists to prevent.

Its adjusted p is **0.059** with the family at its pre-registered size of five,
and 0.047 if the withdrawn AUC test is dropped. We report 0.059 and do not claim
0.047. Removing a broken test from a family makes every surviving member look
more significant, and the 0.012 raw p is on the wrong metric in any case.
"""


FIG = """
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
sub = lead.drop(index=["persistence"], errors="ignore")
ax.barh(range(len(sub)), sub.val_RMSE, color="#4C72B0")
floor = lead.loc["persistence", "val_RMSE"]
ax.axvline(floor, color="#C44E52", ls="--", lw=1.6, label=f"persistence {floor:.1f}")
ax.set_yticks(range(len(sub)), sub.index)
ax.invert_yaxis()
ax.set_xlabel("validation RMSE (mean over 9 origins)")
ax.set_title("Mean RMSE — dominated by one origin")
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
for arm in order:
    ax.plot(piv.index, piv[arm], marker="o", ms=4, lw=1.2,
            ls="--" if arm == "persistence" else "-", label=arm)
ax.set_yscale("log")
ax.set_xlabel("forecast origin")
ax.set_ylabel("validation RMSE (log scale)")
ax.set_title("Per origin — the honest view")
ax.legend(fontsize=7, frameon=False)

fig.tight_layout()
fig.savefig(HERE / "outputs" / "figures" / "fig_confirmatory.png", dpi=160)
plt.show()
"""


VERDICT = """
## What the numbers support

**The proposed method does not beat its baseline.** The pre-registered endpoint
fails on two of three criteria and is not close: p_adj = 0.684 against a 0.05
bar, 5/9 origins against a 6/9 requirement.

**Against the naive persistence floor — not beaten.** On test, 31.70 against
31.76 with 6/9 origins and p = 0.988. On validation there is a consistent edge
(−2.92, 8/9) that does not survive correction, but validation is the metric every
arm was selected on.

**Against SEIR-LSTM — no, on the endpoint metric.** Test RMSE +0.23 with 4/9
origins; SEIR-LSTM is marginally ahead. The −0.75 on 8/9 origins is a validation
result and is reported as exploratory.

**Against the five published architectures — three of five, and not the best.**
Comfortably better than STGAT, A3TGCN and DCRNN; not better than AAGCN or
ASTGCN. Those three also ran with settings tuned around the two stronger ones,
so that comparison is weak evidence in our favour.

**Withdrawn without replacement.** The early-warning significance claim
(`p = 0.04`) was a typed-in literal, never computed. The AUC figures it was
attached to (0.807 → 0.826) are computed and stand.

### What is worth reporting

A physics-informed graph formulation that is directionally better than the LSTM
formulation it replaces on the selection metric, across 8 of 9 independent
origins, and indistinguishable from it out of sample — beside a naive baseline
that neither of them beats. Stated that way it is an honest negative result with
a measured mechanism, which is what the rest of this project already reports and
what its credibility rests on.

## Reproducing this from scratch

```bash
python seirgnn2/sweep.py confirm --workers 6 --epochs 400   # ~70 min, 6 CPU workers
python seirgnn2/stats.py confirm --ref "LSTM+foi_res" --metric val_RMSE
```

Verified identical on Kaggle — 144 rows both sides, every arm within 0.11 RMSE,
persistence identical to the decimal. `seirgnn2/README.md` documents the harness,
the protocol, and every lever tried with its measured effect.
"""


def main() -> None:
    src = REPO / "seirgnn2" / "results" / "confirm.json"
    dst = PKG / "outputs" / "confirm.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    (PKG / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)

    cells = [
        md(INTRO),
        md("## Setup — load the committed confirmatory rows"), code(SETUP),
        md("## The data the forecasts are made on"), code(DATA),
        md("## Leaderboard"), code(LEADER),
        md("## Per origin, and why the means mislead"), code(OUTLIER),
        md("## Paired tests at the origin unit"), code(STATS),
        md("## Figures"), code(FIG),
        md(ENDPOINT), md(VERDICT),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells)")
    print(f"copied {src.name} -> {dst.relative_to(REPO)}")


if __name__ == "__main__":
    main()
