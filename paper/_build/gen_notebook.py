"""Generate ``paper/reproduce_paper.ipynb``.

Repo convention: notebooks are build outputs. Edit the cell sources here and
regenerate; never hand-edit the ``.ipynb``.

    python paper/_build/gen_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
OUT = PAPER / "reproduce_paper.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md(
        """
# Reproducing the paper, end to end

Run this top to bottom and it regenerates **every figure, every generated table
and every number** quoted in `paper/`. It finishes in seconds on a laptop.

## What it does and does not do

This notebook is the *presentation* layer. It reads the JSON that each experiment
wrote into `analysis/results/` and turns it into figures and LaTeX. It does **not**
retrain anything, and that is deliberate: a figure pipeline that retrains models is
a figure pipeline nobody reruns, and numbers that are expensive to regenerate
quietly stop matching the text.

The experiments themselves are separate, need the pinned stack
(`torch==2.1.2` / PyG 2.4.0 / `torch-geometric-temporal` 0.54.0), and take minutes
to hours. Section 7 lists the exact command behind every input file, so any number
here can be traced back to the job that produced it.

**Requirements for this notebook:** `numpy`, `matplotlib`. No torch, no GPU.
"""
    ),
    md("## 1. Setup"),
    code(
        """
import json
import sys
from pathlib import Path

REPO = Path.cwd()
while not (REPO / "analysis" / "results").is_dir() and REPO != REPO.parent:
    REPO = REPO.parent
sys.path.insert(0, str(REPO / "paper" / "_build"))

import figures as F                       # noqa: E402

F.style()
print("repo    :", REPO)
print("results :", len(list((REPO / "analysis" / "results").glob("*.json"))), "JSON files")
"""
    ),
    code(
        """
# Every input this notebook consumes, and the experiment that wrote it.
SOURCES = {
    "improved_sweep":      "EXP-019  five architectures x four increments, Kaggle",
    "reproduced_baseline": "EXP-020  the same architectures + AAGCN adaptive, local",
    "error_diagnosis":     "EXP-021  where the baseline error lives",
    "renewal_feasibility": "EXP-023  renewal equation vs the record, no training",
    "physics_STGAT":       "EXP-023  physics arms on STGAT",
    "r_predictability":    "EXP-024  is R predictable (in-sample reference)",
    "mechanistic_r":       "EXP-024  depletion + thermal structure, out-of-sample",
    "outbreak_signal":     "EXP-025  outbreak detectability, single scorers",
    "response_diagnosis":  "EXP-025  damping, conditional response, Moran's I",
    "paper_measurements":  "consolidated numbers the paper quotes",
}
missing = [k for k in SOURCES if not (REPO / "analysis" / "results" / f"{k}.json").exists()]
assert not missing, f"missing result files: {missing}"
for name, what in SOURCES.items():
    print(f"  {name:22s} {what}")
"""
    ),
    md(
        """
## 2. The evaluation artifact (Fig. 1)

Week 395 is a 19x spike in reported cases across 18 of 25 districts simultaneously
-- an administrative backlog, not an epidemiological event. Under our protocol it
falls in the **test** split of the origin-0.85 fold and in **no** training split.

The consequence is the paper's protocol contribution: the fold ordering inverts
depending on whether those six windows are included, so every arm and the baseline
are scored twice.
"""
    ),
    code(
        """
fig = F.fig_artifact()

records = F.load("improved_sweep")
floors = {}
for field, label in (("RMSE", "all windows"), ("RMSE_clean", "artifact-free")):
    per_origin = {}
    for r in records:
        if r["arch"] == "persistence":
            per_origin[r["origin"]] = r[field]
    floors[field] = sum(per_origin.values()) / len(per_origin)
    detail = "  ".join(f"o{o}: {v:6.2f}" for o, v in sorted(per_origin.items()))
    print(f"persistence, {label:14s} mean {floors[field]:6.2f}   |  {detail}")

print("\\nThe hardest fold on one row is the easiest on the other. That is the point.")
"""
    ),
    md(
        """
## 3. Architectures and increments (Fig. 2, Table 1)

Five reproduced architectures, four evidence-led increments, 3 origins x 3 seeds.
The question is not which cell is smallest -- it is whether any increment separates
from its own baseline once the comparison is paired.
"""
    ),
    code(
        """
fig = F.fig_results()
"""
    ),
    code(
        """
import numpy as np
from scipy import stats

model = [r for r in records if r["arch"] != "persistence"]
key = lambda r: (r["arch"], r["origin"], r["seed"])

for field, label in (("RMSE", "all windows"), ("RMSE_clean", "artifact-free")):
    basis = {key(r): r[field] for r in model if r["increment"] == "base"}
    print(f"paired against `base`, {label}")
    print(f"  {'increment':22s}{'mean dRMSE':>12s}{'sd':>8s}{'better':>10s}{'p':>9s}")
    for inc in F.INC_ORDER[1:]:
        pairs = [(r[field], basis[key(r)]) for r in model
                 if r["increment"] == inc and key(r) in basis]
        d = np.array([a - b for a, b in pairs])
        _, p = stats.ttest_rel([a for a, _ in pairs], [b for _, b in pairs])
        print(f"  {inc:22s}{d.mean():+12.3f}{d.std(ddof=1):8.3f}"
              f"{int((d < 0).sum()):7d}/{len(d):<3d}{p:9.3f}")
    print()

print("No increment reaches p < 0.05 on either column. The two smallest p-values")
print("belong to increments that are WORSE than baseline.")
"""
    ),
    code(
        """
# The one increment that earns its place, on a different axis.
prob = [r for r in records if r["increment"] == "probabilistic" and "PICP" in r]
if prob:
    print(f"{'architecture':14s}{'PICP':>8s}{'MPIW':>10s}   (nominal coverage 0.95)")
    for a in F.ARCH_ORDER:
        rows = [r for r in prob if r["arch"] == a]
        if rows:
            print(f"{a:14s}{np.mean([r['PICP'] for r in rows]):8.3f}"
                  f"{np.mean([r['MPIW'] for r in rows]):10.1f}")
"""
    ),
    md(
        """
## 4. The mechanistic prior (Fig. 3, Table 2)

The renewal equation `y_t = R_t * sum_s w_s y_{t-s}`, with the generation interval
derived from the SEIR-SEI stage durations of Phaijoo & Gurung. It reconstructs the
record given oracle `R_t` and loses as a forecaster -- and the gap between those two
facts is the paper's main negative result.
"""
    ),
    code(
        """
fig = F.fig_physics()

feas = F.load("renewal_feasibility")
m = F.load("paper_measurements")
mr = F.load("mechanistic_r")

gi = feas["generation_interval_weeks"]
print("SEIR-SEI generation interval (weeks 1..%d):" % len(gi))
print("  " + "  ".join(f"{v:.3f}" for v in gi) + f"   mean {feas['mean_gi_weeks']:.2f} wk")
print(f"\\noracle-R replay RMSE      {feas['oracle_replay_rmse']:6.2f}")
print(f"persistence, same windows {feas['persistence_rmse_same_windows']:6.2f}")

sd = m["lead_lag"]["sd_log_r"]
r2 = mr["own_past"]
err = np.exp(sd * np.sqrt(1 - r2))
print(f"\\nsd(log R) = {sd:.3f}, out-of-sample r2 = {r2:.3f}")
print(f"  -> residual multiplicative error per step = exp({sd * np.sqrt(1 - r2):.3f})"
      f" = {err:.2f}x")
print("Persistence carries no such factor: lag-1 explains r2 = 0.85 directly.")
"""
    ),
    code(
        """
# Susceptible depletion, at the magnitude that decides it.
cases_total = 504_722          # reported cases over the 8.8-year record
POP = 21_900_000               # Sri Lanka, ~2020
print(f"{'ascertainment':>16s}{'infections':>13s}{'% of pop':>10s}{'d log R':>10s}")
for factor in (1, 5, 10, 20):
    frac = cases_total * factor / POP
    print(f"{'1 in ' + str(factor):>16s}{cases_total * factor:>13,.0f}"
          f"{100 * frac:>9.1f}%{abs(np.log(max(1 - frac, 1e-6))):>10.3f}")
print(f"\\nThose are CUMULATIVE over ~460 weeks, against a weekly sd(log R) of {sd:.3f}.")
print("Even at 1-in-20 ascertainment the drift is ~0.0013/week -- swamped ~500x.")
"""
    ),
    md(
        """
## 5. The information ceiling (Fig. 4)

Not used in the four-page version -- the text carries these numbers -- but generated
because it is the clearest single statement of why no loss term moves the point
forecast. Useful for slides and for an extended write-up.
"""
    ),
    code(
        """
fig = F.fig_ceiling()

resp = F.load("response_diagnosis")
diag = F.load("error_diagnosis")["arms"]
print(f"growth response slope, predicted on true : {resp['damping_slope_vs_truth']:.3f}")
print(f"                       predicted on logR : {resp['damping_slope_vs_log_rhat']:.3f}")
print("  (1.0 would be undamped; 0.0 is a flat forecast)\\n")
for a, d in diag.items():
    o = d["outbreak"]
    print(f"{a:8s} outbreak windows {o['share_of_windows']:.1%} of data, "
          f"{o['share_of_squared_error']:.1%} of squared error, "
          f"bias {d['bias']['outbreak_mean']:+.2f}")
print(f"\\nMoran's I  log cases {resp['morans_i_log_cases']:.3f}   "
      f"log R {resp['morans_i_log_r']:.3f}")
print("R has no local spatial structure, so a Laplacian penalty on it is unsupported.")
"""
    ),
    md(
        """
## 6. Outbreak detection (Fig. 5, Table 3)

The failure above is specific to *counting*. Ranking is a different question, and
the mechanistic quantity that cannot forecast a count is the largest single
addition to a detector.
"""
    ),
    code(
        """
fig = F.fig_detection()

det = m["detection"]
base = det["level"]["auc"]
print(f"{'predictors':34s}{'AUC':>8s}{'AvgPrec':>10s}{'dAUC':>8s}")
for name, d in det.items():
    print(f"{name:34s}{d['auc']:8.3f}{d['ap']:10.3f}{d['auc'] - base:+8.3f}")

ep = m["episodes"]
print(f"\\noutbreak episodes: {ep['n_episodes']}, "
      f"{ep['at_least_3_weeks']} lasting >= 3 weeks, "
      f"median {ep['median_weeks']:.0f} week(s)")
print("The median episode is one week long -- a sustained-exceedance definition")
print("would be more operationally meaningful, and is the stated next step.")
"""
    ),
    md("## 7. Rebuild the paper's tables and validate the sources"),
    code(
        """
figs, tables = F.build_all()
print()
print(tables[:600] + "\\n...")
"""
    ),
    code(
        """
import subprocess

print(subprocess.run([sys.executable, str(REPO / "paper" / "_build" / "validate.py")],
                     capture_output=True, text=True).stdout)
"""
    ),
    md(
        """
## 8. Regenerating the inputs

Everything above reads saved results. To regenerate those results, run the jobs
below with the pinned stack (`python reproduction/verify_local.py --env-only`).
Times are for a laptop CPU.

| output | command | time |
|---|---|---|
| `improved_sweep_<ARCH>.json` | five Kaggle kernels under `reproduction/kaggle/kernels/improved-architecture-sweep-*`, then `python analysis/_build/merge_sweep.py` | ~0.3-10 h each, in parallel |
| `reproduced_baseline.json` | `python analysis/_build/run_reproduced_baseline.py` | ~30 min |
| `error_diagnosis.json` | `python analysis/_build/diagnose_errors.py` | ~10 min |
| `renewal_feasibility.json` | `python analysis/_build/renewal_feasibility.py` | seconds |
| `physics_STGAT.json` | `python analysis/_build/run_physics.py --arch STGAT` | ~5 min |
| `r_predictability.json` | `python analysis/_build/r_predictability.py` | seconds |
| `mechanistic_r.json` | `python analysis/_build/mechanistic_r.py` | seconds |
| `outbreak_signal.json` | `python analysis/_build/outbreak_signal.py` | seconds |
| `response_diagnosis.json` | `python analysis/_build/response_diagnosis.py` | ~10 min |
| `paper_measurements.json` | `python analysis/_build/paper_measurements.py` | ~2 min |

Then compile the PDF (TeX runs in Docker; see `paper/Makefile`):

```bash
cd paper && make          # main.pdf
cd paper && make pages    # page count, must be <= 4 + references
```

The experiment log in `docs/EXPERIMENT_LOG.md` carries the full config, seeds and
unrounded table for every one of these, newest first.
"""
    ),
]


def main() -> int:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
