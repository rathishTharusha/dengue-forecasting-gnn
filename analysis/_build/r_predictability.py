"""Is the reproduction number ``R_t`` predictable? The physics hypothesis, tested.

Why this is the decisive experiment
-----------------------------------
The renewal equation describes this data well *in hindsight*: with oracle ``R_t``
a 3-week replay scores RMSE 13.44 where persistence scores 55.03. But every
causally available physics anchor loses to persistence, artifact-free:

    persistence            29.52
    force (R = 1)          35.45
    R_hat * force          50.01
    ratio form, damp 0.5   33.05

So the structure is right and the *predictive* content is entirely in ``R_t``.
Everything above estimates ``R_t`` from its own recent past, which is what fails.

That leaves the hypothesis the physics framing actually rests on: **transmission
intensity is driven by climate, not by case counts.** EDA F6 found the best
covariate explains r^2 = 0.02 -- but that was for predicting *cases*, which are
dominated by their own level (lag-1 r = 0.92). ``R_t`` is the normalised
transmission rate with the level divided out, and it is the quantity mosquito
ecology should actually drive. Nobody has regressed it.

If climate explains a useful share of ``R_t``, a network predicting ``R`` from
covariates has real headroom and the decoder is worth pushing. If it explains
next to nothing, the physics direction is exhausted and this project's five
previous physics experiments were right for a reason that can finally be stated.

Also tests the other route the graph could supply: whether a district's neighbours
carry information about its ``R`` beyond its own history.

Run::

    C:/Users/tharu/rp/v/Scripts/python.exe analysis/_build/r_predictability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "src"))

import adaptive as base  # noqa: E402
import renewal  # noqa: E402
from dengue_gnn import seir  # noqa: E402

NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
OUT = REPO / "analysis" / "results" / "r_predictability.json"

#: docs/DATA.md. Covariates already carry their optimal lag; index 5 is the target.
CHANNELS = {
    0: "meanTair", 1: "minTair", 2: "maxTair", 3: "meanQair(humidity)",
    4: "soilMoisture", 6: "canopyInt(+12wk)", 7: "meanPrecip(+12wk)",
    8: "minPrecip(+12wk)", 9: "maxPrecip(+12wk)", 10: "minNdvi(+17wk)",
}


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Coefficient of determination against the mean of ``y``."""
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares fit with an intercept; returns in-sample predictions."""
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return design @ coef


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raw = np.nan_to_num(np.load(NPY, allow_pickle=True)).astype(np.float64)
    cases, _, names = base.load_dataset(NPY, ADJ)
    w = renewal.generation_interval(seir.PAPER_SECTION4)

    # Back-solve R over the whole record, then work in log R: it is the scale the
    # decoder parameterises, roughly symmetric, and not bounded below at 0.
    weeks, districts = cases.shape
    force = np.zeros_like(cases)
    for lag, weight in enumerate(w, start=1):
        force[lag:] += weight * cases[: weeks - lag]
    usable = (force >= 5.0) & (cases > 0)     # enough signal for the ratio to mean something
    log_r = np.full_like(force, np.nan)
    log_r[usable] = np.log(cases[usable] / force[usable])

    print(f"log R_t usable on {usable.sum()} of {usable.size} district-weeks "
          f"({usable.mean():.0%})")
    print(f"  sd(log R) = {np.nanstd(log_r):.3f}\n")

    report: dict = {}

    # 1. How much does R's own past explain? This is the ceiling the
    #    "estimate R from recent history" anchors were implicitly working against.
    prev = np.full_like(log_r, np.nan)
    prev[1:] = log_r[:-1]
    ok = np.isfinite(log_r) & np.isfinite(prev)
    own = r2(log_r[ok], ols(prev[ok][:, None], log_r[ok]))
    print(f"{'predictor of log R_t':32s}{'r^2':>8s}")
    print(f"{'-' * 40}")
    print(f"{'log R_{t-1} (own past)':32s}{own:8.3f}")
    report["own_past"] = own

    # 2. Each covariate alone. These are the physically motivated drivers.
    per_channel = {}
    for idx, name in CHANNELS.items():
        cov = raw[..., idx]
        ok = np.isfinite(log_r) & np.isfinite(cov)
        val = r2(log_r[ok], ols(cov[ok][:, None], log_r[ok]))
        per_channel[name] = val
        print(f"{name:32s}{val:8.3f}")
    report["per_channel"] = per_channel

    # 3. All covariates together.
    cov = np.stack([raw[..., i] for i in CHANNELS], axis=-1)
    ok = np.isfinite(log_r)
    allcov = r2(log_r[ok], ols(cov[ok], log_r[ok]))
    print(f"{'all 10 covariates':32s}{allcov:8.3f}")
    report["all_covariates"] = allcov

    # 4. Covariates plus R's own past -- does climate add anything on top?
    both = np.column_stack([cov[ok], prev[ok]])
    okb = np.isfinite(both).all(axis=1)
    combined = r2(log_r[ok][okb], ols(both[okb], log_r[ok][okb]))
    print(f"{'all covariates + own past':32s}{combined:8.3f}")
    report["covariates_plus_own_past"] = combined

    # 5. Neighbours: mean log R of adjacent districts, same week. This is what the
    #    graph could contribute if transmission intensity is spatially coherent.
    adj = json.loads(Path(ADJ).read_text(encoding="utf-8"))
    index = {n: i for i, n in enumerate(names)}
    neigh = np.full_like(log_r, np.nan)
    for district, others in adj.items():
        cols = [index[o] for o in others if o in index]
        if cols:
            neigh[:, index[district]] = np.nanmean(log_r[:, cols], axis=1)
    ok = np.isfinite(log_r) & np.isfinite(neigh)
    nb = r2(log_r[ok], ols(neigh[ok][:, None], log_r[ok]))
    print(f"{'neighbours mean log R, same wk':32s}{nb:8.3f}")
    report["neighbours_same_week"] = nb

    # The honest version of the neighbour test: last week's neighbours, which is
    # the only thing available at forecast time.
    nprev = np.full_like(neigh, np.nan)
    nprev[1:] = neigh[:-1]
    ok = np.isfinite(log_r) & np.isfinite(nprev)
    nbp = r2(log_r[ok], ols(nprev[ok][:, None], log_r[ok]))
    print(f"{'neighbours mean log R, t-1':32s}{nbp:8.3f}")
    report["neighbours_lagged"] = nbp

    print(f"\nFor scale: cases at t-1 explain r^2 = 0.85 of cases at t (EDA F6).")
    print("A driver of R worth building on needs to be visible here, not at 0.0x.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
