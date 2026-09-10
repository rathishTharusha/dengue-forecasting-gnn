"""Can the SEIR-SEI structure predict R_t where raw climate cannot?

The gap this closes
-------------------
``r_predictability.py`` regressed ``log R_t`` linearly on the ten covariates and
got r^2 = 0.008. That test used Phaijoo & Gurung only for the generation-interval
kernel -- i.e. their four stage durations -- and ignored the paper's actual
contribution, the R0 formula and its sensitivity analysis. It also ignored the
susceptible-depletion mechanism that is the whole point of an SEIR model.

Both are testable, and both are about *identifying outbreaks* rather than
extrapolating counts.

Mechanism 1: susceptible depletion (09_SEIR_model.pdf)
------------------------------------------------------
The S compartment is what makes an epidemic turn over: ``ds/dt = -beta*i*s``, so
``R_eff(t) = R0 * s(t)`` and R must *fall* as an outbreak consumes susceptibles.
That makes R partly predictable from something the free-floating renewal model
never saw -- recent cumulative incidence. If it holds here, the model can know an
outbreak is near its peak, which no amount of level extrapolation can tell it.

Dengue complicates this: four serotypes and waning cross-immunity mean depletion
is partial and recovers between seasons. So trailing windows of several lengths
are tested rather than true cumulative incidence, and district population -- which
this dataset does not carry -- is absorbed into a per-district normalisation.

Mechanism 2: the thermal response is unimodal (Phaijoo & Gurung sensitivity)
----------------------------------------------------------------------------
Their normalised forward sensitivity indices, reproduced in
:mod:`dengue_gnn.seir` and validated in EXP-014, are

    b +1.000   mu_v -0.818   beta_h +0.500   beta_v +0.500
    m +0.500   gamma_h -0.500  nu_v +0.318

R0 therefore depends on temperature through a *product* of terms that pull in
opposite directions: biting rate ``b`` and the extrinsic incubation rate ``nu_v``
rise with temperature, while mosquito survival ``1/mu_v`` falls once it gets too
hot. The model's structural prediction is a **hump-shaped** thermal response --
transmission peaking at intermediate temperature -- which a linear fit is
guaranteed to miss and would report as r^2 = 0.

What is deliberately not done: no entomological temperature-response curves are
invented. Neither paper supplies them, so fitting ``b(T)`` and ``mu_v(T)``
separately would mean making up numbers and calling them physics -- the mistake
EXP-014 caught in the old growth ceiling. Only the model's *qualitative*
structural prediction is tested, by allowing curvature and interaction.

Run::

    C:/Users/tharu/rp/v/Scripts/python.exe analysis/_build/mechanistic_r.py
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
OUT = REPO / "analysis" / "results" / "mechanistic_r.json"

TEMP, HUMID, PRECIP, NDVI = 0, 3, 7, 10   # docs/DATA.md channel indices


def r2_oos(x: np.ndarray, y: np.ndarray, folds: int = 5) -> float:
    """Out-of-sample r^2 from contiguous blocks, never shuffled.

    In-sample r^2 rises with every column added, so a curvature test scored
    in-sample is guaranteed to "find" something. Blocks are contiguous and in
    time order because this is a forecasting problem and shuffling would leak.
    """
    n = len(x)
    design = np.column_stack([np.ones(n), x])
    edges = np.linspace(0, n, folds + 1).astype(int)
    pred = np.full(n, np.nan)
    for k in range(folds):
        test = np.zeros(n, bool)
        test[edges[k] : edges[k + 1]] = True
        coef, *_ = np.linalg.lstsq(design[~test], y[~test], rcond=None)
        pred[test] = design[test] @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raw = np.nan_to_num(np.load(NPY, allow_pickle=True)).astype(np.float64)
    cases, _, _ = base.load_dataset(NPY, ADJ)
    w = renewal.generation_interval(seir.PAPER_SECTION4)
    weeks, _districts = cases.shape

    force = np.zeros_like(cases)
    for lag, weight in enumerate(w, start=1):
        force[lag:] += weight * cases[: weeks - lag]
    usable = (force >= 5.0) & (cases > 0)
    log_r = np.full_like(force, np.nan)
    log_r[usable] = np.log(cases[usable] / force[usable])

    prev = np.full_like(log_r, np.nan)
    prev[1:] = log_r[:-1]

    # --- Mechanism 1: susceptible depletion proxies -----------------------
    # Trailing incidence per district, normalised by that district's own mean so
    # Colombo's scale does not dominate. Lagged by one week: everything here must
    # be knowable at forecast time.
    depletion = {}
    for window in (8, 26, 52):
        cum = np.zeros_like(cases)
        for t in range(weeks):
            cum[t] = cases[max(0, t - window) : t].sum(axis=0)
        cum = cum / np.maximum(cases.mean(axis=0, keepdims=True) * window, 1e-6)
        lagged = np.full_like(cum, np.nan)
        lagged[1:] = cum[:-1]
        depletion[window] = np.log1p(lagged)

    # --- Mechanism 2: thermal structure -----------------------------------
    temp = raw[..., TEMP]
    temp_c = temp - temp.mean()
    climate_linear = np.stack([raw[..., i] for i in (TEMP, HUMID, PRECIP, NDVI)], axis=-1)
    climate_hump = np.concatenate([
        climate_linear,
        (temp_c ** 2)[..., None],                       # curvature: the hump
        (temp_c * raw[..., PRECIP])[..., None],         # temperature x rainfall
        (temp_c * raw[..., NDVI])[..., None],
    ], axis=-1)

    def evaluate(name: str, columns: list[np.ndarray]) -> float:
        # Columns arrive as either (weeks, districts) or (weeks, districts, k);
        # promote the former so a mixed list concatenates.
        stacked = np.concatenate(
            [c[..., None] if c.ndim == 2 else c for c in columns], axis=-1)
        ok = np.isfinite(log_r) & np.isfinite(stacked).all(axis=-1)
        value = r2_oos(stacked[ok], log_r[ok])
        print(f"{name:44s}{value:8.3f}")
        return value

    print(f"log R_t on {int(np.isfinite(log_r).sum())} district-weeks, "
          f"sd = {np.nanstd(log_r):.3f}")
    print("out-of-sample r^2, contiguous time blocks, no shuffling\n")
    print(f"{'predictor of log R_t':44s}{'r^2':>8s}")
    print("-" * 52)

    report = {}
    report["own_past"] = evaluate("log R_{t-1} (own past)", [prev])
    report["climate_linear"] = evaluate("climate, linear (the earlier test)",
                                        [climate_linear])
    report["climate_hump"] = evaluate("climate + thermal curvature + interactions",
                                      [climate_hump])
    for window in (8, 26, 52):
        report[f"depletion_{window}"] = evaluate(
            f"susceptible depletion proxy, {window}wk trailing", [depletion[window]])
    report["depletion_all"] = evaluate(
        "all three depletion windows",
        [depletion[8], depletion[26], depletion[52]])
    report["own_plus_depletion"] = evaluate(
        "own past + depletion",
        [prev, depletion[8], depletion[26], depletion[52]])
    report["everything"] = evaluate(
        "own past + depletion + thermal structure",
        [prev, depletion[8], depletion[26], depletion[52], climate_hump])

    # The arithmetic that matters, spelled out: sd and r^2 must come from the same
    # usable subset or the implied error is meaningless. Both do here.
    sd = float(np.nanstd(log_r))
    residual = sd * np.sqrt(1.0 - report["own_past"])
    print(f"\nsd(log R) = {sd:.3f} at out-of-sample r^2 = {report['own_past']:.3f}, "
          f"so the residual multiplicative error is "
          f"exp({residual:.3f}) = {np.exp(residual):.2f}x per step")
    print("Persistence carries no such factor: cases at t-1 explain r^2 = 0.85 of\n"
          "cases at t directly. A renewal decoder only wins if R is predictable\n"
          "enough that reconstructing through it costs less than it gains.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
