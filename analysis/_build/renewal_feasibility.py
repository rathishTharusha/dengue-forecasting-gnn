"""Can a renewal equation seeded by Phaijoo & Gurung's SEIR-SEI track this data?

Why this test exists
--------------------
EXP-014 killed the previous physics term for a specific reason: an *upper* bound
on growth is slack against a model that under-reacts, so it contributed exactly
zero gradient. ``analysis/results/error_diagnosis.json`` confirms that on the
current protocol and makes it worse -- A3TGCN's predicted |weekly log growth|
tops out at 0.336 against an observed 3.638, a 7.7x under-reaction, and it runs a
-12.9 case bias on outbreak windows that hold 61.7% of the squared error.

So the constraint has to *drive* growth, not cap it. EXP-014 warned that anything
aimed at under-reaction is not automatically physics. The renewal equation is the
exception: it is a mechanistic identity on the observable itself,

    cases_t  =  R_t * sum_s w_s * cases_{t-s}

where ``w`` is the generation-interval distribution -- which follows from the
SEIR-SEI stage durations this project has already implemented and validated
against the paper's Table 1 (:mod:`dengue_gnn.seir`). No latent compartments are
inferred, which is what made the full ODE residual under-determined.

Before building anything on it, this script asks the cheap question: with ``w``
fixed by the epidemiology, does the equation actually describe these 25 series?
It fits nothing and trains nothing -- it back-solves ``R_t`` from observed cases
and checks whether the result is epidemiologically sane and whether replaying the
equation reproduces the data.

If ``R_t`` comes out wild or the replay diverges, the approach is dead and no
amount of network is going to rescue it.

Run::

    C:/Users/tharu/rp/v/Scripts/python.exe analysis/_build/renewal_feasibility.py
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

from dengue_gnn import seir  # noqa: E402

NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
OUT = REPO / "analysis" / "results" / "renewal_feasibility.json"

#: Weeks of history the renewal sum looks back over. The SEIR-SEI generation
#: interval is ~2-3 weeks; 6 covers its tail without inventing long-range memory.
KERNEL_WEEKS = 6


def generation_interval_weeks(p: seir.Params) -> np.ndarray:
    """Discrete weekly generation-interval distribution from the SEIR-SEI stages.

    A host-to-host transmission chain passes through four exponential stages:
    host incubation (``1/nu_h``), host infectious period (``1/gamma_h``), vector
    incubation (``1/nu_v``) and the vector's remaining life (``1/mu_v``). The
    generation interval is their sum, so its density is the convolution of four
    exponentials -- computed here by Monte Carlo rather than in closed form,
    because the closed form is numerically delicate when rates nearly coincide.

    Args:
        p: SEIR-SEI parameters, per day.

    Returns:
        Probabilities over lags ``1..KERNEL_WEEKS`` weeks, normalised. Lag 0 is
        excluded: a case cannot infect anyone in the same week it is reported.
    """
    rng = np.random.default_rng(0)
    n = 400_000
    days = (rng.exponential(1 / p.nu_h, n) + rng.exponential(1 / p.gamma_h, n)
            + rng.exponential(1 / p.nu_v, n) + rng.exponential(1 / p.mu_v, n))
    weeks = np.clip(np.ceil(days / 7).astype(int), 1, KERNEL_WEEKS)
    w = np.bincount(weeks, minlength=KERNEL_WEEKS + 1)[1:].astype(float)
    return w / w.sum()


def implied_r(cases: np.ndarray, w: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Back-solve ``R_t`` from observed cases: ``R_t = cases_t / (w * cases)_t``.

    Args:
        cases: ``(weeks, districts)`` raw counts.
        w: Generation-interval probabilities over lags 1..len(w).
        floor: Minimum denominator. Weeks whose recent history is essentially
            empty give a meaningless ratio; they are masked, not clipped, so they
            cannot contribute a fabricated R of 0 or infinity.

    Returns:
        ``(weeks, districts)`` array with ``nan`` where the denominator is below
        ``floor``.
    """
    weeks, _districts = cases.shape
    force = np.zeros_like(cases, dtype=float)
    for lag, weight in enumerate(w, start=1):
        force[lag:] += weight * cases[: weeks - lag]
    r = np.full_like(force, np.nan)
    ok = force >= floor
    r[ok] = cases[ok] / force[ok]
    return r


def replay(cases: np.ndarray, r: np.ndarray, w: np.ndarray, start: int, steps: int) -> np.ndarray:
    """Roll the renewal equation forward from ``start`` using the given ``R_t``.

    This is the reconstruction test: given a perfect R sequence, does the equation
    regenerate the series? It is an upper bound on what any model built on this
    decoder can achieve, because a real model has to predict R rather than be
    handed it.
    """
    hist = list(cases[max(0, start - len(w)) : start])
    out = []
    for step in range(steps):
        force = sum(w[lag] * hist[-(lag + 1)] for lag in range(min(len(w), len(hist))))
        rt = np.nan_to_num(r[start + step], nan=1.0)
        nxt = rt * force
        out.append(nxt)
        hist.append(nxt)
    return np.stack(out, axis=0)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cases, _, _names = base.load_dataset(NPY, ADJ)
    # The values at which the paper's Table 1 sensitivity indices actually
    # reproduce -- not its "Baseline Values" column, which disagrees with itself.
    p = seir.PAPER_SECTION4
    w = generation_interval_weeks(p)

    print("SEIR-SEI generation interval, weekly")
    for lag, weight in enumerate(w, start=1):
        print(f"  lag {lag} wk  {weight:.4f}  {'#' * round(weight * 50)}")
    mean_gi = float((np.arange(1, len(w) + 1) * w).sum())
    print(f"  mean {mean_gi:.2f} weeks   R0 = {seir.r0(p):.3f}\n")

    r = implied_r(cases, w)
    finite = r[np.isfinite(r)]
    print(f"implied R_t over {finite.size} district-weeks "
          f"({finite.size / r.size:.0%} of the record had enough history)")
    for q in (1, 5, 25, 50, 75, 95, 99):
        print(f"  p{q:<3d} {np.percentile(finite, q):6.3f}")
    print(f"  share above 1.0: {(finite > 1).mean():.1%}   "
          f"above 3.0: {(finite > 3).mean():.2%}\n")

    # Reconstruction: 3-week replays from every week with enough history, which is
    # the horizon the protocol forecasts.
    errs, base_errs = [], []
    for start in range(len(w) + 1, cases.shape[0] - 3):
        truth = cases[start : start + 3]
        recon = replay(cases, r, w, start, 3)
        errs.append((recon - truth) ** 2)
        base_errs.append((np.repeat(cases[start - 1][None, :], 3, axis=0) - truth) ** 2)
    rmse = float(np.sqrt(np.mean(errs)))
    persist = float(np.sqrt(np.mean(base_errs)))
    print(f"3-week replay with oracle R_t : RMSE {rmse:.2f}")
    print(f"persistence on the same windows: RMSE {persist:.2f}")
    print(f"  -> the decoder's ceiling is {100 * (1 - rmse / persist):.1f}% below persistence\n")

    # How much of the observed growth range can the decoder even express?
    growth = np.diff(np.log1p(np.maximum(cases, 0)), axis=0)
    print(f"observed |weekly log growth| max {np.abs(growth).max():.3f}, "
          f"p99 {np.percentile(np.abs(growth), 99):.3f}")
    print("A renewal decoder with R_t in [0, 10] can express log growth up to "
          f"{np.log(10 * w[0] + 1e-9) - np.log(w[0]):.3f} in one step from a "
          "steady history -- it is not structurally flat, which is the point.")

    report = {
        "generation_interval_weeks": w.tolist(),
        "mean_gi_weeks": mean_gi,
        "r0": float(seir.r0(p)),
        "implied_r_percentiles": {str(q): float(np.percentile(finite, q))
                                  for q in (1, 5, 25, 50, 75, 95, 99)},
        "coverage": float(finite.size / r.size),
        "oracle_replay_rmse": rmse,
        "persistence_rmse_same_windows": persist,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
