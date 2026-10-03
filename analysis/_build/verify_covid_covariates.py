"""Independent verification of the COVID covariate experiment (EXP-038).

The experiment in `run_covid_covariates.py` is sound and its headline numbers
reproduce exactly. This script re-derives them from the saved rows rather than
trusting `report.md`, and adds the two things that reading the report does not
tell you.

**1. The noise floor.** `CovidModel` forces the encoder to `in_dim=1` and starts
the exposure weights at exactly zero, so where a covariate has no coverage the
three arms are the *same model*, not merely similar ones. Origins 0.50 and 0.55
have zero coverage in train, validation and test alike. Any difference between
arms there is therefore pure numerical noise, and it is large: up to 0.109 RMSE,
because the early-stopping rule `v < best - 1e-5` can select a different epoch
once a 1e-6 wobble appears. The protocol doc's stated tolerance of 0.0001 was
taken from origin 0.50 alone and is ~1000x too small.

This sharpens the verdict rather than weakening it. The two covariate effects are
+0.015 and +0.021 RMSE, roughly 5x *below* the floor: they are not just
non-significant at p = 0.9, they are smaller than the experiment can resolve.

**2. Why adding mobility is the worst arm.** The fitted weights explain it. At
origins 0.50-0.65 they are exactly 0.0000 -- a model cannot learn a lockdown
response from training data containing no lockdown, which is here measured rather
than argued. Where there is coverage, stringency earns a *negative* weight (the
epidemiologically correct sign) but workplace mobility earns a *positive* one.
Both exposures are positive during a lockdown, so the two terms very nearly
cancel: collinear covariates with opposing fitted signs contribute net nothing
and add variance.

Usage::

    python analysis/_build/verify_covid_covariates.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FINAL = REPO / "analysis" / "results" / "covid_covariates_final"
REPORTED = {"base": 30.048672958656592,
            "policy": 30.063734972918475,
            "policy_mobility": 30.069347134342900}


def perm_p(delta: np.ndarray) -> float:
    """Exact paired sign-flip p-value, clustered by origin."""
    flips = np.array(list(itertools.product([-1, 1], repeat=len(delta))))
    return float(np.mean(np.abs((flips * delta).mean(1)) >= abs(delta.mean())))


def main() -> int:
    rows = json.loads((FINAL / "runs.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    if len(df) != 81 or df.groupby(["arm", "origin", "seed"]).ngroups != 81:
        raise SystemExit(f"expected 81 unique rows, found {len(df)}")

    print(f"{len(df)} rows, 81 unique (arm, origin, seed) cells\n")
    print("=== arm means, re-derived against the reported values ===")
    means = df.groupby("arm")["test_RMSE"].mean()
    for arm, value in means.items():
        gap = abs(value - REPORTED[arm])
        print(f"  {arm:16s} {value:.12f}  reported {REPORTED[arm]:.12f}  "
              f"|diff| {gap:.2e}  {'OK' if gap < 1e-9 else 'MISMATCH'}")

    ref = df[df.arm == "base"].groupby("origin")["test_RMSE"].mean()
    print("\n=== paired against base, clustered by origin ===")
    for arm, g in df.groupby("arm"):
        if arm == "base":
            continue
        d = g.groupby("origin")["test_RMSE"].mean() - ref
        print(f"  {arm:16s} mean {d.mean():+.4f} | better {int((d < 0).sum())}/9 "
              f"| p = {perm_p(d.values):.5f}")

    print("\n=== the noise floor: arms that are the same model by construction ===")
    zero = df[(df.train_policy_windows == 0) & (df.test_policy_windows == 0)]
    piv = zero.pivot_table(index=["origin", "seed"], columns="arm", values="test_RMSE")
    piv["spread"] = piv.max(axis=1) - piv.min(axis=1)
    print(piv.to_string(float_format=lambda v: f"{v:.6f}"))
    floor = float(piv.spread.max())
    print(f"\n  measured floor            {floor:.4f} RMSE")
    print(f"  stated tolerance          0.0001 RMSE  ({floor / 1e-4:.0f}x too small)")
    for arm in ("policy", "policy_mobility"):
        effect = means[arm] - means["base"]
        print(f"  effect of {arm:16s} {effect:+.4f} RMSE  "
              f"= {abs(effect) / floor:.2f}x the floor")

    print("\n=== fitted exposure weights: why mobility makes it worse ===")
    w = (df[df.arm != "base"]
         .groupby(["arm", "origin"])[["train_policy_windows",
                                      "policy_weight", "mobility_weight"]]
         .mean().round(4))
    print(w.to_string())
    exposed = df[(df.arm == "policy_mobility") & (df.train_policy_windows > 0)]
    print(f"\n  where training has lockdown weeks: stringency "
          f"{exposed.policy_weight.mean():+.3f}, mobility "
          f"{exposed.mobility_weight.mean():+.3f}")
    print("  both exposures are positive under restriction, so the terms oppose "
          "and nearly cancel")
    unexposed = df[(df.arm != "base") & (df.train_policy_windows == 0)]
    assert (unexposed[["policy_weight", "mobility_weight"]].abs().to_numpy() == 0).all()
    print(f"  all {len(unexposed)} runs with no lockdown in training kept both "
          f"weights at exactly zero")
    return 0


if __name__ == "__main__":
    sys.exit(main())
