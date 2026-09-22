"""Which initial-state seeding gives the physics head a target it can reach.

``diagnose_foi.py`` establishes that the ``lambda = 0`` floor already overshoots
14-16% of targets: with transmission switched entirely off, maturing ``E`` alone
emits more cases than truth. The floor is set by ``E0``, so it is a property of
the seeding and not of the backbone -- A3TGCN inherits it unchanged.

This scores the three seedings in ``models.SEEDS`` on the two things that decide
whether a head can fit at all: how much of the target is inside the reachable
band, and how much of the inverted lambda is a function of what the network
sees. Neither involves training, so the answer does not depend on an optimiser.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import diagnose_foi as dg
import models


def score(origin: float, mode: str) -> dict:
    data, _fold, pack, cum, _ = dg.build(origin)
    pop, y = pack["pop"], pack["y_raw"]
    state = models.seir_state(pack["x_raw"], pack["pop"], cum, mode=mode)

    st_lo, floor = state, []
    for _ in range(dg.core.HORIZON):
        z = torch.zeros(st_lo.shape[:-1])
        floor.append(dg.incidence(st_lo, z) * (dg.RHO * pop))
        st_lo = dg.step(st_lo, z)
    lo = torch.stack(floor, -1)

    st, lam = state, []
    for h in range(dg.core.HORIZON):
        step_lam = dg.invert(st, pop, y[..., h])
        lam.append(step_lam)
        st = dg.step(st, step_lam)
    lam = torch.stack(lam, -1)

    last = np.log1p(pack["x_raw"][..., -1].numpy())
    lg = np.log(np.clip(lam[..., 0].numpy(), 1e-8, None))
    return {
        "below_floor": 100 * float((y < lo).float().mean()),
        "pinned": 100 * float((lam <= 1e-9).float().mean()),
        "over_max": 100 * float((lam > dg.LAMBDA_MAX).float().mean()),
        "r2": dg.r2(last, lg),
        "floor_med": float(lo.median()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=float, nargs="+", default=[0.55, 0.70, 0.85])
    args = ap.parse_args()

    print(f"{'seeding':9s} {'below floor':>12s} {'lambda pinned':>14s} {'over max':>9s} "
          f"{'median floor':>13s} {'r2(log lambda)':>15s}")
    print("-" * 76)
    for mode in models.SEEDS:
        rows = [score(o, mode) for o in args.origins]
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(f"{mode:9s} {agg['below_floor']:11.1f}% {agg['pinned']:13.1f}% "
              f"{agg['over_max']:8.1f}% {agg['floor_med']:13.1f} {agg['r2']:15.3f}")
    print("\nlower is better for the first four; higher for r2 (how learnable lambda is).")
    print("targets below the floor are unreachable at any network output.\n")


if __name__ == "__main__":
    main()
