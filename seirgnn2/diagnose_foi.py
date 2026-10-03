"""Why the ``foi`` head is last, measured rather than argued.

The head predicts a per-district force of infection and lets ``seir_sim`` turn
it into incidence. It lands at validation RMSE 26.9 against 16.8 for every other
head and 17.9 for persistence, so something in that path is not a modelling
choice but a defect. Four things could each do it, and they are separable:

1. **Reach.** For a given initial state the simulator's weekly incidence is
   monotone in lambda, so the head can only reach the band between its output at
   ``lambda = 0`` and at ``lambda = lambda_max``. A target outside that band is
   unreachable at any network output.
2. **The susceptible pool.** ``seir_state`` sets ``S = s0 - cum / (rho * pop)``
   with cumulative *reported* cases inflated by ``1 / rho = 11``. Incidence is
   proportional to S, so if this varies by an order of magnitude across
   districts it is a gain the network has to cancel -- and it never sees S.
3. **Learnability.** Invert the simulator for the lambda that reproduces each
   true count exactly. If that oracle lambda is not a function of what the
   network is given, no architecture recovers it.
4. **Saturation.** ``lambda = lambda_max * sigmoid(raw)``. If the oracle sits in
   a sliver at either end, the gradient there is flat.

Run::

    python seirgnn2/diagnose_foi.py            # all four, origin 0.70
    python seirgnn2/diagnose_foi.py --origin 0.85
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis" / "lib"))

import core
import corrected_data as cd
import models
import seir_sim

LAMBDA_MAX = 1.0 / 7.0
RHO = 1.0 / 11.0
OMEGA, GAMMA = 0.7 / 7.0, 1.0 / 7.0


def incidence(state: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Counts for one week from one constant force of infection, ``(...,)``."""
    _, inc = seir_sim.simulate_weeks(state, lam.unsqueeze(-1), omega=OMEGA, gamma=GAMMA,
                                     substeps=7)
    return inc[..., 0]


def step(state: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    st, _ = seir_sim.simulate_weeks(state, lam.unsqueeze(-1), omega=OMEGA, gamma=GAMMA, substeps=7)
    return st[..., 1, :]


def invert(state: torch.Tensor, pop: torch.Tensor, target: torch.Tensor,
           iters: int = 60) -> torch.Tensor:
    """Bisect for the lambda whose incidence matches ``target`` counts.

    Incidence is monotone increasing in lambda, so bisection is exact to
    machine precision in the reachable band and returns the nearer endpoint
    outside it. Searched well past ``lambda_max`` so the result also says *how
    far* past the bound a target sits.
    """
    lo = torch.zeros_like(target)
    hi = torch.full_like(target, 10.0 * LAMBDA_MAX)
    scale = RHO * pop
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        got = incidence(state, mid) * scale
        too_low = got < target
        lo = torch.where(too_low, mid, lo)
        hi = torch.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


def r2(x: np.ndarray, y: np.ndarray) -> float:
    """Variance of ``y`` explained by an OLS fit on ``x`` (with intercept)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    a = np.stack([x, np.ones_like(x)], 1)
    resid = y - a @ np.linalg.lstsq(a, y, rcond=None)[0]
    return float(1.0 - resid.var() / y.var())


def build(origin: float):
    data = cd.load()
    fold = next(f for f in core.build_folds(data.cases, data.missing) if f.origin == origin)
    pack = core.build_tensors(data, fold, "train", False, False, False)
    cases = np.nan_to_num(data.cases)
    cum = torch.tensor(np.stack([cases[:i].sum(0) for i in pack["idx"]]), dtype=torch.float32)
    state = models.seir_state(pack["x_raw"], pack["pop"], cum)
    return data, fold, pack, cum, state


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", type=float, default=0.70)
    args = ap.parse_args()

    data, _fold, pack, cum, state = build(args.origin)
    pop, y = pack["pop"], pack["y_raw"]
    s = state[..., 0]
    print(f"origin {args.origin}  |  {len(pack['idx'])} training windows x {y.shape[1]} districts")

    # ---- 2. the susceptible pool ------------------------------------------
    section("2. susceptible pool -- the gain the network never sees")
    depl = (cum / (RHO * pop)).numpy()
    print(f"  s0 (1 - seroprevalence)                : {1 - 0.682:.3f}")
    print(f"  cum/(rho*pop): median {np.median(depl):.3f}   p95 {np.percentile(depl, 95):.3f}"
          f"   max {depl.max():.3f}")
    print(f"  implied depletion exceeds s0 in         : {100 * (depl > 1 - 0.682).mean():.1f}% of cells")
    print(f"  S pinned at the 0.01 clamp floor        : {100 * (s <= 0.0100001).float().mean():.1f}% of cells")
    print(f"  S across cells: min {s.min():.4f}  median {s.median():.4f}  max {s.max():.4f}"
          f"   -> {s.max() / s.min():.0f}x spread")
    per_district = s.mean(0)
    worst = torch.argsort(per_district)[:4]
    best = torch.argsort(per_district)[-3:]
    names = data.names
    print("  lowest-S districts : " + ", ".join(f"{names[k]} {per_district[k]:.3f}" for k in worst))
    print("  highest-S districts: " + ", ".join(f"{names[k]} {per_district[k]:.3f}" for k in best))
    drift = np.polyfit(np.arange(len(s)), s.mean(1).numpy(), 1)[0] * len(s)
    print(f"  mean S drifts by {drift:+.3f} across the training span "
          f"(a systematic gain change the net cannot observe)")

    # ---- 1. reach ----------------------------------------------------------
    section("1. reach -- is the true count even attainable")
    lo_counts, hi_counts, st = [], [], state
    st_lo, st_hi = state, state
    for _ in range(core.HORIZON):
        z = torch.zeros(st_lo.shape[:-1])
        lo_counts.append(incidence(st_lo, z) * (RHO * pop))
        st_lo = step(st_lo, z)
        m = torch.full(st_hi.shape[:-1], LAMBDA_MAX)
        hi_counts.append(incidence(st_hi, m) * (RHO * pop))
        st_hi = step(st_hi, m)
    lo = torch.stack(lo_counts, -1)
    hi = torch.stack(hi_counts, -1)
    below = (y < lo).float().mean().item()
    above = (y > hi).float().mean().item()
    print(f"  target below the lambda=0 floor : {100 * below:5.1f}%   "
          f"(the E compartment alone already overshoots)")
    print(f"  target above the lambda_max ceil: {100 * above:5.1f}%")
    print(f"  reachable                       : {100 * (1 - below - above):5.1f}%")
    print(f"  median floor / median target    : {lo.median():.1f} / {y.median():.1f}")
    print("  floor is driven by E0 = cases[t-2]/rho, half of which matures within the week:")
    print(f"    corr(lambda=0 week-1 count, cases[t-2]) = "
          f"{np.corrcoef(lo[..., 0].flatten(), pack['x_raw'][..., -2].flatten())[0, 1]:.3f}")

    # ---- 3. learnability ---------------------------------------------------
    section("3. learnability -- is the oracle lambda a function of the inputs")
    st, lam_star = state, []
    for h in range(core.HORIZON):
        lam = invert(st, pop, y[..., h])
        lam_star.append(lam)
        st = step(st, lam)
    lam_star = torch.stack(lam_star, -1)
    frac = (lam_star / LAMBDA_MAX).numpy()
    print(f"  oracle lambda / lambda_max: p05 {np.percentile(frac, 5):.4f}  "
          f"median {np.median(frac):.4f}  p95 {np.percentile(frac, 95):.4f}")
    print(f"  pinned at 0 (target unreachable from below): {100 * (frac <= 1e-6).mean():.1f}%")
    print(f"  beyond lambda_max                          : {100 * (frac > 1.0).mean():.1f}%")

    last = pack["x_raw"][..., -1].numpy()
    lg = np.log(np.clip(lam_star[..., 0].numpy(), 1e-8, None))
    print("\n  variance of log(oracle lambda, week 1) explained by:")
    print(f"    log cases[t-1]                    r2 = {r2(np.log1p(last), lg):6.3f}")
    print(f"    log S (which the net never sees)  r2 = {r2(np.log(s.numpy()), lg):6.3f}")
    both = np.stack([np.log1p(last).ravel(), np.log(s.numpy()).ravel(), np.ones(last.size)], 1)
    tgt = lg.ravel()
    res = tgt - both @ np.linalg.lstsq(both, tgt, rcond=None)[0]
    print(f"    both together                     r2 = {1 - res.var() / tgt.var():6.3f}")
    print("\n  for contrast, the target the other heads regress on:")
    print(f"    log cases[t] on log cases[t-1]    r2 = "
          f"{r2(np.log1p(last), np.log1p(y[..., 0].numpy())):6.3f}")

    # ---- counterfactual: no spurious depletion -----------------------------
    section("4. counterfactual -- the same inversion with S held at s0")
    flat = state.clone()
    flat[..., 0] = 1.0 - 0.682
    flat[..., 3] = (1.0 - flat[..., 0] - flat[..., 1] - flat[..., 2]).clamp_min(0.0)
    st, lam_flat = flat, []
    for h in range(core.HORIZON):
        lam = invert(st, pop, y[..., h])
        lam_flat.append(lam)
        st = step(st, lam)
    lam_flat = torch.stack(lam_flat, -1)
    lgf = np.log(np.clip(lam_flat[..., 0].numpy(), 1e-8, None))
    print(f"  oracle lambda / lambda_max: median {np.median(lam_flat.numpy()) / LAMBDA_MAX:.4f}"
          f"   beyond bound {100 * (lam_flat.numpy() > LAMBDA_MAX).mean():.1f}%")
    print(f"  log(oracle lambda) explained by log cases[t-1]: r2 = {r2(np.log1p(last), lgf):6.3f}"
          f"   (was {r2(np.log1p(last), lg):.3f})")
    print(f"  spread of log oracle lambda: sd {lgf.std():.3f}  (was {lg.std():.3f})")

    # ---- saturation --------------------------------------------------------
    section("5. saturation -- where sigmoid has to sit")
    q = np.clip(frac[..., 0], 1e-9, 1 - 1e-9)
    raw = np.log(q / (1 - q))
    print(f"  required pre-sigmoid output: p05 {np.percentile(raw, 5):7.2f}  "
          f"median {np.median(raw):7.2f}  p95 {np.percentile(raw, 95):7.2f}")
    grad = q * (1 - q)
    print(f"  d(sigmoid)/d(raw) there    : median {np.median(grad):.4f}  "
          f"p05 {np.percentile(grad, 5):.6f}   (0.25 is the maximum)")
    print(f"  cells needing |raw| > 6    : {100 * (np.abs(raw) > 6).mean():.1f}%")
    print(f"  sigmoid at init (raw~0) gives lambda = {LAMBDA_MAX * 0.5:.4f}/day, "
          f"{LAMBDA_MAX * 0.5 / np.median(lam_star.numpy()):.0f}x the median oracle")

    # ---- 6. the ceiling ----------------------------------------------------
    section("6. ceiling -- best RMSE the head could reach with a perfect lambda")

    def replay(lam: torch.Tensor) -> np.ndarray:
        st, out = state, []
        for h in range(core.HORIZON):
            out.append(incidence(st, lam[..., h]) * (RHO * pop))
            st = step(st, lam[..., h])
        return torch.stack(out, -1).clamp_min(0.0).numpy()

    truth = y.numpy()
    persistence = pack["p_raw"].numpy()
    print(f"  persistence on these windows          RMSE {core.rmse(persistence, truth):7.2f}")
    print(f"  oracle lambda (unreachable cells clip) RMSE {core.rmse(replay(lam_star), truth):7.2f}")

    # lambda from the best linear rule on what the network actually sees.
    feats = np.stack([np.log1p(last).ravel(), np.ones(last.size)], 1)
    lam_hat = []
    st_lam = lam_star.numpy()
    for h in range(core.HORIZON):
        tgt = np.log(np.clip(st_lam[..., h], 1e-8, None)).ravel()
        fit = feats @ np.linalg.lstsq(feats, tgt, rcond=None)[0]
        lam_hat.append(torch.tensor(np.exp(fit).reshape(last.shape), dtype=torch.float32))
    print(f"  lambda = best OLS on log cases[t-1]   RMSE {core.rmse(replay(torch.stack(lam_hat, -1)), truth):7.2f}")
    print("    ^ fitted in log-lambda space, so exponentiating blows up the tail;")
    print("      the fair version fits the same rule against the metric itself:")

    def fit_direct(through_physics: bool) -> float:
        """Best ``exp(a * log1p(cases[t-1]) + b)`` per horizon, fitted on count MSE.

        With ``through_physics`` the quantity produced is lambda and the counts
        come out of the simulator; without it the same rule predicts counts
        directly. Same inputs, same capacity, same objective -- the only
        difference is whether the prediction is routed through the SEIR layer.
        """
        x = torch.log1p(pack["x_raw"][..., -1])
        a = torch.zeros(core.HORIZON, requires_grad=True)
        b = torch.full((core.HORIZON,), -6.0 if through_physics else 1.0, requires_grad=True)
        opt = torch.optim.Adam([a, b], lr=0.05)
        for _ in range(600):
            opt.zero_grad()
            q = torch.exp((a * x.unsqueeze(-1) + b).clamp(-20, 12))
            if through_physics:
                st_, preds = state, []
                for h in range(core.HORIZON):
                    preds.append(incidence(st_, q[..., h]) * (RHO * pop))
                    st_ = step(st_, q[..., h])
                pred = torch.stack(preds, -1)
            else:
                pred = q
            mse = ((pred - y) ** 2).mean()
            mse.backward()
            opt.step()
        return float(torch.sqrt(mse.detach()))

    print(f"  same rule, fitted on count MSE, through SEIR   RMSE {fit_direct(True):7.2f}")
    print(f"  same rule, fitted on count MSE, predicting directly RMSE {fit_direct(False):7.2f}")
    print("\n  the gap between those last two is the cost of the physics layer alone:")
    print("   same inputs, same capacity, same objective, one routed through SEIR.\n")


if __name__ == "__main__":
    main()
