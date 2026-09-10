"""Beat the persistence floor: statistical power, retransformation bias, combination.

Why this file exists
--------------------
``improved_sweep`` reports A3TGCN at 29.05 against a floor of 29.52 -- nominally
ahead, ``p=0.10``. That p-value is **pseudoreplicated**. Seed variance on this
task is negligible (sd 0.02--0.05 within an origin) while origin variance is
enormous (22.07 / 37.74 / 27.35), so nine (origin, seed) rows carry three
independent observations, not nine. Clustered by origin the same result is
``-0.467, p=0.436``. No number of extra seeds fixes that; only extra origins do.

Three levers are applied here, each of which is a standard estimator correction
rather than a new architecture. None of them changes what the network learns:
the network is trained exactly as in ``run_physics_experiments``, and every
correction below is calibrated on the **validation split** and applied to test.

``origins``
    Disjoint rolling-origin blocks instead of three overlapping ones. Test
    fraction defaults to the origin spacing so consecutive test sets do not
    share windows -- overlapping blocks would re-introduce the same
    pseudoreplication in a new form.

``smear`` / ``lognorm`` -- retransformation bias
    The model is trained on ``log1p`` and scored on counts. For
    ``L = log(1+y) = m + e``, ``expm1(m)`` is the conditional **median**, while
    RMSE is minimised by the conditional **mean**
    ``E[y] = exp(m) E[exp(e)] - 1``. Reporting ``expm1(m)`` therefore
    under-predicts by construction, by an amount that grows with residual
    variance. That is exactly the measured signature: ``error_diagnosis.json``
    puts A3TGCN's bias at **-12.92 on outbreak windows and +0.38 on quiet
    ones**. Two corrections for it:

    * ``smear`` -- Duan's smearing estimator, ``S = mean(exp(e))`` on validation
      residuals. Nonparametric; assumes nothing about the error distribution.
    * ``lognorm`` -- the Gaussian special case ``S = exp(v/2)``. With a
      ``probabilistic`` head ``v`` is predicted per window, so the correction is
      heteroscedastic and largest exactly where the bias is measured to be
      largest. Without one, ``v`` is the pooled validation residual variance.

``blend`` -- combination with the floor
    ``w * model + (1 - w) * persistence``, with ``w`` fit per horizon by least
    squares on validation. The forecast-combination literature is unusually
    unanimous that combining a model with a strong naive baseline reduces error;
    here persistence has lag-1 ``r^2 = 0.85``, so it is a strong baseline by
    construction. ``w`` is clipped to ``[0, 1]``: the arm may fall back to the
    floor but may not extrapolate past the model.

``ens`` -- seed ensembling
    Mean of the seeds' count-space predictions. Pure variance reduction, and the
    one lever here that needs no calibration at all.

``--head nb`` -- remove the bias instead of correcting it
    A negative-binomial head on raw counts, ``Var = mu + alpha mu^2``, which
    emits the conditional **mean** directly. Where the corrections above repair
    a log-space median after the fact, this never creates the problem: there is
    no retransformation step. NB2 rather than Poisson because this target is
    heavily overdispersed (median 13, max 2631, 9.7% zeros), and it is the
    specification the dengue count-forecasting literature settles on. Fitting
    ``calib`` on top of it is then a diagnostic -- a factor near 1.0 says the
    likelihood really did remove the bias.

Every arm is scored against persistence on the identical windows, with the
week-395 artifact split out the same way for both.

Run::

    python analysis/_build/run_beat_baseline.py --quick
    python analysis/_build/run_beat_baseline.py --arch A3TGCN --n-origins 9 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "src"))

import adaptive as base  # noqa: E402
import count_loss as cl  # noqa: E402
import improved as imp  # noqa: E402
import reproduced as arch  # noqa: E402

NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
OUT_DIR = REPO / "analysis" / "results"

WINDOW, HORIZON = 3, 3
LOG_CLIP = 12.0

#: Output heads.
#:
#: ``det``   -- Huber on log1p, the existing pipeline.
#: ``gauss`` -- Gaussian NLL on log1p; predicts a per-window variance, which is
#:              what makes the ``lognorm`` correction heteroscedastic.
#: ``nb``    -- negative binomial on raw counts. Predicts the count **mean**
#:              directly, so there is no retransformation step to be biased by.
#:              Its ``calib`` factor is then a diagnostic rather than a fix: a
#:              well-specified NB head should need a factor near 1.0.
HEADS = ("det", "gauss", "nb")

#: Point estimators, all calibrated on validation and applied to test.
ESTIMATORS = ("raw", "smear", "lognorm", "calib")

#: Bounds on the multiplicative retransformation factor.
#:
#: The correction is a ratio on ``1 + y``, and it is bounded on both sides for a
#: reason the smoke test found before the real run did: ``exp(v / 2)`` is
#: unbounded in the residual variance, so a poorly fit fold inflates the forecast
#: without limit and turns a bias correction into a catastrophe. The lower bound
#: is below 1 only for ``calib``, which is fit against RMSE directly and is
#: allowed to shrink; the theory-driven estimators correct a known *downward*
#: bias and may not push a forecast down.
FACTOR_MAX = 3.0
FACTOR_MIN_THEORY = 1.0
FACTOR_MIN_CALIB = 0.5


def make_origins(n: int, lo: float = 0.50, hi: float = 0.90) -> tuple[tuple[float, ...], float]:
    """``n`` evenly spaced origins and the test fraction that makes them disjoint.

    Returns ``(origins, test_frac)``. With ``test_frac`` equal to the spacing,
    consecutive test blocks tile the tail of the series without overlapping, so
    each fold is an independent observation. A single origin falls back to the
    protocol's 0.15.
    """
    if n == 1:
        return (lo,), 0.15
    origins = tuple(round(float(x), 6) for x in np.linspace(lo, hi, n))
    return origins, round((hi - lo) / (n - 1), 6)


def train_one(
    arch_name: str,
    fold,
    edge_index: torch.Tensor,
    seed: int,
    head: str,
    epochs: int,
    patience: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
) -> nn.Module:
    """Train one backbone on one fold. Identical protocol to the physics sweep.

    Model selection is on raw validation RMSE -- the same criterion the existing
    sweep uses -- so the trained weights are unchanged by anything in this file
    and every estimator below is strictly post-hoc.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    # Both `gauss` and `nb` need a two-channel head; they differ in what the
    # second channel means -- a log-variance in log1p space, or an NB2 dispersion.
    inc = imp.Increments(probabilistic=head in ("gauss", "nb"))
    kwargs = {"adaptive": False, "channels": 8} if arch_name == "AAGCN" else {}
    net = arch.build(arch_name, 25, WINDOW, HORIZON, edge_index=edge_index, inc=inc, **kwargs)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    n = len(fold.x_train)
    best_weights, best_val, waited = None, float("inf"), 0

    for _epoch in range(epochs):
        net.train()
        order = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            x, p, y = fold.x_train[idx], fold.p_train[idx], fold.y_train[idx]
            opt.zero_grad()
            out = net(x, edge_index)
            if head == "gauss":
                loss = imp.gaussian_nll(out[..., 0] + p, out[..., 1], y)
            elif head == "nb":
                mu, alpha = cl.split_nb(out, p * fold.std + fold.mean)
                counts_true = torch.expm1((y * fold.std + fold.mean).clamp(0.0, LOG_CLIP))
                loss = cl.nb_nll(mu, alpha, counts_true)
            else:
                loss = nn.functional.smooth_l1_loss(out + p, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

        net.eval()
        with torch.no_grad():
            m, _ = _head_output(net, fold, "val", edge_index, head)
            val_rmse = base.rmse(np.expm1(m), fold.inverse(fold.y_val.numpy()))

        if val_rmse < best_val - 1e-6:
            best_val, waited = val_rmse, 0
            best_weights = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    if best_weights is not None:
        net.load_state_dict(best_weights)
    return net


def _head_output(net, fold, split: str, edge_index, head: str):
    """Return ``(m, v)`` in log1p space: mean and variance of ``log(1 + y)``.

    Every head is reported through the same pair so one estimator pipeline
    serves all three. ``m`` is un-z-scored and clipped exactly as
    :meth:`Fold.inverse` clips, so for ``det`` the ``raw`` estimator reproduces
    the existing pipeline's point prediction bit for bit.

    The ``nb`` head already emits a count mean, so it is reported as
    ``m = log1p(mu)``: ``raw`` then returns ``mu`` unchanged, and the ``calib``
    factor fitted on top becomes a *test* of the head rather than a repair of it.
    A factor near 1.0 says the count likelihood removed the bias; a factor well
    above 1.0 says it did not.
    """
    with torch.no_grad():
        out = net(getattr(fold, f"x_{split}"), edge_index)
        p = getattr(fold, f"p_{split}")
        if head == "nb":
            mu, _ = cl.split_nb(out, p * fold.std + fold.mean)
            return np.log1p(mu.numpy()), np.zeros(mu.shape, dtype=np.float64)
        z = (out[..., 0] if head == "gauss" else out) + p
        m = np.clip(z.numpy() * fold.std + fold.mean, 0.0, LOG_CLIP)
        if head == "gauss":
            v = np.exp(out[..., 1].clamp(-10.0, 10.0).numpy()) * (fold.std**2)
        else:
            v = np.zeros_like(m)
    return m, v


def predict(net, fold, split: str, edge_index, head: str):
    """``_head_output`` with the module in eval mode."""
    net.eval()
    return _head_output(net, fold, split, edge_index, head)


def fit_factors(m_val, v_val, truth_val) -> dict[str, np.ndarray]:
    """Per-horizon retransformation factors, fit on validation only.

    Every estimator ends as a multiplicative factor on ``1 + y``, differing only
    in how it is derived. Fitting per horizon rather than globally matters
    because residual variance grows with the step -- horizon RMSE runs
    23.8 / 29.2 / 35.7 -- so one global factor is under-correcting at ``h=3``
    while over-correcting at ``h=1``.

    Returns a dict of arrays broadcastable against ``m``.
    """
    horizon = m_val.shape[-1]
    log_truth = np.log1p(np.clip(truth_val, 0.0, None))
    resid = log_truth - m_val
    a_val = np.exp(m_val)

    smear = np.empty(horizon)
    pooled_v = np.empty(horizon)
    calib = np.empty(horizon)
    for h in range(horizon):
        e = resid[..., h].ravel()
        smear[h] = float(np.mean(np.exp(e)))
        pooled_v[h] = float(np.var(e))
        # RMSE-optimal single factor: argmin_c || c*a - (1+y) ||^2.
        a, yp = a_val[..., h].ravel(), 1.0 + np.clip(truth_val[..., h].ravel(), 0.0, None)
        denom = float(a @ a)
        calib[h] = 1.0 if denom < 1e-12 else float(a @ yp / denom)

    clip_t = lambda z: np.clip(z, FACTOR_MIN_THEORY, FACTOR_MAX)  # noqa: E731
    return {
        "raw": np.ones(horizon),
        "smear": clip_t(smear),
        # Heteroscedastic when a Gaussian head supplies a per-window variance,
        # pooled otherwise. This is where the probabilistic head earns its keep:
        # the correction is then largest exactly on the high-variance outbreak
        # windows where the measured bias is -12.9 rather than +0.4.
        "lognorm": clip_t(np.exp(0.5 * (v_val if v_val.any() else pooled_v))
                          if v_val.any() else np.exp(0.5 * pooled_v)),
        "calib": np.clip(calib, FACTOR_MIN_CALIB, FACTOR_MAX),
    }


def counts_from(m: np.ndarray, v: np.ndarray, estimator: str, factors: dict) -> np.ndarray:
    """Point prediction on the count scale under one retransformation estimator."""
    if estimator == "raw":
        return np.expm1(m)
    if estimator not in factors:
        raise ValueError(f"unknown estimator {estimator!r}")
    if estimator == "lognorm" and v.any():
        factor = np.clip(np.exp(0.5 * v), FACTOR_MIN_THEORY, FACTOR_MAX)
    else:
        factor = factors[estimator]
    return np.clip(np.exp(m) * factor - 1.0, 0.0, None)


def blend_weights(model_val: np.ndarray, pers_val: np.ndarray, truth_val: np.ndarray):
    """Per-horizon least-squares weight on the model, clipped to ``[0, 1]``.

    Minimises ``|| w a + (1 - w) b - y ||^2`` in closed form, per horizon step,
    on validation only. Clipping keeps the arm between the two forecasts it
    combines rather than extrapolating beyond either.
    """
    w = np.zeros(model_val.shape[-1])
    for h in range(model_val.shape[-1]):
        a, b, y = model_val[..., h].ravel(), pers_val[..., h].ravel(), truth_val[..., h].ravel()
        d = a - b
        denom = float(d @ d)
        w[h] = 1.0 if denom < 1e-12 else float((y - b) @ d / denom)
    return np.clip(w, 0.0, 1.0)


def persistence_counts(cases: np.ndarray, index: np.ndarray, horizon: int) -> np.ndarray:
    """Last-value-carried-forward on the given windows."""
    return np.stack([np.repeat(cases[i - 1][:, None], horizon, axis=1) for i in index])


def score_arm(pred, truth, artifact, **tags) -> dict:
    """Pooled scores with the artifact split, tagged for the results table."""
    return dict(**tags, **base.pooled_scores(pred, truth, artifact))


def run_fold(arch_name, fold, cases, edge_index, seeds, head, epochs, verbose=True):
    """Train every seed on one fold and score all estimator/combination arms."""
    truth_val = fold.inverse(fold.y_val.numpy())
    truth_test = fold.inverse(fold.y_test.numpy())
    pers_val = persistence_counts(cases, fold.val_index, HORIZON)
    pers_test = persistence_counts(cases, fold.test_index, HORIZON)
    artifact = imp.artifact_windows(fold.test_index, WINDOW, HORIZON)

    records: list[dict] = []
    per_seed_test: dict[str, list[np.ndarray]] = {e: [] for e in ESTIMATORS}
    per_seed_val: dict[str, list[np.ndarray]] = {e: [] for e in ESTIMATORS}

    for seed in seeds:
        t0 = time.time()
        net = train_one(arch_name, fold, edge_index, seed, head, epochs)
        m_val, v_val = predict(net, fold, "val", edge_index, head)
        m_test, v_test = predict(net, fold, "test", edge_index, head)

        # Every correction is calibrated on validation residuals only.
        factors = fit_factors(m_val, v_val, truth_val)

        for est in ESTIMATORS:
            pv = counts_from(m_val, v_val, est, factors)
            pt = counts_from(m_test, v_test, est, factors)
            per_seed_val[est].append(pv)
            per_seed_test[est].append(pt)
            records.append(score_arm(pt, truth_test, artifact, arch=arch_name,
                                     origin=fold.origin, seed=seed, arm=est, ens=1))
            w = blend_weights(pv, pers_val, truth_val)
            records.append(score_arm(w * pt + (1.0 - w) * pers_test, truth_test, artifact,
                                     arch=arch_name, origin=fold.origin, seed=seed,
                                     arm=f"blend_{est}", ens=1, w=[round(float(x), 4) for x in w]))
        if verbose:
            fac = {k: np.round(np.mean(v_), 3) for k, v_ in factors.items() if k != "raw"}
            print(f"      seed {seed}: factors={fac} ({time.time() - t0:.0f}s)", flush=True)

    # Seed ensembles: mean of the seeds' count-space predictions.
    for est in ESTIMATORS:
        pv = np.mean(per_seed_val[est], axis=0)
        pt = np.mean(per_seed_test[est], axis=0)
        records.append(score_arm(pt, truth_test, artifact, arch=arch_name, origin=fold.origin,
                                 seed=-1, arm=f"ens_{est}", ens=len(seeds)))
        w = blend_weights(pv, pers_val, truth_val)
        records.append(score_arm(w * pt + (1.0 - w) * pers_test, truth_test, artifact,
                                 arch=arch_name, origin=fold.origin, seed=-1,
                                 arm=f"ens_blend_{est}", ens=len(seeds),
                                 w=[round(float(x), 4) for x in w]))

    # The floor, on exactly these windows and split the same way.
    records.append(score_arm(pers_test, truth_test, artifact, arch="persistence",
                             origin=fold.origin, seed=-1, arm="floor", ens=1))
    return records, {e: np.mean(per_seed_test[e], axis=0) for e in ESTIMATORS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", nargs="*", default=["A3TGCN"])
    ap.add_argument("--n-origins", type=int, default=9)
    ap.add_argument("--origin-lo", type=float, default=0.50)
    ap.add_argument("--origin-hi", type=float, default=0.90)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--head", choices=HEADS, default="det",
                    help="det: Huber on log1p. gauss: Gaussian NLL, enabling the "
                         "heteroscedastic lognorm correction. nb: negative binomial "
                         "on counts, which predicts the mean directly.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--tag", type=str, default="", help="Suffix for the output filename")
    args = ap.parse_args()

    epochs = 25 if args.quick else args.epochs
    seeds = tuple(range(1 if args.quick else args.seeds))
    n_origins = 2 if args.quick else args.n_origins

    origins, test_frac = make_origins(n_origins, args.origin_lo, args.origin_hi)
    cases, adjacency, _ = base.load_dataset(NPY, ADJ)
    src, dst = np.nonzero(adjacency)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    folds = base.build_folds(cases, WINDOW, HORIZON, origins=origins, test_frac=test_frac)

    print(f"origins={origins} test_frac={test_frac} -> {len(folds)} folds, "
          f"seeds={seeds}, epochs={epochs}, head={args.head}", flush=True)
    sizes = [len(f.test_index) for f in folds]
    print(f"test windows per fold: {sizes} (disjoint: "
          f"{len({i for f in folds for i in f.test_index.tolist()}) == sum(sizes)})", flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR / "beat_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    ens_by_arch: dict[str, dict] = {}
    for arch_name in args.arch:
        print(f"\n{'=' * 24} {arch_name} {'=' * 24}", flush=True)
        ens_by_arch[arch_name] = {}
        for fold in folds:
            print(f"  origin {fold.origin} (test n={len(fold.test_index)})", flush=True)
            recs, ens = run_fold(arch_name, fold, cases, edge_index, seeds,
                                 args.head, epochs)
            for r in recs:
                r["head"] = args.head
            all_records.extend(r for r in recs if r["arch"] != "persistence"
                               or arch_name == args.arch[0])
            ens_by_arch[arch_name][fold.origin] = ens

    # Cross-architecture ensemble: the combination literature's strongest single
    # recommendation is to average diverse models, so it is measured whenever
    # more than one architecture is present in the same run.
    if len(args.arch) > 1:
        for fold in folds:
            truth = fold.inverse(fold.y_test.numpy())
            artifact = imp.artifact_windows(fold.test_index, WINDOW, HORIZON)
            for est in ESTIMATORS:
                stack = [ens_by_arch[a][fold.origin][est] for a in args.arch]
                all_records.append(score_arm(np.mean(stack, axis=0), truth, artifact,
                                             arch="+".join(args.arch), origin=fold.origin,
                                             seed=-1, arm=f"multi_{est}", ens=len(stack)))

    tag = args.tag or "_".join(args.arch)
    suffix = "" if args.head == "det" else f"_{args.head}"
    out = out_dir / f"beat_{tag}{suffix}.json"
    out.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    print(f"\nwrote {len(all_records)} records -> {out}", flush=True)

    summarise(all_records)
    return 0


def summarise(records: list[dict]) -> None:
    """Print each arm against the floor, clustered by origin."""
    from scipy import stats

    floor = {r["origin"]: r["RMSE_clean"] for r in records if r["arch"] == "persistence"}
    if not floor:
        return
    print(f"\nfloor by origin: "
          f"{ {k: round(v, 2) for k, v in sorted(floor.items())} }")
    print(f"\n{'arch':22s}{'arm':20s}{'RMSE':>9s}{'vs floor':>10s}{'wins':>8s}{'p':>9s}")
    print("-" * 78)
    rows = [r for r in records if r["arch"] != "persistence"]
    for arch_name in dict.fromkeys(r["arch"] for r in rows):
        for arm in dict.fromkeys(r["arm"] for r in rows if r["arch"] == arch_name):
            sub = [r for r in rows if r["arch"] == arch_name and r["arm"] == arm]
            # Cluster by origin: average seeds within an origin first, so the
            # test has one observation per independent fold.
            by_origin: dict[float, list[float]] = {}
            for r in sub:
                by_origin.setdefault(r["origin"], []).append(r["RMSE_clean"])
            o = sorted(by_origin)
            v = np.array([np.mean(by_origin[k]) for k in o])
            f = np.array([floor[k] for k in o])
            d = v - f
            p = stats.ttest_rel(v, f)[1] if len(v) > 1 else float("nan")
            print(f"{arch_name:22s}{arm:20s}{v.mean():9.3f}{d.mean():+10.3f}"
                  f"{int((d < 0).sum()):5d}/{len(d):<3d}{p:9.4f}")


if __name__ == "__main__":
    sys.exit(main())
