"""What caps the architectures: measured on their forecasts, not their scores.

Every grid so far kept one RMSE per run, which says *how much* a model is wrong
and nothing about *how*. This retrains the five architectures that remain in
play -- STGAT, A3TGCN, ASTGCN, AAGCN and Liu et al.'s LSTM encoder, DCRNN dropped
for cost and rank -- keeps every forecast, and asks eight questions of them.

**Everything is measured on validation windows.** Test is sealed: studying test
errors and then designing a model around them spends the held-out set. The
frozen three origins give ~90 validation weeks x 25 districts x 3 horizons.

Two configurations, so a weakness can be told apart from a missing lever:
``native`` is each architecture as published (direct head, squared error on
log1p, no covariates); ``best`` adds the two levers that held up (negative-binomial
likelihood, seasonal features). SEIR-LSTM appears as ``LSTM+foi`` natively and
``LSTM+foi_res`` in the best configuration.

    python seirgnn2/diagnose_arch.py collect --workers 6   # trains, ~20-30 min
    python seirgnn2/diagnose_arch.py analyse               # seconds
"""

from __future__ import annotations

import argparse
import itertools
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import core  # noqa: E402
import corrected_data as cd  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "arch_preds.pkl"
SEEDS = (0, 1, 2)
ARCHS = ("STGAT", "A3TGCN", "ASTGCN", "AAGCN", "LSTM")

#: Growth, in log1p units, beyond which a week counts as rising or falling.
#: 0.3 is a ~35% change -- large enough that "flat" is not a plausible call.
MOVE = 0.3

NATIVE = dict(loss="mse_z", dist="point")
BEST = dict(loss="nb", dist="nb", use_season=True)
PHYS = dict(lam_param="log", state_fit=True)


def arms() -> list[dict]:
    out = []
    for a in ARCHS:
        out.append(dict(config="native", name=f"{a}", backbone=a, head="direct", **NATIVE))
    out.append(dict(config="native", name="SEIR-LSTM", backbone="LSTM", head="foi",
                    **NATIVE, **PHYS))
    for a in ARCHS:
        out.append(dict(config="best", name=f"{a}", backbone=a, head="direct", **BEST))
    out.append(dict(config="best", name="SEIR-LSTM", backbone="LSTM", head="foi_res",
                    **BEST, **PHYS))
    return out


def _job(spec: dict):
    import torch

    import train
    torch.set_num_threads(1)
    data = cd.load()
    fold = next(f for f in core.build_folds(data.cases, data.missing)
                if f.origin == spec["origin"])
    edge, fixed = core.adjacency(data.names)
    cfg = {k: v for k, v in spec.items() if k not in ("config", "name", "origin", "seed")}
    row, extra = train.run_fold(data, fold, seed=spec["seed"], edge=edge, fixed=fixed,
                                keep=True, **cfg)
    return (spec["config"], spec["name"], spec["origin"], spec["seed"]), row, extra


def collect(workers: int, epochs: int) -> None:
    jobs = [dict(a, origin=o, seed=s, epochs=epochs)
            for a in arms() for o, s in itertools.product(core.ORIGINS, SEEDS)]
    print(f"{len(jobs)} runs on {workers} workers", flush=True)
    store, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_job, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                key, row, extra = fut.result()
                store[key] = {"row": row, "extra": extra}
                print(f"[{k:3d}/{len(jobs)}] {key[0]:6s} {key[1]:10s} o{key[2]} s{key[3]} "
                      f"val {row['val_RMSE']:6.2f}", flush=True)
            except Exception as exc:  # a dead worker must not silently shrink the set
                print(f"[{k:3d}/{len(jobs)}] FAILED: {exc!r}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(pickle.dumps(store))
    print(f"wrote {OUT} ({len(store)} runs) in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- analysis --

def _pool(store: dict, config: str, name: str, split: str = "val") -> dict | None:
    """Seed-averaged forecasts for one arm, pooled over the three origins.

    Averaging seeds first removes initialisation noise, which is not what is
    being diagnosed -- the question is what the architecture does.
    """
    parts = []
    for o in core.ORIGINS:
        runs = [store[(config, name, o, s)]["extra"][split] for s in SEEDS
                if (config, name, o, s) in store]
        if not runs:
            return None
        r0 = runs[0]
        parts.append({"pred": np.mean([r["pred"] for r in runs], axis=0),
                      "truth": r0["truth"], "persist": r0["persist"], "last": r0["last"],
                      "idx": r0["idx"], "origin": np.full(len(r0["idx"]), o)})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _thresholds(data) -> dict[float, np.ndarray]:
    """Per-district 90th percentile of weekly cases, from each fold's training weeks."""
    out = {}
    for f in core.build_folds(data.cases, data.missing):
        hist = data.cases[: int(f.idx["train"].max()) + 1]
        out[f.origin] = np.nanpercentile(hist, 90, axis=0)
    return out


def analyse() -> None:
    store = pickle.loads(OUT.read_bytes())
    data = cd.load()
    names = data.names
    _, fixed = core.adjacency(names)
    nb_mask = (fixed.numpy() > 0) & ~np.eye(len(names), dtype=bool)
    thr = _thresholds(data)
    heavy = [names.index(d) for d in ("Colombo", "Gampaha") if d in names]

    for config in ("native", "best"):
        pool = {n: _pool(store, config, n) for n in (*ARCHS, "SEIR-LSTM")}
        pool = {k: v for k, v in pool.items() if v is not None}
        ref = next(iter(pool.values()))
        truth, last, persist = ref["truth"], ref["last"], ref["persist"]
        g_true = np.log1p(truth) - np.log1p(last)[..., None]
        outbreak = truth > np.stack([thr[o] for o in ref["origin"]])[..., None]

        print(f"\n{'#' * 78}\n# config = {config}   (validation windows, 3 origins, seeds averaged)"
              f"\n{'#' * 78}")

        # 1. skill by horizon
        print("\n1. RMSE by horizon, and skill = 1 - RMSE/persistence (positive is better)")
        print(f"   {'arm':11s} {'all':>7s} {'h1':>7s} {'h2':>7s} {'h3':>7s}   "
              f"{'skill h1':>8s} {'h2':>6s} {'h3':>6s}")
        pr = [_rmse(persist[..., h], truth[..., h]) for h in range(3)]
        print(f"   {'persistence':11s} {_rmse(persist, truth):7.2f} "
              + " ".join(f"{v:7.2f}" for v in pr))
        for n, p in pool.items():
            hr = [_rmse(p["pred"][..., h], truth[..., h]) for h in range(3)]
            print(f"   {n:11s} {_rmse(p['pred'], truth):7.2f} " + " ".join(f"{v:7.2f}" for v in hr)
                  + "   " + " ".join(f"{1 - hr[h] / pr[h]:+7.3f}" for h in range(3)))

        # 2. responsiveness
        print("\n2. Does the forecast move with the truth?  Regress predicted log-growth on")
        print("   realised log-growth: slope 0 = persistence, 1 = tracks change fully.")
        print(f"   {'arm':11s} {'slope':>7s} {'r':>7s} {'r2':>7s}  {'sd(pred growth)/sd(true)':>26s}")
        for n, p in pool.items():
            gp = np.log1p(p["pred"]) - np.log1p(last)[..., None]
            x, y = g_true.ravel(), gp.ravel()
            slope = np.cov(x, y)[0, 1] / np.var(x)
            r = np.corrcoef(x, y)[0, 1]
            print(f"   {n:11s} {slope:7.3f} {r:7.3f} {r * r:7.3f}  {np.std(y) / np.std(x):26.3f}")

        # 3. bias by regime
        print(f"\n3. Mean error (pred - truth, cases) by regime; |growth| > {MOVE} is a move.")
        rising, falling = g_true > MOVE, g_true < -MOVE
        flat = ~rising & ~falling
        print(f"   share of cells: rising {rising.mean():.1%}  falling {falling.mean():.1%}  "
              f"flat {flat.mean():.1%}  outbreak {outbreak.mean():.1%}")
        print(f"   {'arm':11s} {'rising':>8s} {'falling':>8s} {'flat':>7s} {'outbreak':>9s} {'quiet':>7s}"
              f"   {'SSE share rising+falling':>24s}")
        for n, p in [("persistence", {"pred": persist}), *pool.items()]:
            e = p["pred"] - truth
            sse = e ** 2
            print(f"   {n:11s} {e[rising].mean():8.2f} {e[falling].mean():8.2f} {e[flat].mean():7.2f} "
                  f"{e[outbreak].mean():9.2f} {e[~outbreak].mean():7.2f}   "
                  f"{sse[rising | falling].sum() / sse.sum():24.1%}")

        # 4. direction of change
        print("\n4. Direction called correctly, among cells that actually moved (chance = 50%)")
        moved = rising | falling
        for n, p in pool.items():
            gp = np.log1p(p["pred"]) - np.log1p(last)[..., None]
            ok = np.sign(gp[moved]) == np.sign(g_true[moved])
            up = (np.sign(gp) > 0)[rising].mean()
            dn = (np.sign(gp) < 0)[falling].mean()
            print(f"   {n:11s} {ok.mean():6.1%}   (rises called up {up:5.1%}, falls called down {dn:5.1%})")

        # 5. where the error lives
        print("\n5. Where the squared error lives")
        for n, p in [("persistence", {"pred": persist}), *pool.items()]:
            sse = ((p["pred"] - truth) ** 2)
            share_heavy = sse[:, heavy].sum() / sse.sum()
            flat_sse = np.sort(sse.ravel())[::-1]
            top5 = flat_sse[: max(1, len(flat_sse) // 20)].sum() / flat_sse.sum()
            print(f"   {n:11s} Colombo+Gampaha {share_heavy:6.1%} of SSE   top 5% of cells {top5:6.1%}")

        # 6. do the architectures make the same mistakes?
        print("\n6. Correlation of residuals between architectures (cell level)")
        keys = list(pool)
        res = {n: (pool[n]["pred"] - truth).ravel() for n in keys}
        print("   " + " " * 11 + "".join(f"{k[:8]:>9s}" for k in keys))
        for a in keys:
            print(f"   {a:11s}" + "".join(f"{np.corrcoef(res[a], res[b])[0, 1]:9.3f}" for b in keys))
        ens = np.mean([pool[n]["pred"] for n in keys if n != "SEIR-LSTM"], axis=0)
        best_single = min(_rmse(pool[n]["pred"], truth) for n in keys)
        print(f"   mean of the {len(keys) - 1} direct encoders: RMSE {_rmse(ens, truth):.2f}  "
              f"(best single {best_single:.2f})")

        # 7. spatial structure left in the residuals
        print("\n7. Is there spatial signal left?  Mean correlation of standardised residuals")
        print("   between district pairs, over (window, horizon).")
        for n, p in [("persistence", {"pred": persist}), *pool.items()]:
            e = np.log1p(p["pred"]) - np.log1p(truth)                      # (K,N,H)
            e = e.transpose(0, 2, 1).reshape(-1, e.shape[1])                # (K*H, N)
            e = (e - e.mean(0)) / (e.std(0) + 1e-9)
            c = np.corrcoef(e.T)
            iu = np.triu_indices_from(c, 1)
            nb = nb_mask[iu]
            print(f"   {n:11s} neighbours {c[iu][nb].mean():+.3f}   non-neighbours "
                  f"{c[iu][~nb].mean():+.3f}")

        # 8. could a linear model on the same inputs have fixed the residual?
        print("\n8. Residual left on the table: fit OLS on TRAIN residuals, score on VAL")
        print("   (log space; out-of-sample R2 > 0 means signal the architecture missed)")
        for n in keys:
            tr, va = _pool(store, config, n, "train"), pool[n]
            r2 = _residual_r2(tr, va, data, fixed.numpy())
            print(f"   {n:11s} R2 = {r2:+.3f}")


def _features(part: dict, data, adj: np.ndarray) -> np.ndarray:
    """What every architecture was given, flattened per (window, district, horizon)."""
    k = len(part["idx"])
    hist = np.stack([np.log1p(np.nan_to_num(data.cases[i - core.WINDOW : i])).T
                     for i in part["idx"]])                                   # (K,N,W)
    nbr = np.einsum("ij,kjw->kiw", adj, hist)[..., -1:]                      # (K,N,1)
    season = core.seasonal_features(data.week_start, part["idx"])            # (K,4)
    season = np.repeat(season[:, None, :], hist.shape[1], 1)
    pred = np.log1p(part["pred"])                                             # (K,N,H)
    rows = []
    for h in range(pred.shape[-1]):
        onehot = np.zeros((k, hist.shape[1], 3)); onehot[..., h] = 1
        rows.append(np.concatenate([hist, nbr, season, pred[..., h:h + 1], onehot], -1))
    return np.stack(rows, 2).reshape(-1, rows[0].shape[-1])


def _residual_r2(tr: dict, va: dict, data, adj: np.ndarray) -> float:
    xt, xv = _features(tr, data, adj), _features(va, data, adj)
    yt = (np.log1p(tr["truth"]) - np.log1p(tr["pred"])).ravel()
    yv = (np.log1p(va["truth"]) - np.log1p(va["pred"])).ravel()
    xt1, xv1 = np.c_[xt, np.ones(len(xt))], np.c_[xv, np.ones(len(xv))]
    beta = np.linalg.lstsq(xt1, yt, rcond=None)[0]
    resid = yv - xv1 @ beta
    return float(1 - np.sum(resid ** 2) / np.sum((yv - yv.mean()) ** 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("collect", "analyse"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    collect(args.workers, args.epochs) if args.step == "collect" else analyse()
