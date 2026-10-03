"""Movement restrictions as a mechanistic prior on the contact rate.

This is the complement to a learned covariate, and it exists because a learned
covariate cannot work on the folds that fail. Lockdown weeks present in each
fold's **training** data:

    origin 0.50  0     origin 0.70   8
    origin 0.55  0     origin 0.75  35
    origin 0.60  0     origin 0.80  63
    origin 0.65  0     origin 0.85  91
                       origin 0.90 118

The two windows the model gets worst -- 0.60, where cases collapse from 3057 to
299 a week, and 0.65, which sits inside the first lockdown -- have **no lockdown
examples to learn from**. A fitted coefficient on a covariate that is identically
zero across training is not estimable, so no amount of data wrangling makes a
learned arm able to repair those folds.

What *is* available there is knowledge from outside the series: when a government
closes schools and workplaces and restricts movement, contact rates fall. That is
not a claim this dataset has to establish. So it enters as a **prior on the
mechanism**, not as a parameter:

    beta_effective = beta * (1 - s / 100) ** gamma

with `s` the OxCGRT stringency index and `gamma` **fixed, never fitted**. At
s = 0 the multiplier is 1 and the arm is identical to the baseline, which is why
every pre-2020 window is unchanged by construction.

Causality (rule R2). The multiplier uses stringency at week *i*-1, strictly
before the forecast origin, held constant across the three forecast weeks --
a persistence assumption about policy, stated rather than hidden. Weeks with no
OxCGRT coverage take s = 0, which is the factual value: these policies did not
exist before 2020.

**Declared in advance:** the functional form and the gamma grid below were chosen
before running, from the mechanism alone, and gamma is not selected on test data.
The decision to test this at all was prompted by the 2020 failure, which is a
design-level use of hindsight and is reported as such.

Usage::

    python analysis/_build/fetch_covid_response.py        # writes the weekly file
    python analysis/_build/run_stringency_prior.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

REPO = Path(__file__).resolve().parent.parent.parent
for _p in (REPO / "analysis" / "lib", REPO / "analysis" / "_build", REPO / "src"):
    sys.path.insert(0, str(_p))

import corrected_data as cd  # noqa: E402
import run_corrected_benchmark as rcb  # noqa: E402
import run_s5_seir_gnn_v2 as v2  # noqa: E402

COVID = REPO / "data" / "external" / "covid_response_weekly.csv"
OUT = REPO / "analysis" / "results" / "seir_gnn" / "s5_v2" / "stringency_prior_runs.csv"

ARMS = {
    "base": None,       # the v2 arm, unchanged
    "prior_g0.5": 0.5,  # a mild contact reduction
    "prior_g1": 1.0,    # contacts fall in proportion to the index
    "prior_g2": 2.0,    # a strong reduction
}


def stringency_series() -> np.ndarray:
    """Weekly stringency, 0 where the index does not exist yet (pre-2020)."""
    if not COVID.exists():
        raise SystemExit("run analysis/_build/fetch_covid_response.py first")
    df = pd.read_csv(COVID)
    return df["stringency_index"].fillna(0.0).to_numpy(dtype=np.float64)


def run_job(job):
    arm, seed, gamma = job["arm"], job["seed"], job["gamma"]
    cfg = v2.ARMS["v2"]
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    ft = torch.tensor(job["features"], dtype=torch.float32)
    model = v2.SEIRGNNv2("STGAT", job["features"].shape[-1],
                         torch.tensor(job["edge_index"], dtype=torch.long),
                         torch.tensor(job["adj"], dtype=torch.float32),
                         mass_action=True, per_week=True, learn_rho=True,
                         learn_rates=True, mod_clamp=cfg["mod_clamp"], learn_state=True)
    opt = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    sti = job["stringency"]

    def prep(idx):
        st0, s_, oi, oe, pop, y = v2.initial_state(job["cases"], job["population"], idx,
                                                   True, 0.0)
        # stringency at i-1: strictly before the origin, held across the horizon
        s_prev = np.array([sti[i - 1] for i in idx], dtype=np.float32)
        mult = np.ones_like(s_prev) if gamma is None else (1.0 - s_prev / 100.0) ** gamma
        return (ft[idx], st0, pop, y, s_, oi, oe,
                torch.tensor(mult, dtype=torch.float32).view(-1, 1, 1))

    tr, va, te = prep(job["train_idx"]), prep(job["val_idx"]), prep(job["test_idx"])

    def fwd(b):
        """As v2.SEIRGNNv2.forward, with the contact-rate multiplier applied to beta."""
        st = model.build_state(b[4], b[5], b[6])
        z = model.head_fc(model.backbone(model.in_proj(b[0]).squeeze(-1), model.edge_index))
        if z.shape[-1] == 1:
            z = z.repeat(1, 1, v2.HORIZON)
        beta = (torch.nn.functional.softplus(model.beta0)
                * torch.exp(z.clamp(-model.mod_clamp, model.mod_clamp))
                if model.mod_clamp > 0 else torch.nn.functional.softplus(z))
        beta = beta * b[7]                       # the prior: contacts fall under restriction
        lam = beta * st[..., 2].unsqueeze(-1)
        omega, gamma_r = model.rates()
        _, inc = v2.seir_sim.simulate_weeks(st, lam, omega, gamma_r, substeps=7)
        return model.to_cases(inc, b[2])

    best, bw, wait = float("inf"), None, 0
    for e in range(cfg["epochs"]):
        model.train()
        opt.zero_grad()
        torch.mean((fwd(tr) - tr[3]) ** 2).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if e % 3 == 0:
            model.eval()
            with torch.no_grad():
                v = torch.sqrt(torch.mean((fwd(va) - va[3]) ** 2)).item()
            if v < best - 1e-5:
                best, wait = v, 0
                bw = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 20:
                    break
    if bw:
        model.load_state_dict(bw)
    model.eval()
    with torch.no_grad():
        tp = fwd(te)
        rec = {"arm": arm, "gamma": gamma if gamma is not None else 0.0,
               "origin": job["origin"], "seed": seed, "val_RMSE": best,
               "test_RMSE": torch.sqrt(torch.mean((tp - te[3]) ** 2)).item(),
               "test_MAE": torch.mean((tp - te[3]).abs()).item(),
               "bias": (tp - te[3]).mean().item(),
               "mean_multiplier": float(te[7].mean()),
               "elapsed": round(time.time() - t0, 1)}
    print(f"  [OK] {arm:11s} o{job['origin']} s{seed} | test {rec['test_RMSE']:7.2f} "
          f"| bias {rec['bias']:+7.2f} | mult {rec['mean_multiplier']:.3f}", flush=True)
    return rec


def perm_p(x):
    x = np.asarray(x)
    f = np.array(list(itertools.product([-1, 1], repeat=len(x))))
    return float(np.mean(np.abs((f * x).mean(1)) >= abs(x.mean())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs="*", default=list(ARMS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    data = cd.load()
    cases, adj, artifact, missing, _ = rcb.prepare("rebuilt")
    sti = stringency_series()
    src, dst = np.nonzero(adj)
    edge_index = np.stack([src, dst])
    folds = v2.build_folds(cases, missing, "9origin")

    print("lockdown weeks (stringency > 20) inside each fold's training data")
    for f in folds:
        tr = np.asarray(f.train_index)
        print(f"  origin {f.origin}: {int((sti[tr.min():tr.max() + 1] > 20).sum())}")

    jobs = []
    for arm in args.arm:
        for fold in folds:
            feats = v2.features_train_only(data, "cases", fold)
            for seed in (0, 1, 2):
                jobs.append({"arm": arm, "gamma": ARMS[arm], "seed": seed,
                             "origin": fold.origin, "stringency": sti,
                             "train_idx": np.asarray(fold.train_index),
                             "val_idx": np.asarray(fold.val_index),
                             "test_idx": np.asarray(fold.test_index),
                             "cases": cases, "features": feats,
                             "population": data.population,
                             "edge_index": edge_index, "adj": adj})

    print(f"\n{len(jobs)} jobs / {args.workers} workers", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run_job, j) for j in jobs]):
            try:
                out.append(f.result())
            except Exception as exc:
                import traceback
                print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()

    df = pd.DataFrame(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\n=== all nine origins ===")
    print(df.groupby("arm")[["val_RMSE", "test_RMSE", "test_MAE"]].mean().round(2)
            .sort_values("test_RMSE").to_string())
    print("\n=== the two COVID windows, where no learned arm can help ===")
    covid = df[df.origin.isin([0.6, 0.65])]
    print(covid.groupby(["arm", "origin"])[["test_RMSE", "bias", "mean_multiplier"]]
          .mean().round(2).to_string())
    pers = {0.6: 48.71, 0.65: 11.34}
    print("\npersistence on those windows:", pers)
    if "base" in df.arm.values:
        ref = df[df.arm == "base"].groupby("origin")["test_RMSE"].mean()
        print("\n=== against the baseline, paired and clustered by origin ===")
        for arm, g in df.groupby("arm"):
            if arm == "base":
                continue
            d = g.groupby("origin")["test_RMSE"].mean() - ref
            print(f"  {arm:11s} all nine: mean {d.mean():+6.2f} | better "
                  f"{int((d < 0).sum())}/9 | p = {perm_p(d.values):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
