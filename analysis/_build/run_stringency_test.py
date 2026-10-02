"""Does the COVID-19 policy record repair the window no epidemiology could?

The largest single loss in the frozen 9-origin protocol is origin 0.60, the
window 2019-11 to 2020-06, where national cases collapse from 3057 to 299 a week
and the corrected physics arm over-predicts with a bias of +19.8. EXP-036 showed
that serotype timing does nothing for it: the collapse is a policy shock, not an
epidemiological one.

`data/external/covid_response_weekly.csv` carries the OxCGRT stringency index and
Google mobility on the project's own calendar. Both were published within days of
the events they describe, so they are covariates under rule R2.

Two ways to use stringency, because they test different claims:

* ``feature`` -- an extra input channel. Tests whether the encoder can find a use
  for it.
* ``beta`` -- a multiplier on the transmission rate,
  ``beta_eff = beta * exp(-s * stringency)`` with ``s >= 0`` learned. This is the
  mechanistically correct place for it: movement restrictions reduce contact, and
  contact is what beta measures. A model that finds no use for it drives ``s`` to
  zero and recovers the baseline.

Rule R2 is respected: a forecast at week *i* reads stringency up to week *i*-1,
never the weeks it is forecasting. Weeks before the series begins are set to 0,
which is a statement of fact rather than an imputation -- these policies did not
exist then -- and is recorded as a modelling decision in `DATA_PROVENANCE.md`.

Usage::

    python analysis/_build/fetch_covid_response.py   # if not already present
    python analysis/_build/run_stringency_test.py
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
import torch.nn as nn
import torch.optim as optim

REPO = Path(__file__).resolve().parent.parent.parent
for _p in (REPO / "analysis" / "lib", REPO / "analysis" / "_build", REPO / "src"):
    sys.path.insert(0, str(_p))

import corrected_data as cd  # noqa: E402
import run_corrected_benchmark as rcb  # noqa: E402
import run_s5_seir_gnn_v2 as v2  # noqa: E402

COVID = REPO / "data" / "external" / "covid_response_weekly.csv"
ARMS = {
    "none":    dict(use="none"),
    "feature": dict(use="feature"),
    "beta":    dict(use="beta"),
}


def stringency_series() -> np.ndarray:
    """Weekly stringency, 0-1, zero before the policies existed."""
    if not COVID.exists():
        raise SystemExit("run analysis/_build/fetch_covid_response.py first")
    df = pd.read_csv(COVID)
    s = df["stringency_index"].to_numpy(dtype=np.float64)
    return np.nan_to_num(s, nan=0.0) / 100.0


class BetaDamped(v2.SEIRGNNv2):
    """v2 with transmission scaled by the policy response at the forecast origin."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_s = nn.Parameter(torch.tensor(0.0))

    def set_policy(self, policy: torch.Tensor) -> None:
        self._policy = policy                    # (B,) stringency at week i-1

    def forward(self, x, st0):
        h = self.backbone(self.in_proj(x).squeeze(-1), self.edge_index)
        z = self.head_fc(h)
        if z.shape[-1] == 1:
            z = z.repeat(1, 1, v2.HORIZON)
        beta = (nn.functional.softplus(self.beta0)
                * torch.exp(z.clamp(-self.mod_clamp, self.mod_clamp))
                if self.mod_clamp > 0 else nn.functional.softplus(z))
        damp = torch.exp(-nn.functional.softplus(self.raw_s)
                         * self._policy.view(-1, 1, 1))
        lam = beta * damp * st0[..., 2].unsqueeze(-1)
        omega, gamma = self.rates()
        _, inc = v2.seir_sim.simulate_weeks(st0, lam, omega, gamma, substeps=7)
        return inc


def run_job(job):
    arm, seed = job["arm"], job["seed"]
    use, cfg = ARMS[arm]["use"], v2.ARMS["v2"]
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    ft = torch.tensor(job["features"], dtype=torch.float32)
    cls = BetaDamped if use == "beta" else v2.SEIRGNNv2
    model = cls("STGAT", job["features"].shape[-1],
                torch.tensor(job["edge_index"], dtype=torch.long),
                torch.tensor(job["adj"], dtype=torch.float32),
                mass_action=True, per_week=True, learn_rho=True, learn_rates=True,
                mod_clamp=cfg["mod_clamp"], learn_state=True)
    opt = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    sti = job["stringency"]

    def prep(idx):
        st0, s_, oi, oe, pop, y = v2.initial_state(job["cases"], job["population"], idx,
                                                   True, 0.0)
        pol = torch.tensor(sti[np.asarray(idx) - 1], dtype=torch.float32)   # week i-1
        return ft[idx], st0, pop, y, s_, oi, oe, pol

    tr, va, te = prep(job["train_idx"]), prep(job["val_idx"]), prep(job["test_idx"])

    def fwd(b):
        if use == "beta":
            model.set_policy(b[7])
        return model.to_cases(model(b[0], model.build_state(b[4], b[5], b[6])), b[2])

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
        rec = {"arm": arm, "origin": job["origin"], "seed": seed, "val_RMSE": best,
               "test_RMSE": torch.sqrt(torch.mean((tp - te[3]) ** 2)).item(),
               "test_MAE": torch.mean((tp - te[3]).abs()).item(),
               "bias": (tp - te[3]).mean().item(),
               "policy_weight": float(nn.functional.softplus(model.raw_s))
               if use == "beta" else 0.0,
               "elapsed": round(time.time() - t0, 1)}
    print(f"  [OK] {arm:8s} o{job['origin']} s{seed} | test {rec['test_RMSE']:7.2f} "
          f"| bias {rec['bias']:+7.2f} | s {rec['policy_weight']:.3f}", flush=True)
    return rec


def features_with_policy(data, fold, sti: np.ndarray) -> np.ndarray:
    """v2's cases channel plus stringency over the same input window."""
    base = v2.features_train_only(data, "cases", fold)
    T, N, W, _ = base.shape
    out = np.zeros((T, N, W, 2), dtype=np.float32)
    out[..., 0] = base[..., 0]
    for i in range(W, T):
        out[i, :, :, 1] = np.tile(sti[i - W:i], (N, 1))
    return out


def perm_p(x):
    x = np.asarray(x)
    f = np.array(list(itertools.product([-1, 1], repeat=len(x))))
    return float(np.mean(np.abs((f * x).mean(1)) >= abs(x.mean())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs="*", default=list(ARMS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(REPO / "analysis" / "results" / "seir_gnn" / "s5_v2"
                                        / "stringency_test_runs.csv"))
    args = ap.parse_args()

    data = cd.load()
    cases, adj, artifact, missing, _ = rcb.prepare("rebuilt")
    sti = stringency_series()
    print(f"stringency: {int((sti > 0).sum())} weeks non-zero, max {sti.max():.2f}")
    src, dst = np.nonzero(adj)
    edge_index = np.stack([src, dst])
    folds = v2.build_folds(cases, missing, "9origin")

    jobs = []
    for arm in args.arm:
        for fold in folds:
            feats = (features_with_policy(data, fold, sti) if ARMS[arm]["use"] == "feature"
                     else v2.features_train_only(data, "cases", fold))
            for seed in (0, 1, 2):
                jobs.append({"arm": arm, "seed": seed, "origin": fold.origin,
                             "train_idx": np.asarray(fold.train_index),
                             "val_idx": np.asarray(fold.val_index),
                             "test_idx": np.asarray(fold.test_index),
                             "cases": cases, "features": feats, "stringency": sti,
                             "population": data.population,
                             "edge_index": edge_index, "adj": adj})

    print(f"{len(jobs)} jobs / {args.workers} workers", flush=True)
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
    print("\n=== 9 disjoint origins x 3 seeds ===")
    print(df.groupby("arm")[["val_RMSE", "test_RMSE", "test_MAE", "policy_weight"]]
            .mean().round(3).sort_values("val_RMSE").to_string())
    print("\n=== origin 0.60, the COVID collapse window ===")
    print(df[df.origin == 0.6].groupby("arm")[["test_RMSE", "test_MAE", "bias"]]
            .mean().round(2).to_string())
    ref = df[df.arm == "none"].groupby("origin")["test_RMSE"].mean()
    print("\n=== against the baseline, paired and clustered by origin ===")
    for arm, g in df.groupby("arm"):
        if arm == "none":
            continue
        d = g.groupby("origin")["test_RMSE"].mean() - ref
        print(f"  {arm:8s} mean {d.mean():+6.2f} | better {int((d < 0).sum())}/9 "
              f"| p = {perm_p(d.values):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
