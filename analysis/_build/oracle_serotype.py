"""What would serotype surveillance be worth? An oracle measurement.

The corrected SEIR-GNN beats persistence on 7 of 9 origins and then loses badly
on one: origin 0.60, the collapse after the 2017 DENV-2 epidemic, where it
over-predicts with a bias of +21.4 against persistence's +10.2. The gains are
small and steady, the loss is large and rare.

Arm A -- waning immunity (a physics fix aimed at that failure).
  Removing susceptible depletion entirely is right over ten years, because
  dengue's four serotypes give no lasting cross-protection. It is wrong in the
  months after a large epidemic, when that serotype really has burned through
  the susceptibles. So deplete S from *recent* infections only, over a memory
  window, which is temporary cross-protection in its simplest form.

Arm B -- capped correction (a safety net on top of persistence).
  Forecast = persistence + clip(mechanism - persistence, +- c * (persistence + 1)).
  Keeps the small steady gains, bounds the rare large loss. `c` is learned.

Arm C -- damped correction in log space.
  Forecast = persistence * (mechanism / persistence) ** alpha, alpha in [0, 1].
  The classic damped-trend idea, applied to a mechanistic growth estimate.

Everything else is the v2 configuration, and every arm keeps the SEIR simulator
in the prediction path. 9 disjoint origins x 3 seeds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

REPO = Path("C:/Users/ASUS/Desktop/dengue-forecasting")
HERE = Path(__file__).resolve().parent
for _p in (REPO / "analysis" / "lib", REPO / "analysis" / "_build", REPO / "src"):
    sys.path.insert(0, str(_p))

import corrected_data as cd  # noqa: E402
import run_corrected_benchmark as rcb  # noqa: E402
import run_s5_seir_gnn_v2 as v2  # noqa: E402

RHO, S0, H = v2.RHO, v2.S0, v2.HORIZON

# Serotype emergence weeks in the rebuilt index: DENV-2 from 2016-09 (week 168),
# DENV-3 from 2019-09 (week 328). Published after the fact, so any arm using them
# is an ORACLE under rule R4 and is never a forecaster.
SWITCHES = (168, 328)

ARMS = {
    "v2":            dict(memory=0,  combine="none",   oracle=False),
    "damped":        dict(memory=0,  combine="damped", oracle=False),
    "waning_52w":    dict(memory=52, combine="none",   oracle=False),
    "waning_damped": dict(memory=52, combine="damped", oracle=False),
    "oracle_w52":        dict(memory=52, combine="none",   oracle=True),
    "oracle_w52_damped": dict(memory=52, combine="damped", oracle=True),
    "oracle_full":       dict(memory=520, combine="damped", oracle=True),
}


def initial_state(cases, population, idx, memory: int, oracle: bool = False):
    """Compartment fractions.

    `memory` weeks of past infection deplete S (0 = no depletion). With
    `oracle`, depletion also restarts at each serotype emergence week, because
    immunity to the outgoing serotype gives no protection against the incoming
    one. Those dates were published years later: this is hindsight, and the arm
    is labelled an oracle.
    """
    y = np.stack([cases[i: i + H].T for i in idx])
    c1 = np.stack([cases[i - 1] for i in idx])
    c2 = np.stack([cases[i - 2] for i in idx])
    pop = np.stack([population[i - 1] for i in idx])
    onset_i = c1 / (RHO * pop)
    onset_e = c2 / (RHO * pop)
    if memory > 0:
        starts = []
        for i in idx:
            lo = max(0, i - memory)
            if oracle:
                past = [w for w in SWITCHES if w <= i]
                if past:
                    lo = max(lo, past[-1])     # immunity to the old serotype does not carry
            starts.append(lo)
        recent = np.stack([np.nansum(cases[lo: i], axis=0) for lo, i in zip(starts, idx)])
        s = np.clip(S0 - recent / (RHO * pop), 0.01, 1.0)
    else:
        s = np.full_like(onset_i, S0)
    t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return (t(s), t(onset_i), t(onset_e), t(pop).unsqueeze(-1), t(y),
            t(np.repeat(c1[:, :, None], H, axis=2)))


class Combiner(nn.Module):
    """How the mechanism is allowed to move the forecast away from persistence."""

    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        if mode == "capped":
            self.raw_c = nn.Parameter(torch.tensor(-0.5))     # softplus -> ~0.47
        elif mode == "damped":
            self.raw_a = nn.Parameter(torch.tensor(0.0))      # sigmoid -> 0.5

    def forward(self, mech, pers):
        if self.mode == "none":
            return mech
        if self.mode == "capped":
            c = nn.functional.softplus(self.raw_c)
            lim = c * (pers + 1.0)
            return pers + torch.clamp(mech - pers, -lim, lim)
        a = torch.sigmoid(self.raw_a)
        return torch.expm1(torch.log1p(pers.clamp_min(0))
                           + a * (torch.log1p(mech.clamp_min(0))
                                  - torch.log1p(pers.clamp_min(0))))


def run_job(job):
    arm, seed = job["arm"], job["seed"]
    cfg, spec = v2.ARMS["v2"], ARMS[job["arm"]]
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    ft = torch.tensor(job["features"], dtype=torch.float32)
    m = v2.SEIRGNNv2("STGAT", job["features"].shape[-1],
                     torch.tensor(job["edge_index"], dtype=torch.long),
                     torch.tensor(job["adj"], dtype=torch.float32),
                     mass_action=True, per_week=True, learn_rho=True,
                     learn_rates=True, mod_clamp=cfg["mod_clamp"], learn_state=True)
    comb = Combiner(spec["combine"])
    opt = optim.Adam(list(m.parameters()) + list(comb.parameters()),
                     lr=0.003, weight_decay=1e-4)

    prep = lambda idx: initial_state(job["cases"], job["population"], idx,  # noqa: E731
                                     spec["memory"], spec["oracle"])
    tr, va, te = prep(job["train_idx"]), prep(job["val_idx"]), prep(job["test_idx"])
    xs = {k: ft[job[f"{k}_idx"]] for k in ("train", "val", "test")}

    def fwd(x, b):
        mech = m.to_cases(m(x, m.build_state(b[0], b[1], b[2])), b[3])
        return comb(mech, b[5])

    best, bw, wait = float("inf"), None, 0
    for e in range(400):
        m.train()
        opt.zero_grad()
        torch.mean((fwd(xs["train"], tr) - tr[4]) ** 2).backward()
        torch.nn.utils.clip_grad_norm_(list(m.parameters()) + list(comb.parameters()), 5.0)
        opt.step()
        if e % 3 == 0:
            m.eval()
            with torch.no_grad():
                v = torch.sqrt(torch.mean((fwd(xs["val"], va) - va[4]) ** 2)).item()
            if v < best - 1e-5:
                best, wait = v, 0
                bw = ({k: t.clone() for k, t in m.state_dict().items()},
                      {k: t.clone() for k, t in comb.state_dict().items()})
            else:
                wait += 1
                if wait >= 20:
                    break
    if bw:
        m.load_state_dict(bw[0])
        comb.load_state_dict(bw[1])
    m.eval()
    with torch.no_grad():
        tp = fwd(xs["test"], te)
        rec = {
            "arm": arm, "origin": job["origin"], "seed": seed,
            "val_RMSE": best,
            "test_RMSE": torch.sqrt(torch.mean((tp - te[4]) ** 2)).item(),
            "test_MAE": torch.mean((tp - te[4]).abs()).item(),
            "bias": (tp - te[4]).mean().item(),
            "pers_RMSE": torch.sqrt(torch.mean((te[5] - te[4]) ** 2)).item(),
            "pers_MAE": torch.mean((te[5] - te[4]).abs()).item(),
            "elapsed": round(time.time() - t0, 1),
        }
        if spec["combine"] == "capped":
            rec["cap_c"] = float(nn.functional.softplus(comb.raw_c))
        if spec["combine"] == "damped":
            rec["damp_alpha"] = float(torch.sigmoid(comb.raw_a))
    print(f"  [OK] {arm:16s} o{job['origin']} s{seed} | test {rec['test_RMSE']:7.2f} "
          f"| pers {rec['pers_RMSE']:7.2f} | bias {rec['bias']:+6.2f} "
          f"({rec['elapsed']:.0f}s)", flush=True)
    return rec


def perm_p(x):
    x = np.asarray(x)
    f = np.array(list(itertools.product([-1, 1], repeat=len(x))))
    return float(np.mean(np.abs((f * x).mean(1)) >= abs(x.mean())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="*", default=list(ARMS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(HERE / "persistence_plus.json"))
    args = ap.parse_args()

    data = cd.load()
    cases, adj, artifact, missing, _ = rcb.prepare("rebuilt")
    folds = v2.build_folds(cases, missing, "9origin")
    src, dst = np.nonzero(adj)
    edge_index = np.stack([src, dst])

    jobs = []
    for arm in args.arm:
        for fold in folds:
            feats = v2.features_train_only(data, "cases", fold)
            for seed in (0, 1, 2):
                jobs.append({"arm": arm, "seed": seed, "origin": fold.origin,
                             "train_idx": np.asarray(fold.train_index),
                             "val_idx": np.asarray(fold.val_index),
                             "test_idx": np.asarray(fold.test_index),
                             "cases": cases, "features": feats,
                             "population": data.population,
                             "edge_index": edge_index, "adj": adj})

    print(f"{len(jobs)} jobs / {args.workers} workers | arms: {', '.join(args.arm)}",
          flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run_job, j) for j in jobs]):
            try:
                out.append(f.result())
            except Exception as exc:
                import traceback
                print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    df = pd.DataFrame(out)
    print("\n=== 9 disjoint origins x 3 seeds ===")
    print(df.groupby("arm")[["val_RMSE", "test_RMSE", "test_MAE", "bias"]]
            .mean().round(2).sort_values("val_RMSE").to_string())
    print(f"persistence: RMSE {df.pers_RMSE.mean():.2f}  MAE {df.pers_MAE.mean():.2f}")
    print("\n=== against persistence, paired and clustered by origin ===")
    for arm, g in df.groupby("arm"):
        o = g.groupby("origin")[["test_RMSE", "pers_RMSE"]].mean()
        d = o.test_RMSE - o.pers_RMSE
        print(f"  {arm:16s} mean {d.mean():+6.2f} | wins {int((d < 0).sum())}/9 "
              f"| p = {perm_p(d.values):.4f}")
    if "cap_c" in df:
        print("\nlearned cap c:", df.cap_c.dropna().round(3).unique()[:6])
    if "damp_alpha" in df:
        print("learned damping alpha:", df.damp_alpha.dropna().round(3).unique()[:6])
    return 0


if __name__ == "__main__":
    sys.exit(main())
