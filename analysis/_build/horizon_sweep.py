"""Does the physics advantage grow with the forecast horizon?

At three weeks the forecast is mostly the incubation-and-reporting pipeline
draining, which is arithmetically close to persistence. Further out the pipeline
empties and transmission has to do the work, so a mechanism should pull ahead
while persistence decays.

Three arms, identical except for what produces the forecast:
  physics  -- the corrected SEIR-GNN (mass action, stock state, no depletion,
              learned rho / rates / state scaling, one beta per forecast week)
  direct   -- the same encoder with the SEIR removed
  persistence -- last observed week, carried forward

Protocol: `rebuilt`, 9 disjoint origins (0.50-0.90), 3 seeds, window 3,
training-only normalisation, squared error, early stopping on validation RMSE,
metrics pooled over (window, district, horizon) in raw counts.
"""

from __future__ import annotations

import argparse
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

import adaptive as base  # noqa: E402
import corrected_data as cd  # noqa: E402
import reproduced as arch_lib  # noqa: E402
import run_corrected_benchmark as rcb  # noqa: E402
import seir_sim  # noqa: E402

WINDOW = 3
RHO = 1.0 / 11.0
S0 = 1.0 - 0.682
OMEGA, GAMMA = 0.7 / 7.0, 1.0 / 7.0
RATE_LO, RATE_HI = 0.02, 0.45


class Model(nn.Module):
    def __init__(self, in_dim, horizon, edge_index, adj, physics: bool):
        super().__init__()
        self.h = horizon
        self.physics = physics
        self.edge_index = edge_index
        self.in_proj = nn.Linear(in_dim, 1)
        self.backbone = arch_lib.build("STGAT", 25, WINDOW, horizon, edge_index=edge_index)
        self.head = nn.Linear(horizon, horizon)
        if physics:
            nn.init.constant_(self.head.bias, -0.593)
            nn.init.normal_(self.head.weight, std=0.1)
            self.log_rho = nn.Parameter(torch.zeros(()))
            self.log_e = nn.Parameter(torch.tensor(math.log(1 / (1 - math.exp(-OMEGA * 7)))))
            self.log_i = nn.Parameter(torch.tensor(math.log(1 / (1 - math.exp(-GAMMA * 7)))))
            self.raw_omega = nn.Parameter(torch.tensor(math.log(math.expm1(OMEGA))))
            self.raw_gamma = nn.Parameter(torch.tensor(math.log(math.expm1(GAMMA))))

    def rates(self):
        return (nn.functional.softplus(self.raw_omega).clamp(RATE_LO, RATE_HI),
                nn.functional.softplus(self.raw_gamma).clamp(RATE_LO, RATE_HI))

    def state(self, s_, oi, oe):
        i_ = (oi * torch.exp(self.log_i.clamp(-1, 3))).clamp(1e-6, 0.5)
        e_ = (oe * torch.exp(self.log_e.clamp(-1, 3))).clamp(1e-6, 0.5)
        r_ = (1.0 - s_ - e_ - i_).clamp(0.0, 1.0)
        return torch.stack([s_, e_, i_, r_], dim=-1)

    def forward(self, x, s_, oi, oe, pop, mean, std):
        z = self.head(self.backbone(self.in_proj(x).squeeze(-1), self.edge_index))
        if not self.physics:
            return torch.expm1(torch.clamp(z * std + mean, -1.0, 12.0))
        st = self.state(s_, oi, oe)
        beta = nn.functional.softplus(z)
        lam = beta * st[..., 2].unsqueeze(-1)
        omega, gamma = self.rates()
        _, inc = seir_sim.simulate_weeks(st, lam, omega, gamma, substeps=7)
        return inc * RHO * pop * torch.exp(self.log_rho.clamp(-2.0, 2.0))


def build_folds(cases, missing, horizon):
    bad = {int(m) for m in missing}
    ids = list(range(WINDOW, cases.shape[0] - horizon))
    clean = lambda i: not any(t in bad for t in range(i - WINDOW, i + horizon))  # noqa: E731
    to = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    folds = []
    for origin in [round(float(x), 6) for x in np.linspace(0.50, 0.90, 9)]:
        cut, end = int(origin * len(ids)), int(min(origin + 0.05, 1.0) * len(ids))
        tr = [i for i in ids[: cut - 30] if clean(i)]
        va = [i for i in ids[cut - 30: cut] if clean(i)]
        te = [i for i in ids[cut:end] if clean(i)]
        hist = np.log1p(cases[: ids[: cut - 30][-1] + 1])
        mean, std = float(np.nanmean(hist)), float(np.nanstd(hist) + 1e-8)
        z = (np.log1p(cases) - mean) / std
        pack = lambda ch: (to(np.stack([z[i - WINDOW: i].T for i in ch])),  # noqa: E731
                           to(np.stack([z[i: i + horizon].T for i in ch])),
                           to(np.stack([np.repeat(z[i - 1][:, None], horizon, 1) for i in ch])))
        folds.append(base.Fold(origin, *pack(tr), *pack(va), *pack(te), mean, std,
                               np.asarray(te), np.asarray(tr), np.asarray(va)))
    return folds


def batch(cases, population, feat_t, idx, horizon):
    y = np.stack([cases[i: i + horizon].T for i in idx])
    c1 = np.stack([cases[i - 1] for i in idx])
    c2 = np.stack([cases[i - 2] for i in idx])
    pop = np.stack([population[i - 1] for i in idx])
    oi = c1 / (RHO * pop) / (1 - math.exp(-GAMMA * 7))
    oe = c2 / (RHO * pop) / (1 - math.exp(-OMEGA * 7))
    s = np.full_like(oi, S0)
    t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return (feat_t[idx], t(s), t(oi), t(oe), t(pop).unsqueeze(-1), t(y),
            t(np.repeat(c1[:, :, None], horizon, axis=2)))


def run_job(job):
    h, physics, seed = job["horizon"], job["physics"], job["seed"]
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    ft = torch.tensor(job["features"], dtype=torch.float32)
    m = Model(job["features"].shape[-1], h,
              torch.tensor(job["edge_index"], dtype=torch.long),
              torch.tensor(job["adj"], dtype=torch.float32), physics)
    opt = optim.Adam(m.parameters(), lr=0.003, weight_decay=1e-4)
    mean, std = job["mean"], job["std"]
    prep = lambda idx: batch(job["cases"], job["population"], ft, idx, h)  # noqa: E731
    tr, va, te = prep(job["train_idx"]), prep(job["val_idx"]), prep(job["test_idx"])
    fwd = lambda b: m(b[0], b[1], b[2], b[3], b[4], mean, std)  # noqa: E731

    best, bw, wait = float("inf"), None, 0
    for e in range(job["epochs"]):
        m.train()
        opt.zero_grad()
        torch.mean((fwd(tr) - tr[5]) ** 2).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        if e % 3 == 0:
            m.eval()
            with torch.no_grad():
                v = torch.sqrt(torch.mean((fwd(va) - va[5]) ** 2)).item()
            if v < best - 1e-5:
                best, wait = v, 0
                bw = {k: t.clone() for k, t in m.state_dict().items()}
            else:
                wait += 1
                if wait >= 20:
                    break
    if bw:
        m.load_state_dict(bw)
    m.eval()
    with torch.no_grad():
        tp = fwd(te)
        rec = {
            "horizon": h, "arm": "physics" if physics else "direct",
            "origin": job["origin"], "seed": seed,
            "val_RMSE": torch.sqrt(torch.mean((fwd(va) - va[5]) ** 2)).item(),
            "test_RMSE": torch.sqrt(torch.mean((tp - te[5]) ** 2)).item(),
            "test_MAE": torch.mean((tp - te[5]).abs()).item(),
            "pers_RMSE": torch.sqrt(torch.mean((te[6] - te[5]) ** 2)).item(),
            "pers_MAE": torch.mean((te[6] - te[5]).abs()).item(),
            "elapsed": round(time.time() - t0, 1),
        }
    print(f"  [OK] h={h} {rec['arm']:8s} o{job['origin']} s{seed} | test "
          f"{rec['test_RMSE']:7.2f} | persistence {rec['pers_RMSE']:7.2f} "
          f"({rec['elapsed']:.0f}s)", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="*", type=int, default=[3, 6, 8, 12])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(HERE / "horizon_sweep.json"))
    args = ap.parse_args()

    data = cd.load()
    cases, adj, artifact, missing, _ = rcb.prepare("rebuilt")
    src, dst = np.nonzero(adj)
    edge_index = np.stack([src, dst])

    jobs = []
    for h in args.horizons:
        for fold in build_folds(cases, missing, h):
            T, N = cases.shape
            z = (np.log1p(np.nan_to_num(cases, nan=0.0)) - fold.mean) / fold.std
            feats = np.zeros((T, N, WINDOW, 1), dtype=np.float32)
            for i in range(WINDOW, T):
                feats[i, :, :, 0] = z[i - WINDOW: i].T
            for physics in (True, False):
                for seed in (0, 1, 2):
                    jobs.append({
                        "horizon": h, "physics": physics, "seed": seed,
                        "origin": fold.origin, "epochs": args.epochs,
                        "train_idx": np.asarray(fold.train_index),
                        "val_idx": np.asarray(fold.val_index),
                        "test_idx": np.asarray(fold.test_index),
                        "cases": cases, "features": feats,
                        "population": data.population, "edge_index": edge_index,
                        "adj": adj, "mean": fold.mean, "std": fold.std})

    print(f"{len(jobs)} jobs / {args.workers} workers | horizons {args.horizons}",
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
    piv = df.pivot_table(index="horizon", columns="arm",
                         values=["test_RMSE", "test_MAE"]).round(2)
    pers = df.groupby("horizon")[["pers_RMSE", "pers_MAE"]].mean().round(2)
    print("\n=== test error by horizon (9 origins x 3 seeds) ===")
    print(piv.join(pers).to_string())
    print("\n=== physics minus persistence (negative = physics better) ===")
    for h, g in df[df.arm == "physics"].groupby("horizon"):
        by_o = g.groupby("origin")[["test_RMSE", "pers_RMSE"]].mean()
        d = by_o["test_RMSE"] - by_o["pers_RMSE"]
        print(f"  h={h:2d}: mean {d.mean():+7.2f} RMSE | physics better on "
              f"{int((d < 0).sum())}/9 origins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
