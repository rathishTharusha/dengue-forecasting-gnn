"""Stage S5b: the physics-informed SEIR-GNN, with the physics corrected.

This is a **new** runner. `analysis/_build/run_s5_seir_gnn.py` is untouched and
stays the reference arm; `--arm baseline` reproduces it here for a paired
comparison on identical folds and seeds.

What changed, and why (each was tested on its own before being kept --
see `PHYSICS_EXPERIMENT_LOG.md`):

1. **The compartments are stocks, not weekly flows.** The old code set
   ``I = cases(t-1) / (rho * N)`` and ``E = cases(t-2) / (rho * N)``. A week's
   reported cases are a *flow* into the infectious class; the *stock* standing
   there is larger, because only a fraction of each compartment leaves per week.
   Dividing by that fraction -- ``1 - exp(-gamma * 7)`` for I and
   ``1 - exp(-omega * 7)`` for E -- fixes a systematic factor-of-two
   under-prediction.

2. **No susceptible depletion.** The old code subtracted every case since 2013
   from S, which assumes lifelong immunity to dengue. Dengue has four
   serotypes and infection with one gives no lasting protection against the
   others, so ten years of subtraction starved the model of susceptibles.

3. **Mass action.** The old head produced lambda from a bounded sigmoid with no
   reference to how many people are currently infectious. Here the network
   predicts the transmission rate ``beta`` and the force of infection is
   ``lambda = beta * I`` -- the standard SEIR form, so growth is built in.
   With ``--coupling explicit`` a neighbour import term ``alpha * A_hat I`` is
   added, which is the metapopulation form of the same quantity.

4. **A lambda per forecast week** instead of one value repeated three times.

5. **The observation model is calibrated, not assumed.** The reporting rate
   enters as ``rho * exp(c)`` with ``c`` learned, bounded to a factor of e^2.
   The literature range for dengue reporting is wide (1/2.5 to 1/30); fixing it
   at 1/11 forces the network to absorb the mismatch.

6. **omega and gamma are learned**, softplus-constrained to 0.02-0.45 per day,
   starting at Liu et al.'s values.

7. **Squared error, not SMAPE.** The protocol selects and reports RMSE. SMAPE on
   a zero-heavy district series rewards small predictions, which is the opposite.

8. **Training-only normalisation.** Log case values are scaled by log statistics
   from training weeks only. The old code scaled log values by *raw-count*
   statistics taken over every week, which both flattened the signal ~66x and
   leaked test weeks into training.

Usage::

    python analysis/_build/run_s5_seir_gnn_v2.py --arm baseline v2 v2_mech
    python analysis/_build/run_s5_seir_gnn_v2.py --arm v2 --protocol 9origin
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

_HERE = Path(__file__).resolve()
REPO = (_HERE.parents[2] if _HERE.parent.name == "_build"
        else Path("C:/Users/ASUS/Desktop/dengue-forecasting"))
for _p in (REPO / "analysis" / "lib", REPO / "analysis" / "_build", REPO / "src"):
    sys.path.insert(0, str(_p))

import adaptive as base  # noqa: E402
import corrected_data as cd  # noqa: E402
import reproduced as arch_lib  # noqa: E402
import run_corrected_benchmark as rcb  # noqa: E402
import seir_sim  # noqa: E402

WINDOW, HORIZON = 3, 3
RHO = 1.0 / 11.0
S0 = 1.0 - 0.682
OMEGA, GAMMA = 0.7 / 7.0, 1.0 / 7.0      # per day, Liu et al.
RATE_LO, RATE_HI = 0.02, 0.45            # per day, the range learned rates may take


# ------------------------------------------------------------------ model --

class SEIRGNNv2(nn.Module):
    """GNN encoder -> transmission rate -> SEIR -> reported cases."""

    def __init__(self, arch_name: str, in_dim: int, edge_index: torch.Tensor,
                 adj_dense: torch.Tensor, coupling: str = "implicit",
                 mass_action: bool = True, per_week: bool = True,
                 learn_rho: bool = True, learn_rates: bool = True,
                 mod_clamp: float = 0.0, learn_state: bool = True):
        super().__init__()
        self.coupling = coupling
        self.mass_action = mass_action
        self.learn_rho = learn_rho
        self.learn_rates = learn_rates
        self.mod_clamp = mod_clamp
        self.learn_state = learn_state
        self.edge_index = edge_index
        n_out = HORIZON if per_week else 1

        self.in_proj = nn.Linear(in_dim, 1)
        kwargs = {"adaptive": False, "channels": 8} if arch_name == "AAGCN" else {}
        self.backbone = arch_lib.build(arch_name, 25, WINDOW, HORIZON,
                                       edge_index=edge_index, **kwargs)
        self.head_fc = nn.Linear(HORIZON, n_out)

        adj = adj_dense.clone()
        adj.fill_diagonal_(0.0)
        self.register_buffer("hat_A", adj / adj.sum(1, keepdim=True).clamp_min(1.0))

        if mass_action:
            # softplus(-0.593) ~ 0.44 / day, which with gamma = 1/7 gives R0 ~ 3
            nn.init.constant_(self.head_fc.bias, -0.593)
            nn.init.normal_(self.head_fc.weight, std=0.1)
        if mod_clamp > 0:
            # the network may only modulate a learned baseline beta, by e^(+-clamp)
            self.beta0 = nn.Parameter(torch.tensor(-0.593))
            nn.init.zeros_(self.head_fc.bias)
            nn.init.normal_(self.head_fc.weight, std=0.05)
        if coupling == "explicit":
            self.alpha = nn.Parameter(torch.tensor(0.05))
        if learn_rho:
            self.log_rho_adj = nn.Parameter(torch.zeros(()))
        if learn_state:
            # start at the flow-derived factors, then let the data move them
            self.log_e_scale = nn.Parameter(
                torch.tensor(math.log(1.0 / (1.0 - math.exp(-OMEGA * 7)))))
            self.log_i_scale = nn.Parameter(
                torch.tensor(math.log(1.0 / (1.0 - math.exp(-GAMMA * 7)))))
        if learn_rates:
            self.raw_omega = nn.Parameter(torch.tensor(math.log(math.expm1(OMEGA))))
            self.raw_gamma = nn.Parameter(torch.tensor(math.log(math.expm1(GAMMA))))

    def rates(self):
        if not self.learn_rates:
            return OMEGA, GAMMA
        return (nn.functional.softplus(self.raw_omega).clamp(RATE_LO, RATE_HI),
                nn.functional.softplus(self.raw_gamma).clamp(RATE_LO, RATE_HI))

    def build_state(self, s_, onset_i, onset_e):
        """Compartment fractions, with the stock scaling learned if enabled."""
        if self.learn_state:
            i_ = (onset_i * torch.exp(self.log_i_scale.clamp(-1, 3))).clamp(1e-6, 0.5)
            e_ = (onset_e * torch.exp(self.log_e_scale.clamp(-1, 3))).clamp(1e-6, 0.5)
        else:
            i_ = onset_i.clamp(1e-6, 0.5)
            e_ = onset_e.clamp(1e-6, 0.5)
        r_ = (1.0 - s_ - e_ - i_).clamp(0.0, 1.0)
        return torch.stack([s_, e_, i_, r_], dim=-1)

    def forward(self, x: torch.Tensor, st0: torch.Tensor) -> torch.Tensor:
        """x: (B, N, window, C), st0: (B, N, 4) fractions -> cases (B, N, horizon)."""
        h = self.backbone(self.in_proj(x).squeeze(-1), self.edge_index)
        z = self.head_fc(h)
        if z.shape[-1] == 1:
            z = z.repeat(1, 1, HORIZON)

        if self.mod_clamp > 0:
            beta = (nn.functional.softplus(self.beta0)
                    * torch.exp(z.clamp(-self.mod_clamp, self.mod_clamp)))
        elif self.mass_action:
            beta = nn.functional.softplus(z)
        else:                                   # the old bounded-lambda head
            beta = (1.0 / 7.0) * torch.sigmoid(z)

        if self.mass_action or self.mod_clamp > 0:
            i_frac = st0[..., 2]
            if self.coupling == "explicit":
                i_frac = i_frac + torch.relu(self.alpha) * torch.matmul(i_frac, self.hat_A.T)
            lam = beta * i_frac.unsqueeze(-1)
        else:
            lam = beta
            if self.coupling == "explicit":
                imp = torch.matmul(st0[..., 2], self.hat_A.T).unsqueeze(-1)
                lam = lam + torch.relu(self.alpha) * imp

        omega, gamma = self.rates()
        _, inc = seir_sim.simulate_weeks(st0, lam, omega, gamma, substeps=7)
        return inc

    def to_cases(self, inc: torch.Tensor, pop: torch.Tensor) -> torch.Tensor:
        y = inc * RHO * pop
        if self.learn_rho:
            y = y * torch.exp(self.log_rho_adj.clamp(-2.0, 2.0))
        return y


# ------------------------------------------------------------------- data --

def features_train_only(data, level: str, fold) -> np.ndarray:
    """Inputs scaled with statistics from this fold's training weeks only."""
    end = int(np.asarray(fold.train_index).max()) + 1
    T, N = data.cases.shape
    cases_norm = (np.log1p(np.nan_to_num(data.cases, nan=0.0)) - fold.mean) / fold.std
    if level == "cases":
        f = np.zeros((T, N, WINDOW, 1), dtype=np.float32)
        for i in range(WINDOW, T):
            f[i, :, :, 0] = cases_norm[i - WINDOW: i].T
        return f
    cm = np.nanmean(data.climate[:end], axis=(0, 1), keepdims=True)
    cs = np.nanstd(data.climate[:end], axis=(0, 1), keepdims=True) + 1e-8
    cn = (data.climate - cm) / cs
    if level == "cases+era5":
        f = np.zeros((T, N, WINDOW, 7), dtype=np.float32)
        for i in range(4, T):
            f[i, :, :, 0] = cases_norm[i - WINDOW: i].T
            f[i, :, :, 1:7] = np.moveaxis(cn[i - 4: i - 1], 0, 1)
        return f
    nm, ns = np.nanmean(data.ndvi[:end]), np.nanstd(data.ndvi[:end]) + 1e-8
    nd = (data.ndvi - nm) / ns
    f = np.zeros((T, N, WINDOW, 8), dtype=np.float32)
    for i in range(4, T):
        f[i, :, :, 0] = cases_norm[i - WINDOW: i].T
        f[i, :, :, 1:7] = np.moveaxis(cn[i - 4: i - 1], 0, 1)
        f[i, :, :, 7] = nd[i - 2: i + 1].T
    return f


def initial_state(cases, population, idx, stock: bool, depletion: float):
    """(state, S, onset_I, onset_E, population, truth) at the forecast origin."""
    y = np.stack([cases[i: i + HORIZON].T for i in idx])
    c1 = np.stack([cases[i - 1] for i in idx])
    c2 = np.stack([cases[i - 2] for i in idx])
    pop = np.stack([population[i - 1] for i in idx])
    onset1, onset2 = c1 / (RHO * pop), c2 / (RHO * pop)
    if stock:
        onset1 = onset1 / (1.0 - math.exp(-GAMMA * 7))
        onset2 = onset2 / (1.0 - math.exp(-OMEGA * 7))
    i0 = np.clip(onset1, 1e-6, 0.5)
    e0 = np.clip(onset2, 1e-6, 0.5)
    cum = np.stack([np.nansum(cases[:i], axis=0) for i in idx])
    s = np.clip(S0 - depletion * cum / (RHO * pop), 0.01, 1.0)
    r = np.clip(1.0 - s - e0 - i0, 0.0, 1.0)
    t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return (t(np.stack([s, e0, i0, r], -1)), t(s), t(onset1), t(onset2),
            t(pop).unsqueeze(-1), t(y))


def build_folds(cases, missing, protocol: str):
    if protocol == "3origin":
        return rcb.build_folds_masked(cases, missing)
    bad = {int(m) for m in missing}
    ids = list(range(WINDOW, cases.shape[0] - HORIZON))
    clean = lambda i: not any(t in bad for t in range(i - WINDOW, i + HORIZON))  # noqa: E731
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
                           to(np.stack([z[i: i + HORIZON].T for i in ch])),
                           to(np.stack([np.repeat(z[i - 1][:, None], HORIZON, 1)
                                        for i in ch])))
        folds.append(base.Fold(origin, *pack(tr), *pack(va), *pack(te), mean, std,
                               np.asarray(te), np.asarray(tr), np.asarray(va)))
    return folds


# -------------------------------------------------------------------- run --

_PHYS = dict(stock=True, depletion=0.0, mass_action=True, per_week=True,
             learn_rho=True, learn_rates=True, learn_state=True, loss="mse",
             norm="train_only", epochs=400)

ARMS = {
    # the reference: what run_s5_seir_gnn.py does today
    "baseline": dict(stock=False, depletion=1.0, mass_action=False, per_week=False,
                     learn_rho=False, learn_rates=False, learn_state=False,
                     loss="smape", norm="orig", epochs=100, mod_clamp=0.0),
    # RECOMMENDED: all physics corrections, GNN kept but constrained to
    # modulating the learned baseline beta by at most e^(+-0.2)
    "v2": dict(_PHYS, mod_clamp=0.2),
    # the same physics with the encoder frozen out: a calibrated mechanistic
    # SEIR. Reported because it matches v2 -- the GNN is not shown to add value.
    "v2_mech": dict(_PHYS, mod_clamp=0.0, const=True),
    # the GNN free to set beta outright (no modulation bound)
    "v2_free": dict(_PHYS, mod_clamp=0.0),
}


def loss_of(name, pred, y):
    if name == "smape":
        return torch.mean(torch.abs(pred - y) / ((pred.abs() + y.abs() + 1e-5) / 2))
    if name == "mse":
        return torch.mean((pred - y) ** 2)
    raise ValueError(name)


def run_job(job: dict) -> dict:
    cfg, seed = job["cfg"], job["seed"]
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    feat_t = torch.tensor(job["features"], dtype=torch.float32)
    model = SEIRGNNv2(job["arch"], job["features"].shape[-1],
                      torch.tensor(job["edge_index"], dtype=torch.long),
                      torch.tensor(job["adj_dense"], dtype=torch.float32),
                      coupling=job["coupling"], mass_action=cfg["mass_action"],
                      per_week=cfg["per_week"], learn_rho=cfg["learn_rho"],
                      learn_rates=cfg["learn_rates"], mod_clamp=cfg["mod_clamp"],
                      learn_state=cfg["learn_state"])
    if cfg.get("const"):
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        model.head_fc.weight.data.zero_()
        model.head_fc.weight.requires_grad_(False)
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad],
                     lr=0.003, weight_decay=1e-4)

    def prep(idx):
        st0, s_, o_i, o_e, pop, y = initial_state(job["cases"], job["population"], idx,
                                                  cfg["stock"], cfg["depletion"])
        return feat_t[idx], st0, pop, y, s_, o_i, o_e

    tr, va, te = prep(job["train_idx"]), prep(job["val_idx"]), prep(job["test_idx"])

    def fwd(b):
        st = model.build_state(b[4], b[5], b[6]) if cfg["learn_state"] else b[1]
        return model.to_cases(model(b[0], st), b[2])

    best, best_w, waited = float("inf"), None, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        opt.zero_grad()
        loss_of(cfg["loss"], fwd(tr), tr[3]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if epoch % 3 == 0:
            model.eval()
            with torch.no_grad():
                v = torch.sqrt(torch.mean((fwd(va) - va[3]) ** 2)).item()
            if v < best - 1e-5:
                best, waited = v, 0
                best_w = {k: t.cpu().clone() for k, t in model.state_dict().items()}
            else:
                waited += 1
                if waited >= 20:
                    break
    if best_w:
        model.load_state_dict(best_w)
    model.eval()
    with torch.no_grad():
        vp, tp = fwd(va), fwd(te)
        omega, gamma = model.rates()
        rec = {
            "arm": job["arm"], "arch": job["arch"], "origin": job["origin"], "seed": seed,
            "val_RMSE": torch.sqrt(torch.mean((vp - va[3]) ** 2)).item(),
            "val_MAE": torch.mean((vp - va[3]).abs()).item(),
            "test_RMSE": torch.sqrt(torch.mean((tp - te[3]) ** 2)).item(),
            "test_MAE": torch.mean((tp - te[3]).abs()).item(),
            "pred_mean": tp.mean().item(), "truth_mean": te[3].mean().item(),
            "omega_per_day": float(omega), "gamma_per_day": float(gamma),
            "rho_mult": float(torch.exp(model.log_rho_adj.clamp(-2, 2)))
            if model.learn_rho else 1.0,
            "e_scale": float(torch.exp(model.log_e_scale.clamp(-1, 3)))
            if model.learn_state else 1.0,
            "i_scale": float(torch.exp(model.log_i_scale.clamp(-1, 3)))
            if model.learn_state else 1.0,
            "epochs_used": epoch + 1, "elapsed": round(time.time() - t0, 1),
        }
    print(f"  [OK] {job['arm']:10s} o{job['origin']} s{seed} | val "
          f"{rec['val_RMSE']:6.2f} | test {rec['test_RMSE']:6.2f} | MAE "
          f"{rec['test_MAE']:6.2f} ({rec['elapsed']:.0f}s)", flush=True)
    return rec


def persistence(cases, folds):
    out = {}
    for f in folds:
        idx = np.asarray(f.test_index)
        p = np.stack([np.repeat(cases[i - 1][:, None], HORIZON, axis=1) for i in idx])
        t = np.stack([cases[i: i + HORIZON].T for i in idx])
        out[f.origin] = {"test_RMSE": float(np.sqrt(np.mean((p - t) ** 2))),
                         "test_MAE": float(np.mean(np.abs(p - t)))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs="*", default=["v2"], choices=sorted(ARMS))
    ap.add_argument("--arch", default="STGAT")
    ap.add_argument("--inputs", default="cases",
                    choices=["cases", "cases+era5", "cases+era5+ndvi"])
    ap.add_argument("--coupling", default="implicit", choices=["implicit", "explicit"])
    ap.add_argument("--protocol", default="3origin", choices=["3origin", "9origin"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="s5_v2_results.json")
    args = ap.parse_args()

    data = cd.load()
    cases, adjacency, artifact, missing, _ = rcb.prepare("rebuilt")
    folds = build_folds(cases, missing, args.protocol)
    src, dst = np.nonzero(adjacency)
    edge_index = np.stack([src, dst])

    import run_s5_seir_gnn as rs5
    jobs = []
    for arm in args.arm:
        cfg = ARMS[arm]
        for fold in folds:
            feats = (rs5.prepare_inputs_by_level(data, args.inputs)
                     if cfg["norm"] == "orig"
                     else features_train_only(data, args.inputs, fold))
            for seed in args.seeds:
                jobs.append({"arm": arm, "cfg": cfg, "arch": args.arch, "seed": seed,
                             "coupling": args.coupling, "origin": fold.origin,
                             "train_idx": np.asarray(fold.train_index),
                             "val_idx": np.asarray(fold.val_index),
                             "test_idx": np.asarray(fold.test_index),
                             "cases": cases, "features": feats,
                             "population": data.population,
                             "edge_index": edge_index, "adj_dense": adjacency})

    print(f"{len(jobs)} jobs / {args.workers} workers | arms: {', '.join(args.arm)} "
          f"| protocol: {args.protocol}", flush=True)
    t0, out = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run_job, j) for j in jobs]):
            try:
                out.append(f.result())
            except Exception as exc:
                print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    df = pd.DataFrame(out)
    print("\n=== mean over origins x seeds ===")
    print(df.groupby("arm")[["val_RMSE", "test_RMSE", "test_MAE"]].mean().round(2)
            .sort_values("val_RMSE").to_string())
    print("\n=== test RMSE per origin ===")
    piv = df.groupby(["arm", "origin"])["test_RMSE"].mean().round(2).unstack("origin")
    ref = persistence(cases, folds)
    piv.loc["persistence"] = [round(ref[o]["test_RMSE"], 2) for o in piv.columns]
    print(piv.to_string())
    print(f"\nwrote {args.out} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
