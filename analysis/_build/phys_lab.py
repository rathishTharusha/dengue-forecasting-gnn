"""Experiment lab for the physics-informed SEIR-GNN arm.

Every variant keeps the SEIR simulator in the prediction path. The protocol is
the project's frozen one: dataset `rebuilt`, origins 0.55/0.70/0.85, seeds
0/1/2, window 3 -> horizon 3, early stopping on validation RMSE, metrics pooled
over (window, district, horizon) in raw case counts.

Only the knobs listed in CONFIG_KEYS change between variants. `baseline` is
exactly what `analysis/_build/run_s5_seir_gnn.py` does today.

Usage:
    python phys_lab.py --variant baseline state_flow ...     # run named variants
    python phys_lab.py --list                                # show what exists
    python phys_lab.py --report                              # leaderboard so far
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
RESULTS = HERE / "phys_lab_runs.jsonl"
for p in (REPO / "analysis" / "lib", REPO / "analysis" / "_build", REPO / "src"):
    sys.path.insert(0, str(p))

import corrected_data as cd  # noqa: E402
import reproduced as arch_lib  # noqa: E402
import run_corrected_benchmark as rcb  # noqa: E402
import run_s5_seir_gnn as rs5  # noqa: E402
import seir_sim  # noqa: E402

# ---------------------------------------------------------------- defaults --

BASE = {
    "arch": "STGAT",
    "inputs": "cases",          # cases | cases+era5 | cases+era5+ndvi
    "norm": "orig",             # orig (his) | train_only
    "proj": "linear1",          # linear1 (his) | mlp
    "head": "sigmoid",          # sigmoid (his) | exp | mass
    "horizon_lam": "repeat",    # repeat (his) | per_week
    "coupling": "implicit",     # implicit | explicit
    "sim": "open",              # open (his, lambda fixed in week) | closed (feedback)
    "state": "orig",            # orig (his) | flow | residence | assim
    "loss": "smape",            # smape (his) | mse | poisson | mse_log | huber_log
    "omega": 0.7 / 7.0,
    "gamma": 1.0 / 7.0,
    "rho": 1.0 / 11.0,
    "s0": 1.0 - 0.682,
    "lambda_max": 1.0 / 7.0,
    "depletion": 1.0,           # how much of cumulative infection removes S
    "learn_rho": False,         # learn the reporting rate (observation model)
    "learn_rates": False,       # learn omega and gamma, constrained positive
    "beta_district": False,     # one baseline beta per district
    "seasonal": False,          # harmonic seasonal term on beta
    "mod_clamp": 2.0,           # how far the network may modulate log beta
    "learn_state": False,       # learn the E and I stock scale factors
    "blend": "none",            # none | learned : w*persistence + (1-w)*physics
    "epochs": 100,
    "lr": 0.003,
    "hidden": 0,                # >0 widens the input projection (proj=mlp)
}

VARIANTS: dict[str, dict] = {
    # --- reference ---------------------------------------------------------
    "baseline": {},
    # --- A: training objective --------------------------------------------
    "loss_mse": {"loss": "mse"},
    "loss_poisson": {"loss": "poisson"},
    "loss_mse_log": {"loss": "mse_log"},
    # --- B: SEIR state initialisation -------------------------------------
    "state_flow": {"state": "flow"},
    "state_residence": {"state": "residence"},
    # --- best-so-far stack (set as experiments confirm) --------------------
    "best_A_B": {"state": "flow", "loss": "mse"},
    # --- C: time-varying lambda -------------------------------------------
    "lam_per_week": {"state": "flow", "loss": "mse", "horizon_lam": "per_week"},
    # --- D: parameterisation ----------------------------------------------
    "head_exp": {"state": "flow", "loss": "mse", "head": "exp"},
    "head_mass": {"state": "flow", "loss": "mse", "head": "mass"},
    "sim_closed": {"state": "flow", "loss": "mse", "head": "mass", "sim": "closed"},
    "rates_alt": {"state": "flow", "loss": "mse", "omega": (7.0 / 5.9) / 7.0,
                  "gamma": (7.0 / 4.5) / 7.0},
    # --- E: spatial coupling ----------------------------------------------
    "coup_explicit": {"state": "flow", "loss": "mse", "coupling": "explicit"},
    # --- F: bottleneck ------------------------------------------------------
    "proj_mlp": {"state": "flow", "loss": "mse", "proj": "mlp", "hidden": 16},
    # --- H: inputs / normalisation -----------------------------------------
    "norm_train_only": {"state": "flow", "loss": "mse", "norm": "train_only"},
    "inputs_era5": {"state": "flow", "loss": "mse", "inputs": "cases+era5",
                    "norm": "train_only"},
    "inputs_ndvi": {"state": "flow", "loss": "mse", "inputs": "cases+era5+ndvi",
                    "norm": "train_only"},
    # --- round 3: stacked on the mass-action head -------------------------
    "mass_per_week": {"state": "flow", "loss": "mse", "head": "mass",
                      "horizon_lam": "per_week"},
    "mass_coup": {"state": "flow", "loss": "mse", "head": "mass",
                  "coupling": "explicit"},
    "mass_norm": {"state": "flow", "loss": "mse", "head": "mass", "norm": "train_only"},
    "mass_proj": {"state": "flow", "loss": "mse", "head": "mass", "proj": "mlp",
                  "hidden": 16},
    "mass_rates": {"state": "flow", "loss": "mse", "head": "mass",
                   "omega": (7.0 / 5.9) / 7.0, "gamma": (7.0 / 4.5) / 7.0},
    "mass_smape": {"state": "flow", "head": "mass"},
    "mass_poisson": {"state": "flow", "loss": "poisson", "head": "mass"},
    # --- round 4: what sets the SIZE of the forecast ----------------------
    "mass_s_nodep": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0},
    "mass_s_partial": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.25},
    "mass_s0_high": {"state": "flow", "loss": "mse", "head": "mass", "s0": 1.0 - 0.514},
    "mass_s0_low": {"state": "flow", "loss": "mse", "head": "mass", "s0": 1.0 - 0.908},
    "mass_rho25": {"state": "flow", "loss": "mse", "head": "mass", "rho": 1.0 / 2.5},
    "mass_rho30": {"state": "flow", "loss": "mse", "head": "mass", "rho": 1.0 / 30.0},
    "mass_learn_rho": {"state": "flow", "loss": "mse", "head": "mass", "learn_rho": True},
    "mass_learn_rates": {"state": "flow", "loss": "mse", "head": "mass", "learn_rates": True},
    "mass_era5": {"state": "flow", "loss": "mse", "head": "mass", "inputs": "cases+era5",
                  "norm": "train_only"},
    "mass_era5_mlp": {"state": "flow", "loss": "mse", "head": "mass",
                      "inputs": "cases+era5", "norm": "train_only", "proj": "mlp",
                      "hidden": 16},
    # --- round 5: stacked on no-depletion mass action ---------------------
    "b2": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0},
    "b2_per_week": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "horizon_lam": "per_week"},
    "b2_coup": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "coupling": "explicit"},
    "b2_era5": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "inputs": "cases+era5", "norm": "train_only"},
    "b2_ndvi": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "inputs": "cases+era5+ndvi", "norm": "train_only"},
    "b2_learn_rho": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "learn_rho": True},
    "b2_learn_rates": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "learn_rates": True},
    "b2_closed": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "sim": "closed"},
    "b2_s0_high": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "s0": 1.0 - 0.514},
    "b2_s0_low": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "s0": 1.0 - 0.908},
    "b2_dep10": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.10},
    "b2_dep50": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.50},
    "b2_smape": {"state": "flow", "head": "mass", "depletion": 0.0},
    "b2_poisson": {"state": "flow", "loss": "poisson", "head": "mass", "depletion": 0.0},
    # --- round 6: combine the confirmed wins ------------------------------
    "b3": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "learn_rho": True},
    "b3_rates": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "learn_rho": True, "learn_rates": True},
    "b3_per_week": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "learn_rho": True, "horizon_lam": "per_week"},
    "b3_era5": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "learn_rho": True, "inputs": "cases+era5", "norm": "train_only"},
    "b3_rates_pw": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week"},
    "b3_rates_era5": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "inputs": "cases+era5",
                       "norm": "train_only"},
    "b3_all": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week",
                "inputs": "cases+era5", "norm": "train_only"},
    # --- round 7: is the network doing anything? --------------------------
    "b2_constbeta": {"state": "flow", "loss": "mse", "head": "const",
                      "depletion": 0.0, "learn_rho": True},
    "b2_constbeta_pw": {"state": "flow", "loss": "mse", "head": "const",
                         "depletion": 0.0, "learn_rho": True,
                         "horizon_lam": "per_week"},
    "b3_norm": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "learn_rho": True, "norm": "train_only"},
    "b3_rates_pw_norm": {"state": "flow", "loss": "mse", "head": "mass", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week",
                          "norm": "train_only"},
    # --- round 8: give the network a well-posed job -----------------------
    "mod": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "head": "mod"},
    "mod_district": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "head": "mod", "beta_district": True},
    "mod_season": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "head": "mod", "seasonal": True},
    "mod_district_season": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "head": "mod", "beta_district": True, "seasonal": True},
    "const_district_season": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "head": "const", "seasonal": True},
    "mod_norm": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "head": "mod", "norm": "train_only"},
    "mod_tight": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "head": "mod", "mod_clamp": 0.5},
    # --- round 9: same budget for everyone, 4x longer ---------------------
    "e400_const": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "head": "const", "epochs": 400},
    "e400_mod": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "head": "mod", "epochs": 400},
    "e400_mod_norm": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "head": "mod", "norm": "train_only", "epochs": 400},
    "e400_mass_pw": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "head": "mass", "epochs": 400},
    # --- round 10: push the budget further --------------------------------
    "e800_mass_pw": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "head": "mass", "epochs": 800},
    "e800_const": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "head": "const", "epochs": 800},
    "e800_mass_pw_norm": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "head": "mass", "epochs": 800, "norm": "train_only"},
    "e800_mass_pw_era5": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "head": "mass", "epochs": 800, "inputs": "cases+era5",
                           "norm": "train_only"},
    # --- round 11: keep train-only scaling, regularise the network --------
    "n400_mass": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "norm": "train_only", "epochs": 400, "head": "mass"},
    "n400_mod05": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "norm": "train_only", "epochs": 400, "head": "mod", "mod_clamp": 0.5},
    "n400_mod10": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "norm": "train_only", "epochs": 400, "head": "mod", "mod_clamp": 1.0},
    "n400_mod02": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "norm": "train_only", "epochs": 400, "head": "mod", "mod_clamp": 0.2},
    "n400_const": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "norm": "train_only", "epochs": 400, "head": "const"},
    "n400_mod05_era5": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "norm": "train_only", "epochs": 400, "head": "mod", "mod_clamp": 0.5, "inputs": "cases+era5"},
    # --- round 12: learn the state scaling instead of assuming it ---------
    "ls_const": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "norm": "train_only", "epochs": 400, "head": "const", "learn_state": True},
    "ls_mass": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "norm": "train_only", "epochs": 400, "head": "mass", "learn_state": True},
    "ls_mod02": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "horizon_lam": "per_week", "norm": "train_only", "epochs": 400, "head": "mod", "mod_clamp": 0.2, "learn_state": True},
    # --- round 13: no-physics control under the v2 recipe -----------------
    "v2_direct": {"loss": "mse", "norm": "train_only", "epochs": 400, "head": "direct"},
    "v2_direct_orig_norm": {"loss": "mse", "epochs": 400, "head": "direct"},
    "his_direct": {"head": "direct"},
    # --- round 14: blend the physics forecast with persistence ------------
    "blend_const": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "learn_state": True, "horizon_lam": "per_week", "norm": "train_only", "epochs": 400, "head": "const", "blend": "learned"},
    "blend_mass": {"state": "flow", "loss": "mse", "depletion": 0.0, "learn_rho": True, "learn_rates": True, "learn_state": True, "horizon_lam": "per_week", "norm": "train_only", "epochs": 400, "head": "mass", "blend": "learned"},
    # --- training budget (applied to baseline too, for fairness) -----------
    "epochs400_baseline": {"epochs": 400},
    "epochs400_best": {"state": "flow", "loss": "mse", "epochs": 400},
}


# ------------------------------------------------------------------ model --

class PhysGNN(nn.Module):
    """His SEIR-GNN with the knobs this lab varies."""

    def __init__(self, cfg: dict, in_dim: int, edge_index: torch.Tensor,
                 adj_dense: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.edge_index = edge_index
        n_out = 3 if cfg["horizon_lam"] == "per_week" else 1

        if cfg["proj"] == "mlp":
            h = cfg["hidden"] or 16
            self.in_proj = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, 1))
        else:
            self.in_proj = nn.Linear(in_dim, 1)

        adj = adj_dense.clone()
        adj.fill_diagonal_(0.0)
        self.register_buffer("hat_A", adj / adj.sum(1, keepdim=True).clamp_min(1.0))

        kwargs = {"adaptive": False, "channels": 8} if cfg["arch"] == "AAGCN" else {}
        self.backbone = arch_lib.build(cfg["arch"], 25, 3, 3, edge_index=edge_index, **kwargs)

        self.head_fc = nn.Linear(3, n_out)
        if cfg["head"] == "const":
            self.const_beta = nn.Parameter(torch.full((n_out,), -0.593))
        if cfg["head"] == "mod":
            n_beta = 25 if cfg["beta_district"] else 1
            self.beta0 = nn.Parameter(torch.full((n_beta,), -0.593))
            nn.init.zeros_(self.head_fc.bias)
            nn.init.normal_(self.head_fc.weight, std=0.05)
        if cfg["seasonal"]:
            self.season = nn.Parameter(torch.zeros(2))
        if cfg["learn_state"]:
            # start exactly at the flow-derived factors, then let the data move them
            e_f = 1.0 / (1.0 - math.exp(-cfg["omega"] * 7))
            i_f = 1.0 / (1.0 - math.exp(-cfg["gamma"] * 7))
            self.log_e_scale = nn.Parameter(torch.tensor(math.log(e_f)))
            self.log_i_scale = nn.Parameter(torch.tensor(math.log(i_f)))
        if cfg["head"] == "sigmoid":
            self.act = nn.Sigmoid()
        elif cfg["head"] == "exp":
            nn.init.zeros_(self.head_fc.bias)
            nn.init.normal_(self.head_fc.weight, std=0.1)
            self.log_ref = math.log(2.5e-4)
        else:  # mass action: beta, with lambda = beta * (I + alpha * A_hat I)
            nn.init.constant_(self.head_fc.bias, -0.593)   # softplus(-0.593) ~ 0.44 /day
            nn.init.normal_(self.head_fc.weight, std=0.1)
        if cfg["coupling"] == "explicit" or cfg["head"] == "mass":
            self.alpha = nn.Parameter(torch.tensor(0.05))
        if cfg["learn_rho"]:
            self.log_rho_adj = nn.Parameter(torch.zeros(()))
        if cfg["blend"] == "learned":
            # starts at w = 0.5, learned on TRAINING data only
            self.blend_w = nn.Parameter(torch.zeros(()))
        if cfg["learn_rates"]:
            # softplus(raw) = rate; start exactly at the configured values
            self.raw_omega = nn.Parameter(torch.tensor(math.log(math.expm1(cfg["omega"]))))
            self.raw_gamma = nn.Parameter(torch.tensor(math.log(math.expm1(cfg["gamma"]))))

    def _season(self, beta, phase):
        """beta * exp(a sin(t) + b cos(t)) -- a smooth yearly cycle."""
        if not self.cfg["seasonal"] or phase is None:
            return beta
        harm = (self.season[0] * torch.sin(phase) + self.season[1] * torch.cos(phase))
        return beta * torch.exp(harm.clamp(-2.0, 2.0)).unsqueeze(1)

    def direct(self, x: torch.Tensor) -> torch.Tensor:
        """No physics: the encoder predicts normalised log1p cases itself."""
        return self.backbone(self.in_proj(x).squeeze(-1), self.edge_index)

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder output -> (B, N, n_out) pre-activation."""
        h = self.backbone(self.in_proj(x).squeeze(-1), self.edge_index)
        return self.head_fc(h)

    def lam_or_beta(self, x, st0, phase=None):
        """Returns (B, N, 3): lambda per forecast week, or beta for mass action."""
        if self.cfg["head"] == "mod":
            # a learned baseline transmission rate, modulated by the network
            base = nn.functional.softplus(self.beta0)
            base = base.view(1, -1, 1) if base.numel() > 1 else base.view(1, 1, 1)
            z = self.raw(x)
            if z.shape[-1] == 1:
                z = z.repeat(1, 1, 3)
            out = base * torch.exp(z.clamp(-self.cfg["mod_clamp"], self.cfg["mod_clamp"]))
            return self._season(out, phase)
        if self.cfg["head"] == "const":
            # ablation: no network at all, one learned beta (mass action)
            b = nn.functional.softplus(self.const_beta)
            if b.shape[-1] == 1:
                b = b.repeat(3)
            return self._season(b.view(1, 1, 3).expand(x.shape[0], x.shape[1], 3), phase)
        z = self.raw(x)
        if z.shape[-1] == 1:
            z = z.repeat(1, 1, 3)
        cfg = self.cfg
        if cfg["head"] == "sigmoid":
            out = cfg["lambda_max"] * torch.sigmoid(z)
        elif cfg["head"] == "exp":
            out = torch.exp(self.log_ref + z.clamp(-6.0, 6.0))
        else:
            return nn.functional.softplus(z)          # beta, per day
        if cfg["coupling"] == "explicit":
            imp = torch.matmul(st0[..., 2], self.hat_A.T).unsqueeze(-1)
            out = out + torch.relu(self.alpha) * imp
        return out


# ------------------------------------------------------------------- data --

def build_features(data, level: str, norm: str, fold):
    if norm == "orig":
        return rs5.prepare_inputs_by_level(data, level)
    end = int(np.asarray(fold.train_index).max()) + 1
    T, N = data.cases.shape
    cases_norm = (np.log1p(np.nan_to_num(data.cases, nan=0.0)) - fold.mean) / fold.std
    if level == "cases":
        f = np.zeros((T, N, 3, 1), dtype=np.float32)
        for i in range(3, T):
            f[i, :, :, 0] = cases_norm[i - 3: i].T
        return f
    cm = np.nanmean(data.climate[:end], axis=(0, 1), keepdims=True)
    cs = np.nanstd(data.climate[:end], axis=(0, 1), keepdims=True) + 1e-8
    cn = (data.climate - cm) / cs
    if level == "cases+era5":
        f = np.zeros((T, N, 3, 7), dtype=np.float32)
        for i in range(4, T):
            f[i, :, :, 0] = cases_norm[i - 3: i].T
            f[i, :, :, 1:7] = np.moveaxis(cn[i - 4: i - 1], 0, 1)
        return f
    nm, ns = np.nanmean(data.ndvi[:end]), np.nanstd(data.ndvi[:end]) + 1e-8
    nn_ = (data.ndvi - nm) / ns
    f = np.zeros((T, N, 3, 8), dtype=np.float32)
    for i in range(4, T):
        f[i, :, :, 0] = cases_norm[i - 3: i].T
        f[i, :, :, 1:7] = np.moveaxis(cn[i - 4: i - 1], 0, 1)
        f[i, :, :, 7] = nn_[i - 2: i + 1].T
    return f


def make_batch(cfg, cases, population, feat_t, idx, phase_all=None):
    """Initial compartments as fractions, plus inputs, population and targets."""
    rho, s0v = cfg["rho"], cfg["s0"]
    omega, gamma = cfg["omega"], cfg["gamma"]
    y = np.stack([cases[i: i + 3].T for i in idx])
    c1 = np.stack([cases[i - 1] for i in idx])
    c2 = np.stack([cases[i - 2] for i in idx])
    pop = np.stack([population[i - 1] for i in idx])

    onset1 = c1 / (rho * pop)          # weekly incidence as a fraction
    onset2 = c2 / (rho * pop)
    if cfg["state"] == "orig":
        i_f, e_f = 1.0, 1.0
    elif cfg["state"] == "flow":       # weekly conversion fraction
        i_f = 1.0 / (1.0 - math.exp(-gamma * 7))
        e_f = 1.0 / (1.0 - math.exp(-omega * 7))
    else:                              # residence time (continuous steady state)
        i_f = 1.0 / (gamma * 7)
        e_f = 1.0 / (omega * 7)
    i0 = np.clip(onset1 * i_f, 1e-6, 0.5)
    e0 = np.clip(onset2 * e_f, 1e-6, 0.5)
    cum = np.stack([np.nansum(cases[:i], axis=0) for i in idx])
    s = np.clip(s0v - cfg["depletion"] * cum / (rho * pop), 0.01, 1.0)
    r = np.clip(1.0 - s - e0 - i0, 0.0, 1.0)
    st0 = torch.tensor(np.stack([s, e0, i0, r], -1), dtype=torch.float32)
    persist = torch.tensor(np.repeat(c1[:, :, None], 3, axis=2), dtype=torch.float32)
    raw = (torch.tensor(s, dtype=torch.float32),
           torch.tensor(onset1, dtype=torch.float32),
           torch.tensor(onset2, dtype=torch.float32))
    ph = None
    if phase_all is not None:
        ph = torch.tensor(np.stack([phase_all[i: i + 3] for i in idx]), dtype=torch.float32)
    return (feat_t[idx], st0, torch.tensor(pop, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(y, dtype=torch.float32), ph, raw, persist)


# --------------------------------------------------------------- dynamics --

def simulate(cfg, st0, lam_or_beta):
    """Run the SEIR three weeks ahead. Returns weekly incidence (B, N, 3)."""
    omega, gamma = cfg["omega"], cfg["gamma"]
    if cfg["head"] not in ("mass", "const", "mod"):
        _, inc = seir_sim.simulate_weeks(st0, lam_or_beta, omega, gamma, substeps=7)
        return inc
    beta = lam_or_beta
    if cfg["sim"] == "open":           # lambda from the state at the origin only
        i_frac = st0[..., 2]
        if cfg["coupling"] == "explicit":
            i_frac = i_frac + torch.matmul(st0[..., 2], cfg["_hatA"].T) * cfg["_alpha"]
        lam = beta * i_frac.unsqueeze(-1)
        _, inc = seir_sim.simulate_weeks(st0, lam, omega, gamma, substeps=7)
        return inc
    # closed loop: recompute lambda = beta * I each substep (true mass action)
    dt = 7.0 / 7
    state, weekly = st0, []
    for w in range(3):
        total = torch.zeros(state.shape[:-1], dtype=state.dtype)
        for _ in range(7):
            prev = state[..., 2] / state.sum(-1).clamp_min(1e-8)
            if cfg["coupling"] == "explicit":
                prev = prev + cfg["_alpha"] * torch.matmul(prev, cfg["_hatA"].T)
            lam = beta[..., w] * prev
            state, onset = seir_sim._step(state, lam, omega, gamma, dt)
            total = total + onset
        weekly.append(total)
    return torch.stack(weekly, dim=-1)


def loss_of(name, pred, y):
    if name == "smape":
        return torch.mean(torch.abs(pred - y) / ((pred.abs() + y.abs() + 1e-5) / 2))
    if name == "mse":
        return torch.mean((pred - y) ** 2)
    if name == "mse_log":
        return torch.mean((torch.log1p(pred.clamp_min(0)) - torch.log1p(y)) ** 2)
    if name == "poisson":
        lam = pred.clamp_min(1e-6)
        return torch.mean(lam - y * torch.log(lam))
    if name == "huber_log":
        return nn.functional.smooth_l1_loss(torch.log1p(pred.clamp_min(0)), torch.log1p(y))
    raise ValueError(name)


# ------------------------------------------------------------------- run ---

def run_job(job: dict) -> dict:
    cfg, seed = job["cfg"], job["seed"]
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    feat_t = torch.tensor(job["features"], dtype=torch.float32)
    edge_index = torch.tensor(job["edge_index"], dtype=torch.long)
    adj = torch.tensor(job["adj_dense"], dtype=torch.float32)
    model = PhysGNN(cfg, job["features"].shape[-1], edge_index, adj)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    ph = job.get("phase")
    tr = make_batch(cfg, job["cases"], job["population"], feat_t, job["train_idx"], ph)
    va = make_batch(cfg, job["cases"], job["population"], feat_t, job["val_idx"], ph)
    te = make_batch(cfg, job["cases"], job["population"], feat_t, job["test_idx"], ph)

    run_cfg = dict(cfg)
    run_cfg["_hatA"] = model.hat_A
    lo, hi = 0.02, 0.45   # keep learned rates epidemiologically sane (per day)

    def fwd(b):
        if cfg["head"] == "direct":
            z = model.direct(b[0]) * job["std"] + job["mean"]
            return torch.expm1(torch.clamp(z, -1.0, 12.0))
        st = b[1]
        if cfg["learn_state"]:
            s_, o1, o2 = b[5]
            i_ = (o1 * torch.exp(model.log_i_scale.clamp(-1, 3))).clamp(1e-6, 0.5)
            e_ = (o2 * torch.exp(model.log_e_scale.clamp(-1, 3))).clamp(1e-6, 0.5)
            r_ = (1.0 - s_ - e_ - i_).clamp(0.0, 1.0)
            st = torch.stack([s_, e_, i_, r_], dim=-1)
        out = model.lam_or_beta(b[0], st, b[4])
        run_cfg["_alpha"] = torch.relu(model.alpha) if hasattr(model, "alpha") else 0.0
        if cfg["learn_rates"]:
            run_cfg["omega"] = nn.functional.softplus(model.raw_omega).clamp(lo, hi)
            run_cfg["gamma"] = nn.functional.softplus(model.raw_gamma).clamp(lo, hi)
        inc = simulate(run_cfg, st, out)
        y = inc * cfg["rho"] * b[2]
        if cfg["learn_rho"]:
            y = y * torch.exp(model.log_rho_adj.clamp(-2.0, 2.0))
        if cfg["blend"] == "learned":
            w = torch.sigmoid(model.blend_w)
            y = w * b[6] + (1.0 - w) * y
        return y

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
        rec = {
            "variant": job["variant"], "origin": job["origin"], "seed": seed,
            "val_RMSE": torch.sqrt(torch.mean((vp - va[3]) ** 2)).item(),
            "val_MAE": torch.mean((vp - va[3]).abs()).item(),
            "test_RMSE": torch.sqrt(torch.mean((tp - te[3]) ** 2)).item(),
            "test_MAE": torch.mean((tp - te[3]).abs()).item(),
            "pred_mean": tp.mean().item(), "truth_mean": te[3].mean().item(),
            "epochs_used": epoch + 1, "elapsed": round(time.time() - t0, 1),
        }
        if cfg["learn_rho"]:
            rec["learned_rho_mult"] = float(torch.exp(model.log_rho_adj.clamp(-2, 2)))
        if cfg["blend"] == "learned":
            rec["blend_w_persistence"] = float(torch.sigmoid(model.blend_w))
        if cfg["learn_state"]:
            rec["learned_e_scale"] = float(torch.exp(model.log_e_scale.clamp(-1, 3)))
            rec["learned_i_scale"] = float(torch.exp(model.log_i_scale.clamp(-1, 3)))
        if cfg["learn_rates"]:
            rec["learned_omega"] = float(nn.functional.softplus(model.raw_omega).clamp(lo, hi))
            rec["learned_gamma"] = float(nn.functional.softplus(model.raw_gamma).clamp(lo, hi))
    rec.update({f"cfg_{k}": v for k, v in cfg.items() if not k.startswith("_")})
    print(f"  [OK] {job['variant']:18s} o{job['origin']} s{seed} | val "
          f"{rec['val_RMSE']:6.2f} | test {rec['test_RMSE']:6.2f} | MAE "
          f"{rec['test_MAE']:6.2f} | pred {rec['pred_mean']:5.1f}/{rec['truth_mean']:5.1f} "
          f"({rec['elapsed']:.0f}s)", flush=True)
    return rec


def build_folds_9(cases, missing):
    """The plan's confirmatory protocol: 9 disjoint origins, 0.50..0.90.

    Same construction as run_corrected_benchmark.build_folds_masked, with the
    origin list and test fraction from run_beat_baseline's defaults.
    """
    import torch as _t

    import adaptive as base
    W, H, VAL = 3, 3, 30
    bad = {int(m) for m in missing}
    ids = list(range(W, cases.shape[0] - H))
    clean = lambda i: not any(t in bad for t in range(i - W, i + H))  # noqa: E731
    to = lambda a: _t.tensor(a, dtype=_t.float32)  # noqa: E731
    origins = [round(float(x), 6) for x in np.linspace(0.50, 0.90, 9)]
    test_frac = 0.05
    folds = []
    for origin in origins:
        cut = int(origin * len(ids))
        end = int(min(origin + test_frac, 1.0) * len(ids))
        train_ids = [i for i in ids[: cut - VAL] if clean(i)]
        val_ids = [i for i in ids[cut - VAL: cut] if clean(i)]
        test_ids = [i for i in ids[cut:end] if clean(i)]
        history = np.log1p(cases[: ids[: cut - VAL][-1] + 1])
        mean, std = float(np.nanmean(history)), float(np.nanstd(history) + 1e-8)
        z = (np.log1p(cases) - mean) / std

        def pack(chosen, scaled=z):
            x = np.stack([scaled[i - W: i].T for i in chosen])
            y = np.stack([scaled[i: i + H].T for i in chosen])
            p_ = np.stack([np.repeat(scaled[i - 1][:, None], H, axis=1) for i in chosen])
            return to(x), to(y), to(p_)

        folds.append(base.Fold(origin, *pack(train_ids), *pack(val_ids), *pack(test_ids),
                               mean, std, np.asarray(test_ids), np.asarray(train_ids),
                               np.asarray(val_ids)))
    return folds


def persistence_reference(cases, folds):
    out = []
    for f in folds:
        idx = np.asarray(f.test_index)
        pred = np.stack([np.repeat(cases[i - 1][:, None], 3, axis=1) for i in idx])
        truth = np.stack([cases[i: i + 3].T for i in idx])
        out.append({"origin": f.origin,
                    "test_RMSE": float(np.sqrt(np.mean((pred - truth) ** 2))),
                    "test_MAE": float(np.mean(np.abs(pred - truth)))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--protocol", choices=["3origin", "9origin"], default="3origin")
    args = ap.parse_args()

    if args.list:
        for k, v in VARIANTS.items():
            print(f"{k:22s} {v}")
        return 0
    if args.report:
        return report()

    data = cd.load()
    cases, adjacency, artifact, missing, folds = rcb.prepare("rebuilt")
    if args.protocol == "9origin":
        folds = build_folds_9(cases, missing)
    src, dst = np.nonzero(adjacency)
    edge_index = np.stack([src, dst])

    doy = pd.to_datetime(pd.Series(data.week_start).reset_index(drop=True)).dt.dayofyear.to_numpy()
    phase_all = (2.0 * np.pi * doy / 365.25).astype(np.float32)

    jobs = []
    for name in args.variant:
        cfg = dict(BASE)
        cfg.update(VARIANTS[name])
        for fold in folds:
            feats = build_features(data, cfg["inputs"], cfg["norm"], fold)
            for seed in args.seeds:
                jobs.append({"variant": name + ("@9" if args.protocol == "9origin" else ""),
                             "cfg": cfg, "seed": seed,
                             "origin": fold.origin,
                             "train_idx": np.asarray(fold.train_index),
                             "val_idx": np.asarray(fold.val_index),
                             "test_idx": np.asarray(fold.test_index),
                             "cases": cases, "features": feats,
                             "population": data.population,
                             "edge_index": edge_index, "adj_dense": adjacency,
                             "phase": phase_all, "mean": fold.mean, "std": fold.std})

    print(f"{len(jobs)} jobs / {args.workers} workers : {', '.join(args.variant)}", flush=True)
    t0 = time.time()
    done = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run_job, j) for j in jobs]):
            try:
                done.append(f.result())
            except Exception as exc:
                import traceback
                print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
    with RESULTS.open("a", encoding="utf-8") as fh:
        for r in done:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(done)} rows in {(time.time() - t0) / 60:.1f} min")

    ref = persistence_reference(cases, folds)
    print("persistence:", {r["origin"]: round(r["test_RMSE"], 2) for r in ref})
    (HERE / f"persistence_{args.protocol}.json").write_text(json.dumps(ref, indent=2),
                                                            encoding="utf-8")
    return report()


def report() -> int:
    if not RESULTS.exists():
        print("no runs yet")
        return 0
    df = pd.DataFrame([json.loads(x) for x in RESULTS.read_text(encoding="utf-8").splitlines()])
    df = df.drop_duplicates(subset=["variant", "origin", "seed"], keep="last")
    agg = (df.groupby("variant")[["val_RMSE", "test_RMSE", "test_MAE", "pred_mean"]]
             .mean().round(2).sort_values("val_RMSE"))
    agg["n"] = df.groupby("variant").size()
    print("\n=== leaderboard (mean over origins x seeds), ranked by validation RMSE ===")
    print(agg.to_string())
    if "baseline" in df.variant.values:
        b = df[df.variant == "baseline"].set_index(["origin", "seed"])
        rows = []
        for name, g in df.groupby("variant"):
            if name == "baseline":
                continue
            g = g.set_index(["origin", "seed"])
            common = g.index.intersection(b.index)
            if len(common) == 0:
                continue
            d = g.loc[common, "test_RMSE"] - b.loc[common, "test_RMSE"]
            dv = g.loc[common, "val_RMSE"] - b.loc[common, "val_RMSE"]
            rows.append({"variant": name, "d_val": round(dv.mean(), 2),
                         "d_test": round(d.mean(), 2),
                         "better_runs": f"{int((d < 0).sum())}/{len(d)}",
                         "folds_better": int(
                             (g.loc[common].groupby("origin")["test_RMSE"].mean()
                              - b.loc[common].groupby("origin")["test_RMSE"].mean() < 0).sum())})
        print("\n=== paired against baseline (same origins and seeds; minus = better) ===")
        print(pd.DataFrame(rows).sort_values("d_val").to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
