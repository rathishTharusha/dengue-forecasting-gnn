"""Stage S5: SEIR-GNN benchmark and physics-informed GNN comparison.

Phase 1 -- Screen: 12 arms on A3TGCN across 3 origins x 3 seeds.
           Factors: inputs (cases, cases+era5, cases+era5+ndvi),
                    spatial coupling (implicit, explicit),
                    head (foi, direct).
Phase 2 -- Expand: Top 2 SEIR configurations + direct controls across 5 GNNs
           (A3TGCN, STGAT, ASTGCN, AAGCN, DCRNN).

Output: analysis/results/seir_gnn/s5_seir_gnn/s5_seir_gnn_results.json
"""

from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "analysis" / "_build"))
sys.path.insert(0, str(REPO / "src"))

import corrected_data as cd  # noqa: E402
import reproduced as arch  # noqa: E402
import run_corrected_benchmark as rcb  # noqa: E402
import seir_sim  # noqa: E402


class SEIRGNNModel(nn.Module):
    def __init__(
        self,
        arch_name: str,
        in_dim: int,
        edge_index: torch.Tensor,
        adj_dense: torch.Tensor,
        head_type: str = "foi",
        coupling: str = "implicit",
        lambda_max: float = 1.0 / 7.0,
    ):
        super().__init__()
        self.arch_name = arch_name
        self.head_type = head_type
        self.coupling = coupling
        self.lambda_max = lambda_max
        self.adj_dense = adj_dense
        self.edge_index = edge_index

        # Input feature projection (B, N, 3, C) -> (B, N, 3)
        self.in_proj = nn.Linear(in_dim, 1)

        # Create row-normalized adjacency without self-loops for explicit spatial coupling
        adj_no_diag = adj_dense.clone()
        adj_no_diag.fill_diagonal_(0.0)
        deg = adj_no_diag.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("hat_A", adj_no_diag / deg)

        # Build GNN backbone: n_nodes=25, window=3, horizon=3
        kwargs = {"adaptive": False, "channels": 8} if arch_name == "AAGCN" else {}
        self.backbone = arch.build(arch_name, 25, 3, 3, edge_index=edge_index, **kwargs)

        if head_type == "foi":
            self.foi_fc = nn.Sequential(
                nn.Linear(3, 1),
                nn.Sigmoid()
            )
            if coupling == "explicit":
                self.alpha = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(self, x: torch.Tensor, st0: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, N, T_in, C)
        Returns:
            If head_type == 'foi': lambda_daily (B, N)
            If head_type == 'direct': log1p_cases (B, N, 3)
        """
        B, N, T, C = x.shape
        x_proj = self.in_proj(x).squeeze(-1) # (B, N, 3)
        # Backbone output: (B, N, 3)
        h = self.backbone(x_proj, self.edge_index)

        if self.head_type == "direct":
            return h

        # FOI head: map (B, N, 3) -> (B, N)
        sig = self.foi_fc(h).squeeze(-1) # (B, N)
        lam = self.lambda_max * sig

        if self.coupling == "explicit" and st0 is not None:
            # Import term: alpha * sum_j hat_A_ij * I_j
            i_frac = st0[..., 2] # (B, N)
            import_term = torch.matmul(i_frac, self.hat_A.T) # (B, N)
            alpha_pos = torch.relu(self.alpha)
            lam = lam + alpha_pos * import_term

        return lam


def prepare_inputs_by_level(data: cd.CorrectedData, level: str) -> np.ndarray:
    """Build input tensor features (T, N, 3, C) based on input level."""
    T, N = data.cases.shape
    c_mean = np.nanmean(data.cases)
    c_std = np.nanstd(data.cases) + 1e-8
    cases_norm = (np.log1p(np.nan_to_num(data.cases, nan=0.0)) - c_mean) / c_std

    if level == "cases":
        feat = np.zeros((T, N, 3, 1), dtype=np.float32)
        for i in range(3, T):
            feat[i, :, :, 0] = cases_norm[i - 3 : i].T
        return feat

    clim_mean = np.nanmean(data.climate, axis=(0, 1), keepdims=True)
    clim_std = np.nanstd(data.climate, axis=(0, 1), keepdims=True) + 1e-8
    clim_norm = (data.climate - clim_mean) / clim_std

    if level == "cases+era5":
        feat = np.zeros((T, N, 3, 7), dtype=np.float32)
        for i in range(4, T):
            feat[i, :, :, 0] = cases_norm[i - 3 : i].T
            cl_slice = clim_norm[i - 4 : i - 1]
            feat[i, :, :, 1:7] = np.moveaxis(cl_slice, 0, 1)
        return feat

    ndvi_mean = np.nanmean(data.ndvi)
    ndvi_std = np.nanstd(data.ndvi) + 1e-8
    ndvi_norm = (data.ndvi - ndvi_mean) / ndvi_std

    feat = np.zeros((T, N, 3, 8), dtype=np.float32)
    for i in range(4, T):
        feat[i, :, :, 0] = cases_norm[i - 3 : i].T
        cl_slice = clim_norm[i - 4 : i - 1]
        feat[i, :, :, 1:7] = np.moveaxis(cl_slice, 0, 1)
        nd_slice = ndvi_norm[i - 2 : i + 1]
        feat[i, :, :, 7] = nd_slice.T
    return feat


def run_gnn_job(job_kwargs: dict) -> dict:
    arch_name = job_kwargs["arch"]
    input_level = job_kwargs["input_level"]
    coupling = job_kwargs["coupling"]
    head_type = job_kwargs["head_type"]
    origin = job_kwargs["origin"]
    seed = job_kwargs["seed"]
    epochs = job_kwargs.get("epochs", 120)

    train_idx = job_kwargs["train_idx"]
    val_idx = job_kwargs["val_idx"]
    test_idx = job_kwargs["test_idx"]
    cases = job_kwargs["cases"]
    features = job_kwargs["features"]
    population = job_kwargs["population"]
    edge_index = torch.tensor(job_kwargs["edge_index"], dtype=torch.long)
    adj_dense = torch.tensor(job_kwargs["adj_dense"], dtype=torch.float32)
    mean = job_kwargs["mean"]
    std = job_kwargs["std"]
    rho = 1.0 / 11.0
    s0 = 1.0 - 0.682

    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = features.shape[-1]

    model = SEIRGNNModel(
        arch_name=arch_name,
        in_dim=in_dim,
        edge_index=edge_index,
        adj_dense=adj_dense,
        head_type=head_type,
        coupling=coupling,
        lambda_max=1.0 / 7.0,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    feat_t = torch.tensor(features, dtype=torch.float32)

    def prepare_batch_data(indices):
        y_true = np.stack([cases[i : i + 3].T for i in indices])
        c1 = np.stack([cases[i - 1] for i in indices])
        c2 = np.stack([cases[i - 2] for i in indices])
        pop_i = np.stack([population[i - 1] for i in indices])

        i0 = np.clip(c1 / (rho * pop_i), 1e-6, 0.5)
        e0 = np.clip(c2 / (rho * pop_i), 1e-6, 0.5)
        cum_cases = np.stack([np.nansum(cases[:i], axis=0) for i in indices])
        s0_i = np.clip(s0 - cum_cases / (rho * pop_i), 0.01, 1.0)
        r0_i = np.clip(1.0 - s0_i - e0 - i0, 0.0, 1.0)

        st0 = torch.tensor(np.stack([s0_i, e0, i0, r0_i], axis=-1), dtype=torch.float32)
        x_batch = feat_t[indices]
        pop_t = torch.tensor(pop_i, dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(y_true, dtype=torch.float32)
        return x_batch, st0, pop_t, y_t

    tr_x, tr_st0, tr_pop, tr_y = prepare_batch_data(train_idx)
    va_x, va_st0, va_pop, va_y = prepare_batch_data(val_idx)
    te_x, te_st0, te_pop, te_y = prepare_batch_data(test_idx)

    def run_forward(x_batch, st0, pop_t):
        if head_type == "direct":
            log1p_pred = model(x_batch) # (B, N, 3)
            log1p_cases = log1p_pred * std + mean
            y_pred = torch.expm1(torch.clamp(log1p_cases, -1.0, 12.0))
            return y_pred

        lam = model(x_batch, st0) # (B, N)
        lam_seq = lam.unsqueeze(-1).repeat(1, 1, 3)
        _, inc = seir_sim.simulate_weeks(st0, lam_seq, omega=0.7/7.0, gamma=1.0/7.0, substeps=7)
        y_pred = inc * rho * pop_t
        return y_pred

    best_val_rmse = float("inf")
    best_weights = None
    waited = 0
    patience = 20

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = run_forward(tr_x, tr_st0, tr_pop)

        denom = (torch.abs(preds) + torch.abs(tr_y) + 1e-5) / 2.0
        loss = torch.mean(torch.abs(preds - tr_y) / denom)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if epoch % 3 == 0:
            model.eval()
            with torch.no_grad():
                v_preds = run_forward(va_x, va_st0, va_pop)
                v_rmse = torch.sqrt(torch.mean((v_preds - va_y) ** 2)).item()
                if v_rmse < best_val_rmse - 1e-5:
                    best_val_rmse = v_rmse
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    waited = 0
                else:
                    waited += 1
                    if waited >= patience:
                        break

    if best_weights:
        model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        v_preds = run_forward(va_x, va_st0, va_pop)
        val_rmse = torch.sqrt(torch.mean((v_preds - va_y) ** 2)).item()
        val_mae = torch.mean(torch.abs(v_preds - va_y)).item()

        t_preds = run_forward(te_x, te_st0, te_pop)
        test_rmse = torch.sqrt(torch.mean((t_preds - te_y) ** 2)).item()
        test_mae = torch.mean(torch.abs(t_preds - te_y)).item()

    record = {
        "arch": arch_name,
        "input_level": input_level,
        "coupling": coupling,
        "head_type": head_type,
        "origin": origin,
        "seed": seed,
        "val_RMSE": val_rmse,
        "val_MAE": val_mae,
        "test_RMSE": test_rmse,
        "test_MAE": test_mae,
        "elapsed": round(time.time() - t0, 3)
    }
    print(f"  [OK] {arch_name:7s} in={input_level:15s} coup={coupling:8s} head={head_type:6s} o{origin} s{seed} | "
          f"Val RMSE: {val_rmse:6.2f} | Test RMSE: {test_rmse:6.2f} ({time.time()-t0:.2f}s)", flush=True)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(REPO / "analysis" / "results" / "seir_gnn" / "s5_seir_gnn" / "s5_seir_gnn_results.json"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = cd.load()
    cases, adjacency, artifact, missing, folds = rcb.prepare("rebuilt")
    src, dst = np.nonzero(adjacency)
    edge_index = np.stack([src, dst])

    feat_dict = {
        lvl: prepare_inputs_by_level(data, lvl)
        for lvl in ("cases", "cases+era5", "cases+era5+ndvi")
    }

    # Phase 1: Screen (12 arms on A3TGCN)
    input_levels = ["cases", "cases+era5", "cases+era5+ndvi"]
    couplings = ["implicit", "explicit"]
    heads = ["foi", "direct"]
    seeds = [0, 1, 2]

    jobs = []
    for lvl in input_levels:
        for coup in couplings:
            for head in heads:
                for fold in folds:
                    for seed in seeds:
                        jobs.append({
                            "arch": "A3TGCN",
                            "input_level": lvl,
                            "coupling": coup,
                            "head_type": head,
                            "origin": fold.origin,
                            "seed": seed,
                            "epochs": args.epochs,
                            "train_idx": np.asarray(fold.train_index),
                            "val_idx": np.asarray(fold.val_index),
                            "test_idx": np.asarray(fold.test_index),
                            "cases": cases,
                            "features": feat_dict[lvl],
                            "population": data.population,
                            "edge_index": edge_index,
                            "adj_dense": adjacency,
                            "mean": fold.mean,
                            "std": fold.std,
                        })

    print(f"\n=== Phase 1: Screening 12 arms on A3TGCN ({len(jobs)} jobs on {args.workers} CPU workers) ===")
    t_start = time.time()
    screen_results = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_gnn_job, job) for job in jobs]
        for f in as_completed(futures):
            try:
                screen_results.append(f.result())
            except Exception as e:
                print(f"ERROR: {e}", flush=True)

    df_screen = pd.DataFrame(screen_results)
    screen_summary = df_screen.groupby(["arch", "input_level", "coupling", "head_type"])[["val_RMSE", "test_RMSE"]].mean().reset_index()
    screen_summary = screen_summary.sort_values("val_RMSE")
    print("\n=== Phase 1 Screen Results (A3TGCN) Ranked by Validation RMSE ===")
    print(screen_summary.round(3).to_string(index=False))

    top_seir = screen_summary[screen_summary.head_type == "foi"].head(2)
    print("\nTop 2 SEIR configurations selected for Phase 2 expansion:")
    print(top_seir[["input_level", "coupling", "val_RMSE", "test_RMSE"]].to_string(index=False))

    # Phase 2: Expand across all 5 architectures
    all_archs = ["A3TGCN", "STGAT", "ASTGCN", "AAGCN", "DCRNN"]
    expand_jobs = []

    for _, config in top_seir.iterrows():
        lvl = config["input_level"]
        coup = config["coupling"]
        for h_type in ["foi", "direct"]:
            for arch_name in all_archs:
                if arch_name == "A3TGCN":
                    continue
                for fold in folds:
                    for seed in seeds:
                        expand_jobs.append({
                            "arch": arch_name,
                            "input_level": lvl,
                            "coupling": coup,
                            "head_type": h_type,
                            "origin": fold.origin,
                            "seed": seed,
                            "epochs": args.epochs,
                            "train_idx": np.asarray(fold.train_index),
                            "val_idx": np.asarray(fold.val_index),
                            "test_idx": np.asarray(fold.test_index),
                            "cases": cases,
                            "features": feat_dict[lvl],
                            "population": data.population,
                            "edge_index": edge_index,
                            "adj_dense": adjacency,
                            "mean": fold.mean,
                            "std": fold.std,
                        })

    print(f"\n=== Phase 2: Expanding Top Configurations across {len(all_archs)} Architectures ({len(expand_jobs)} jobs) ===")
    expand_results = []

    if expand_jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_gnn_job, job) for job in expand_jobs]
            for f in as_completed(futures):
                try:
                    expand_results.append(f.result())
                except Exception as e:
                    print(f"ERROR: {e}", flush=True)

    all_results = screen_results + expand_results
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote full Stage S5 results to {out_path} (Total time: {time.time()-t_start:.1f}s)")

    df_full = pd.DataFrame(all_results)
    full_summary = df_full.groupby(["arch", "input_level", "coupling", "head_type"])[["val_RMSE", "test_RMSE"]].mean().reset_index()
    full_summary = full_summary.sort_values("val_RMSE")

    print("\n=== Full Stage S5 SEIR-GNN Leaderboard Ranked by Validation RMSE ===")
    print(full_summary.round(3).to_string(index=False))

    best_s = full_summary.iloc[0]
    print(f"\n[Gate G5] Finalist S*: {best_s['arch']} (input={best_s['input_level']}, coupling={best_s['coupling']}, head={best_s['head_type']}) "
          f"with Val RMSE={best_s['val_RMSE']:.3f}, Test RMSE={best_s['test_RMSE']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
