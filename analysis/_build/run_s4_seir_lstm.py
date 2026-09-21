"""Stage S4: SEIR-LSTM benchmark on corrected dataset (rebuilt).

Formulations compared:
- F-win: independent windows, state rebuilt per window from past cases.
- F-seq: continuous rollout through the series.

Protocol: 3 rolling origins (0.55, 0.70, 0.85) x 3 seeds (0, 1, 2).
Hyperparameters: lambda_max in {1/7, 2/7, 4/7}, loss in {SMAPE, MSE_log1p}.
Parallelized with multiprocessing across CPU cores.
Output: analysis/results/seir_gnn/s4_seir_lstm/s4_seir_lstm_results.json
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

import corrected_data as cd  # noqa: E402
import run_corrected_benchmark as rcb  # noqa: E402
import seir_sim  # noqa: E402


class SEIRLSTMModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, lambda_max: float = 2.0 / 7.0):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.lambda_max = lambda_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T, C)
        Returns: lambda_daily (B, N)
        """
        B, N, T, C = x.shape
        x_flat = x.view(B * N, T, C)
        out, (h_n, c_n) = self.lstm(x_flat)
        h_last = h_n[-1]  # (B*N, hidden_dim)
        sig = self.fc(h_last).view(B, N)
        return self.lambda_max * sig


def build_inputs(data: cd.CorrectedData, window: int = 3, covariate_window: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Build input features (T, N, T_in, C) for every week i."""
    T, N = data.cases.shape
    c_mean = np.nanmean(data.cases)
    c_std = np.nanstd(data.cases) + 1e-8
    cases_norm = (np.log1p(np.nan_to_num(data.cases, nan=0.0)) - c_mean) / c_std

    clim_mean = np.nanmean(data.climate, axis=(0, 1), keepdims=True)
    clim_std = np.nanstd(data.climate, axis=(0, 1), keepdims=True) + 1e-8
    clim_norm = (data.climate - clim_mean) / clim_std

    ndvi_mean = np.nanmean(data.ndvi)
    ndvi_std = np.nanstd(data.ndvi) + 1e-8
    ndvi_norm = (data.ndvi - ndvi_mean) / ndvi_std

    first = max(window, cd.LAGS["climate"] + covariate_window - 1)
    features = np.zeros((T, N, covariate_window, 8), dtype=np.float32)
    for i in range(first, T):
        c_slice = cases_norm[i - covariate_window : i]
        cl_slice = clim_norm[i - 1 - covariate_window : i - 1]
        nd_slice = ndvi_norm[i - covariate_window + 1 : i + 1]
        for t_idx in range(covariate_window):
            features[i, :, t_idx, 0] = c_slice[t_idx]
            features[i, :, t_idx, 1:7] = cl_slice[t_idx]
            features[i, :, t_idx, 7] = nd_slice[t_idx]
    return features, data.population


def run_single_job(job_kwargs: dict) -> dict:
    form = job_kwargs["formulation"]
    l_max = job_kwargs["lambda_max"]
    l_type = job_kwargs["loss_type"]
    origin = job_kwargs["origin"]
    seed = job_kwargs["seed"]
    epochs = job_kwargs["epochs"]
    train_idx = job_kwargs["train_idx"]
    val_idx = job_kwargs["val_idx"]
    test_idx = job_kwargs["test_idx"]
    cases = job_kwargs["cases"]
    features = job_kwargs["features"]
    population = job_kwargs["population"]
    rho = job_kwargs.get("rho", 1.0 / 11.0)
    s0 = job_kwargs.get("s0", 1.0 - 0.682)
    lr = job_kwargs.get("lr", 0.005)

    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = features.shape[-1]
    model = SEIRLSTMModel(in_dim=in_dim, hidden_dim=32, lambda_max=l_max)
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
        lam = model(x_batch)
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

        if l_type == "SMAPE":
            denom = (torch.abs(preds) + torch.abs(tr_y) + 1e-5) / 2.0
            loss = torch.mean(torch.abs(preds - tr_y) / denom)
        else:
            loss = nn.functional.mse_loss(torch.log1p(torch.relu(preds)), torch.log1p(tr_y))

        loss.backward()
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
        "formulation": form,
        "lambda_max": round(l_max * 7.0, 1),
        "loss_type": l_type,
        "origin": origin,
        "seed": seed,
        "val_RMSE": val_rmse,
        "val_MAE": val_mae,
        "test_RMSE": test_rmse,
        "test_MAE": test_mae,
        "elapsed": round(time.time() - t0, 3)
    }
    print(f"  [OK] {form:5s} lmax={l_max*7:.0f}/wk loss={l_type:9s} o{origin} s{seed} | "
          f"Val RMSE: {val_rmse:6.2f} | Test RMSE: {test_rmse:6.2f} ({time.time()-t0:.2f}s)", flush=True)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(REPO / "analysis" / "results" / "seir_gnn" / "s4_seir_lstm" / "s4_seir_lstm_results.json"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = cd.load()
    cases, adjacency, artifact, missing, folds = rcb.prepare("rebuilt")
    features, population = build_inputs(data)

    formulations = ["F-win"]
    lambda_maxs = [1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0]
    losses = ["MSE_log1p", "SMAPE"]
    seeds = [0, 1, 2]

    jobs = []
    for form in formulations:
        for l_max in lambda_maxs:
            for l_type in losses:
                for fold in folds:
                    for seed in seeds:
                        jobs.append({
                            "formulation": form,
                            "lambda_max": l_max,
                            "loss_type": l_type,
                            "origin": fold.origin,
                            "seed": seed,
                            "epochs": args.epochs,
                            "train_idx": np.asarray(fold.train_index),
                            "val_idx": np.asarray(fold.val_index),
                            "test_idx": np.asarray(fold.test_index),
                            "cases": cases,
                            "features": features,
                            "population": population,
                        })

    print(f"Running {len(jobs)} S4 SEIR-LSTM benchmark jobs on {args.workers} CPU workers...")
    t_start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_single_job, job) for job in jobs]
        for f in as_completed(futures):
            try:
                rec = f.result()
                results.append(rec)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote results to {out_path} (Total time: {time.time()-t_start:.1f}s)")

    df = pd.DataFrame(results)
    summary = df.groupby(["formulation", "lambda_max", "loss_type"])[["val_RMSE", "test_RMSE"]].mean().reset_index()
    summary = summary.sort_values("val_RMSE")
    print("\n=== Stage S4 Formulations Ranked by Validation RMSE ===")
    print(summary.round(3).to_string(index=False))

    best = summary.iloc[0]
    print(f"\n[Gate G4] Winner F*: {best['formulation']} (lambda_max={best['lambda_max']:.0f}/wk, loss={best['loss_type']}) "
          f"with Val RMSE={best['val_RMSE']:.3f}, Test RMSE={best['test_RMSE']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
