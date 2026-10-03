"""Stage S3 — SEIR Core Twin Experiment (Synthetic Code & Identifiability Test)

Generates synthetic epidemic data with known force of infection lambda(t) and Poisson-sampled
cases. Fits both F-win and F-seq formulations to test identifiability of lambda.

Checks Gate G3:
1. r(lambda_hat, lambda_true) >= 0.90
2. Model case RMSE <= 1.10 * Poisson noise floor.

Saves results to `analysis/results/seir_gnn/s3_twin/`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))

import corrected_data  # noqa: E402
import seir_sim  # noqa: E402

OUT_DIR = REPO / "analysis" / "results" / "seir_gnn" / "s3_twin"


class TwinLambdaHead(nn.Module):
    def __init__(self, num_nodes: int, in_window: int = 3, hidden_dim: int = 64, lambda_max: float = 0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.lambda_max = lambda_max
        self.net = nn.Sequential(
            nn.Linear(in_window, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigmoid_out = self.net(x).squeeze(-1)
        return self.lambda_max * sigmoid_out


def run_s3():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    data = corrected_data.load()
    num_nodes = len(data.names)
    pop = torch.tensor(data.population[0], dtype=torch.float32)

    weeks = 120
    omega = 0.7 / 7.0
    gamma = 1.0 / 7.0
    rho = 1.0 / 11.0

    t_weeks = torch.linspace(0, weeks, weeks)
    lambda_true = torch.zeros((num_nodes, weeks), dtype=torch.float32)
    for i in range(num_nodes):
        phase = (i / num_nodes) * 2 * np.pi
        lambda_true[i] = 0.05 + 0.03 * torch.sin(2 * np.pi * t_weeks / 52.0 + phase)

    state0 = torch.stack([
        pop * 0.50,
        pop * 0.001,
        pop * 0.001,
        pop * 0.498
    ], dim=-1)

    states, incidence = seir_sim.simulate_weeks(state0, lambda_true, omega, gamma, substeps=7)

    expected_cases = rho * incidence
    observed_cases = torch.poisson(expected_cases)

    np.savez(
        OUT_DIR / "synthetic_seir.npz",
        lambda_true=lambda_true.numpy(),
        incidence=incidence.detach().numpy(),
        observed_cases=observed_cases.numpy(),
        population=pop.numpy(),
        tag="synthetic"
    )

    poisson_var = expected_cases.numpy()
    noise_floor_rmse = float(np.sqrt(np.mean(poisson_var)))
    print(f"Synthetic dataset generated ({weeks} weeks, {num_nodes} districts).", flush=True)
    print(f"Theoretical Poisson noise floor RMSE: {noise_floor_rmse:.4f}", flush=True)

    in_win = 3
    log_cases = torch.log1p(observed_cases)

    formulations = {}
    for form_name in ["F-win", "F-seq"]:
        print(f"\n--- Training Twin Model ({form_name}) ---", flush=True)
        model = TwinLambdaHead(num_nodes=num_nodes, in_window=in_win, hidden_dim=64, lambda_max=0.2)
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.MSELoss()

        epochs = 150
        t0 = time.time()

        if form_name == "F-win":
            windows_x = []
            targets_y = []
            s0_list = []
            for t in range(in_win, weeks - 1):
                windows_x.append(log_cases[:, t - in_win:t])
                targets_y.append(observed_cases[:, t])
                I_est = (observed_cases[:, t - 1] / rho).clamp_min(1.0)
                E_est = (observed_cases[:, t - 2] / rho).clamp_min(1.0)
                cum_cases = observed_cases[:, :t].sum(dim=-1)
                S_est = (pop * 0.50 - cum_cases / rho).clamp_min(10.0)
                R_est = (pop - S_est - E_est - I_est).clamp_min(0.0)
                s0_list.append(torch.stack([S_est, E_est, I_est, R_est], dim=-1))

            batch_x = torch.stack(windows_x)  # (B, N, in_win)
            batch_y = torch.stack(targets_y)  # (B, N)
            batch_s0 = torch.stack(s0_list)    # (B, N, 4)

            for epoch in range(epochs):
                optimizer.zero_grad()
                lam_pred = model(batch_x)  # (B, N)
                lam_1w = lam_pred.unsqueeze(-1)  # (B, N, 1)
                _, inc_pred = seir_sim.simulate_weeks(batch_s0, lam_1w, omega, gamma, substeps=7)
                pred_cases = rho * inc_pred.squeeze(-1)  # (B, N)
                loss = criterion(pred_cases, batch_y)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                pred_lambdas_stack = model(batch_x).T  # (N, B)
                pred_cases_stack = pred_cases.T        # (N, B)
                target_cases = batch_y.T                # (N, B)

        else:  # F-seq: single vectorized rollout
            rollout_x = []
            for t in range(in_win, weeks):
                rollout_x.append(log_cases[:, t - in_win:t])
            batch_rollout_x = torch.stack(rollout_x)  # (T_roll, N, in_win)

            target_cases = observed_cases[:, in_win:]  # (N, T_roll)

            for epoch in range(epochs):
                optimizer.zero_grad()
                lam_pred_all = model(batch_rollout_x)  # (T_roll, N)
                lam_seq = lam_pred_all.T  # (N, T_roll)

                _, inc_pred_seq = seir_sim.simulate_weeks(state0, lam_seq, omega, gamma, substeps=7)
                pred_cases_stack = rho * inc_pred_seq  # (N, T_roll)

                loss = criterion(pred_cases_stack, target_cases)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                pred_lambdas_stack = lam_seq

        target_lambdas = lambda_true[:, in_win:in_win + pred_lambdas_stack.shape[1]]
        lam_true_flat = target_lambdas.detach().numpy().flatten()
        lam_pred_flat = pred_lambdas_stack.detach().numpy().flatten()

        r_corr = float(np.corrcoef(lam_true_flat, lam_pred_flat)[0, 1])

        target_c_flat = target_cases.detach().numpy().flatten()
        pred_c_flat = pred_cases_stack.detach().numpy().flatten()
        model_rmse = float(np.sqrt(np.mean((pred_c_flat - target_c_flat) ** 2)))

        noise_ratio = model_rmse / noise_floor_rmse
        gate_pass = (r_corr >= 0.90) and (noise_ratio <= 1.10)

        formulations[form_name] = {
            "r_lambda": r_corr,
            "rmse_cases": model_rmse,
            "noise_floor_rmse": noise_floor_rmse,
            "noise_ratio": noise_ratio,
            "gate_g3_pass": gate_pass
        }

        print(f"Finished {form_name} in {time.time() - t0:.1f}s:", flush=True)
        print(f"  r(lambda_hat, lambda_true): {r_corr:.4f}  (target >= 0.90)", flush=True)
        print(f"  Cases RMSE:                 {model_rmse:.4f}  (Noise floor: {noise_floor_rmse:.4f})", flush=True)
        print(f"  RMSE / Noise Floor Ratio:   {noise_ratio:.4f}  (target <= 1.10)", flush=True)
        print(f"  Gate G3 Pass:              {gate_pass}", flush=True)

    summary_path = OUT_DIR / "s3_twin_results.json"
    summary_path.write_text(json.dumps(formulations, indent=2), encoding="utf-8")
    print(f"\nWrote twin experiment results to {summary_path}", flush=True)

    overall_g3 = any(v["gate_g3_pass"] for v in formulations.values())
    print(f"\n==========================================", flush=True)
    print(f"GATE G3 OVERALL STATUS: {'PASSED' if overall_g3 else 'FAILED'}", flush=True)
    print(f"==========================================", flush=True)

    return 0 if overall_g3 else 1


if __name__ == "__main__":
    sys.exit(run_s3())
