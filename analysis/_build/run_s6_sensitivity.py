"""Stage S6: Sensitivity and Oracle Arms on S*.

Takes the finalist model S* selected from Stage S5 (lowest validation RMSE)
and evaluates robustness across parameter variations:
- rho in {1/2.5, 1/11, 1/30}
- S0 in {1 - 0.514, 1 - 0.682, 1 - 0.908}
- omega/gamma: (0.7/7, 1/7) vs ( (7/5.9)/7, (7/4.5)/7 )
- state_assimilation: False vs True
- oracle_reset2017: False vs True

Output: analysis/results/seir_gnn/s6_sensitivity/s6_sensitivity_results.json
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
from run_s5_seir_gnn import SEIRGNNModel, prepare_inputs_by_level  # noqa: E402


def run_s6_arm(job_kwargs: dict) -> dict:
    arch_name = job_kwargs["arch"]
    input_level = job_kwargs["input_level"]
    coupling = job_kwargs["coupling"]
    head_type = job_kwargs["head_type"]
    origin = job_kwargs["origin"]
    seed = job_kwargs["seed"]
    epochs = job_kwargs.get("epochs", 120)

    arm_name = job_kwargs["arm_name"]
    rho = job_kwargs.get("rho", 1.0 / 11.0)
    s0_val = job_kwargs.get("s0_val", 1.0 - 0.682)
    omega = job_kwargs.get("omega", 0.7 / 7.0)
    gamma = job_kwargs.get("gamma", 1.0 / 7.0)
    state_assimilation = job_kwargs.get("state_assimilation", False)
    oracle_reset2017 = job_kwargs.get("oracle_reset2017", False)

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

        if oracle_reset2017:
            # 2017 mega-outbreak serotype reset around week 230
            # For windows after week 230, reset cumulative count baseline to week 230
            cum_after_2017 = np.stack([
                np.nansum(cases[230:i], axis=0) if i > 230 else np.nansum(cases[:i], axis=0)
                for i in indices
            ])
            s0_i = np.clip(s0_val - cum_after_2017 / (rho * pop_i), 0.01, 1.0)
        else:
            s0_i = np.clip(s0_val - cum_cases / (rho * pop_i), 0.01, 1.0)

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
            log1p_pred = model(x_batch)
            log1p_cases = log1p_pred * std + mean
            return torch.expm1(torch.clamp(log1p_cases, -1.0, 12.0))

        lam = model(x_batch, st0)
        lam_seq = lam.unsqueeze(-1).repeat(1, 1, 3)

        if state_assimilation:
            # Rescale I0 to match observed c1 exactly under recovery rate gamma
            c1_obs = tr_y[:, :, 0] # (B, N)
            i0_assim = torch.clamp(c1_obs / (gamma * rho * pop_t.squeeze(-1)), 1e-6, 0.5)
            st0 = st0.clone()
            st0[..., 2] = i0_assim
            # Rebalance S and R
            st0[..., 0] = torch.clamp(1.0 - st0[..., 1] - st0[..., 2] - st0[..., 3], 0.01, 1.0)

        _, inc = seir_sim.simulate_weeks(st0, lam_seq, omega=omega, gamma=gamma, substeps=7)
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
        "arm_name": arm_name,
        "arch": arch_name,
        "input_level": input_level,
        "coupling": coupling,
        "head_type": head_type,
        "rho": rho,
        "s0_val": s0_val,
        "omega": omega,
        "gamma": gamma,
        "state_assimilation": state_assimilation,
        "oracle_reset2017": oracle_reset2017,
        "origin": origin,
        "seed": seed,
        "val_RMSE": val_rmse,
        "val_MAE": val_mae,
        "test_RMSE": test_rmse,
        "test_MAE": test_mae,
        "elapsed": round(time.time() - t0, 3)
    }
    print(f"  [OK] Arm={arm_name:18s} o{origin} s{seed} | Val RMSE: {val_rmse:6.2f} | Test RMSE: {test_rmse:6.2f}", flush=True)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s5-results", default=str(REPO / "analysis" / "results" / "seir_gnn" / "s5_seir_gnn" / "s5_seir_gnn_results.json"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(REPO / "analysis" / "results" / "seir_gnn" / "s6_sensitivity" / "s6_sensitivity_results.json"))
    args = ap.parse_args()

    s5_path = Path(args.s5_results)
    if not s5_path.exists():
        print(f"Error: S5 results not found at {s5_path}")
        return 1

    s5_data = json.loads(s5_path.read_text(encoding="utf-8"))
    df_s5 = pd.DataFrame(s5_data)
    # Pick S* (lowest validation RMSE in S5)
    summary_s5 = df_s5.groupby(["arch", "input_level", "coupling", "head_type"])[["val_RMSE", "test_RMSE"]].mean().reset_index()
    summary_s5 = summary_s5.sort_values("val_RMSE")
    s_star = summary_s5.iloc[0]

    print(f"=== Stage S6: Sensitivity Analysis on S* ===")
    print(f"Selected S*: Arch={s_star['arch']}, Input={s_star['input_level']}, Coupling={s_star['coupling']}, Head={s_star['head_type']}")
    print(f"Val RMSE: {s_star['val_RMSE']:.3f}, Test RMSE: {s_star['test_RMSE']:.3f}")

    data = cd.load()
    cases, adjacency, artifact, missing, folds = rcb.prepare("rebuilt")
    src, dst = np.nonzero(adjacency)
    edge_index = np.stack([src, dst])
    feat_dict = {
        lvl: prepare_inputs_by_level(data, lvl)
        for lvl in ("cases", "cases+era5", "cases+era5+ndvi")
    }

    seeds = [0, 1, 2]
    sensitivity_arms = [
        {"arm_name": "primary_S*", "rho": 1.0/11.0, "s0_val": 1.0-0.682, "omega": 0.7/7.0, "gamma": 1.0/7.0, "state_assimilation": False, "oracle_reset2017": False},
        {"arm_name": "rho_1_div_2.5", "rho": 1.0/2.5, "s0_val": 1.0-0.682, "omega": 0.7/7.0, "gamma": 1.0/7.0, "state_assimilation": False, "oracle_reset2017": False},
        {"arm_name": "rho_1_div_30", "rho": 1.0/30.0, "s0_val": 1.0-0.682, "omega": 0.7/7.0, "gamma": 1.0/7.0, "state_assimilation": False, "oracle_reset2017": False},
        {"arm_name": "s0_0.486", "rho": 1.0/11.0, "s0_val": 1.0-0.514, "omega": 0.7/7.0, "gamma": 1.0/7.0, "state_assimilation": False, "oracle_reset2017": False},
        {"arm_name": "s0_0.092", "rho": 1.0/11.0, "s0_val": 1.0-0.908, "omega": 0.7/7.0, "gamma": 1.0/7.0, "state_assimilation": False, "oracle_reset2017": False},
        {"arm_name": "omega_gamma_alt", "rho": 1.0/11.0, "s0_val": 1.0-0.682, "omega": (7.0/5.9)/7.0, "gamma": (7.0/4.5)/7.0, "state_assimilation": False, "oracle_reset2017": False},
        {"arm_name": "state_assimilation", "rho": 1.0/11.0, "s0_val": 1.0-0.682, "omega": 0.7/7.0, "gamma": 1.0/7.0, "state_assimilation": True, "oracle_reset2017": False},
        {"arm_name": "oracle_reset2017", "rho": 1.0/11.0, "s0_val": 1.0-0.682, "omega": 0.7/7.0, "gamma": 1.0/7.0, "state_assimilation": False, "oracle_reset2017": True},
    ]

    jobs = []
    for arm in sensitivity_arms:
        for fold in folds:
            for seed in seeds:
                job = {
                    "arch": s_star["arch"],
                    "input_level": s_star["input_level"],
                    "coupling": s_star["coupling"],
                    "head_type": s_star["head_type"],
                    "origin": fold.origin,
                    "seed": seed,
                    "epochs": args.epochs,
                    "train_idx": np.asarray(fold.train_index),
                    "val_idx": np.asarray(fold.val_index),
                    "test_idx": np.asarray(fold.test_index),
                    "cases": cases,
                    "features": feat_dict[s_star["input_level"]],
                    "population": data.population,
                    "edge_index": edge_index,
                    "adj_dense": adjacency,
                    "mean": fold.mean,
                    "std": fold.std,
                }
                job.update(arm)
                jobs.append(job)

    print(f"\n=== Executing {len(jobs)} Stage S6 Jobs on {args.workers} Workers ===")
    t_start = time.time()
    s6_results = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_s6_arm, j) for j in jobs]
        for f in as_completed(futures):
            try:
                s6_results.append(f.result())
            except Exception as e:
                print(f"ERROR in S6 arm: {e}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(s6_results, indent=2), encoding="utf-8")

    df_s6 = pd.DataFrame(s6_results)
    s6_summary = df_s6.groupby("arm_name")[["val_RMSE", "test_RMSE"]].mean().reset_index()
    s6_summary = s6_summary.sort_values("val_RMSE")

    print("\n=== Stage S6 Sensitivity Summary Ranked by Validation RMSE ===")
    print(s6_summary.round(3).to_string(index=False))
    print(f"Wrote S6 results to {out_path} in {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
