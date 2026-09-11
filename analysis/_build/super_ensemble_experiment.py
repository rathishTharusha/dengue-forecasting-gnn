"""Super-ensemble optimization: combining ASTGCN, A3TGCN, and Persistence.

Solves the optimal simplex-constrained forecast combination weights on validation:
    min_{w >= 0, sum(w)=1} || y_val - sum(w_i * y_hat_i) ||^2
per horizon step, and evaluates on out-of-sample disjoint test origins.

Compares strictly against:
1. Persistence floor (27.571 RMSE)
2. A3TGCN single-model baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy import stats

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "src"))

import adaptive as base
import improved as imp
import reproduced as arch
from run_beat_baseline import (
    WINDOW, HORIZON, NPY, ADJ, make_origins, persistence_counts,
    train_one, predict, score_arm
)

OUT_DIR = REPO / "analysis" / "results" / "super_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def fit_simplex_weights(preds_val: list[np.ndarray], truth_val: np.ndarray) -> np.ndarray:
    """Fit convex combination weights on validation per horizon: min ||y - W*p||^2 s.t. w >= 0, sum(w)=1."""
    n_models = len(preds_val)
    horizon = preds_val[0].shape[-1]
    weights = np.zeros((horizon, n_models))

    for h in range(horizon):
        # Stack model predictions: (N, n_models)
        cols = [p[..., h].ravel() for p in preds_val]
        X = np.column_stack(cols)
        y = truth_val[..., h].ravel()

        def loss(w):
            pred = X @ w
            return np.mean((y - pred) ** 2)

        # Initial equal weights
        w0 = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(loss, w0, bounds=bounds, constraints=constraints, method="SLSQP")
        if res.success:
            weights[h] = res.x
        else:
            weights[h] = w0

    return weights

def apply_weights(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Apply horizon weights: (horizon, n_models) to predictions."""
    horizon = preds[0].shape[-1]
    out = np.zeros_like(preds[0])
    for h in range(horizon):
        for i, p in enumerate(preds):
            out[..., h] += weights[h, i] * p[..., h]
    return out

def run_experiment(origins, seeds, epochs=150):
    cases, adjacency, _ = base.load_dataset(NPY, ADJ)
    test_frac = 0.05
    folds = base.build_folds(cases, WINDOW, HORIZON, origins=origins, test_frac=test_frac)
    src, dst = np.nonzero(adjacency)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    print(f"Running Super-Ensemble across {len(folds)} origins: {origins}")
    print(f"Seeds: {seeds} | Epochs: {epochs}")

    results_by_origin = []

    for fold in folds:
        print(f"\n--- Origin {fold.origin:.2f} (test n={len(fold.test_index)}) ---")
        truth_val = fold.inverse(fold.y_val.numpy())
        truth_test = fold.inverse(fold.y_test.numpy())
        pers_val = persistence_counts(cases, fold.val_index, HORIZON)
        pers_test = persistence_counts(cases, fold.test_index, HORIZON)
        artifact = imp.artifact_windows(fold.test_index, WINDOW, HORIZON)

        models = ["ASTGCN", "A3TGCN"]
        preds_val_by_model = {m: [] for m in models}
        preds_test_by_model = {m: [] for m in models}

        for model_name in models:
            t0 = time.time()
            for seed in seeds:
                net = train_one(model_name, fold, edge_index, seed, head="det", epochs=epochs, in_width=WINDOW)
                m_val, _ = predict(net, fold, "val", edge_index, "det")
                m_test, _ = predict(net, fold, "test", edge_index, "det")
                preds_val_by_model[model_name].append(np.expm1(m_val))
                preds_test_by_model[model_name].append(np.expm1(m_test))
            dt = time.time() - t0
            print(f"  {model_name} {len(seeds)} seeds trained in {dt:.1f}s")

        # Seed-average predictions
        ast_val = np.mean(preds_val_by_model["ASTGCN"], axis=0)
        ast_test = np.mean(preds_test_by_model["ASTGCN"], axis=0)

        a3_val = np.mean(preds_val_by_model["A3TGCN"], axis=0)
        a3_test = np.mean(preds_test_by_model["A3TGCN"], axis=0)

        # Baseline evaluation
        pers_score = score_arm(pers_test, truth_test, artifact, arch="persistence", arm="floor", origin=fold.origin)
        ast_score = score_arm(ast_test, truth_test, artifact, arch="ASTGCN", arm="ens_raw", origin=fold.origin)
        a3_score = score_arm(a3_test, truth_test, artifact, arch="A3TGCN", arm="ens_raw", origin=fold.origin)

        # Strategy 1: Simple Equal Blend of (ASTGCN + A3TGCN)
        combo_5050 = 0.5 * ast_test + 0.5 * a3_test
        combo_5050_score = score_arm(combo_5050, truth_test, artifact, arch="Combo_AST_A3", arm="equal_blend", origin=fold.origin)

        # Strategy 2: Optimal Simplex Weights of (ASTGCN + A3TGCN)
        w_gnn = fit_simplex_weights([ast_val, a3_val], truth_val)
        combo_opt_test = apply_weights([ast_test, a3_test], w_gnn)
        combo_opt_score = score_arm(combo_opt_test, truth_test, artifact, arch="Combo_AST_A3", arm="optimal_weights", origin=fold.origin,
                                    w_ast=[round(float(x), 3) for x in w_gnn[:, 0]],
                                    w_a3=[round(float(x), 3) for x in w_gnn[:, 1]])

        # Strategy 3: Super-Ensemble (ASTGCN + A3TGCN + Persistence)
        w_super = fit_simplex_weights([ast_val, a3_val, pers_val], truth_val)
        super_test = apply_weights([ast_test, a3_test, pers_test], w_super)
        super_score = score_arm(super_test, truth_test, artifact, arch="SuperEnsemble", arm="ast_a3_pers", origin=fold.origin,
                                w_ast=[round(float(x), 3) for x in w_super[:, 0]],
                                w_a3=[round(float(x), 3) for x in w_super[:, 1]],
                                w_pers=[round(float(x), 3) for x in w_super[:, 2]])

        print(f"  Floor RMSE: {pers_score['RMSE_clean']:.3f}")
        print(f"  A3TGCN RMSE: {a3_score['RMSE_clean']:.3f} ({a3_score['RMSE_clean'] - pers_score['RMSE_clean']:+.3f})")
        print(f"  ASTGCN RMSE: {ast_score['RMSE_clean']:.3f} ({ast_score['RMSE_clean'] - pers_score['RMSE_clean']:+.3f})")
        print(f"  AST+A3 (50/50): {combo_5050_score['RMSE_clean']:.3f} ({combo_5050_score['RMSE_clean'] - pers_score['RMSE_clean']:+.3f})")
        print(f"  SuperEnsemble: {super_score['RMSE_clean']:.3f} ({super_score['RMSE_clean'] - pers_score['RMSE_clean']:+.3f})")

        results_by_origin.append({
            "origin": fold.origin,
            "floor": pers_score,
            "A3TGCN": a3_score,
            "ASTGCN": ast_score,
            "Combo_5050": combo_5050_score,
            "Combo_Opt": combo_opt_score,
            "SuperEnsemble": super_score,
        })

    # Consolidate and test significance across origins
    print("\n==========================================================================")
    print("                    FINAL SUPER-ENSEMBLE DECISION TABLE                   ")
    print("==========================================================================")
    
    n_org = len(results_by_origin)
    floors = np.array([r["floor"]["RMSE_clean"] for r in results_by_origin])
    a3_vals = np.array([r["A3TGCN"]["RMSE_clean"] for r in results_by_origin])
    ast_vals = np.array([r["ASTGCN"]["RMSE_clean"] for r in results_by_origin])
    c50_vals = np.array([r["Combo_5050"]["RMSE_clean"] for r in results_by_origin])
    copt_vals = np.array([r["Combo_Opt"]["RMSE_clean"] for r in results_by_origin])
    super_vals = np.array([r["SuperEnsemble"]["RMSE_clean"] for r in results_by_origin])

    models_eval = [
        ("Persistence Floor", floors),
        ("A3TGCN (det ens)", a3_vals),
        ("ASTGCN (det ens)", ast_vals),
        ("ASTGCN + A3TGCN (50/50)", c50_vals),
        ("ASTGCN + A3TGCN (Opt Val)", copt_vals),
        ("SuperEnsemble (AST+A3+Pers)", super_vals),
    ]

    print(f"{'Model':30s} {'RMSE':>8s} {'vs Floor':>10s} {'Wins/Floor':>12s} {'p (vs Floor)':>14s} {'vs A3TGCN':>11s} {'p (vs A3)':>12s}")
    print("-" * 105)

    for name, vals in models_eval:
        diff_floor = vals - floors
        wins_floor = np.sum(diff_floor < 0)
        t_f, p_f = stats.ttest_rel(vals, floors) if name != "Persistence Floor" else (0, 1.0)
        
        diff_a3 = vals - a3_vals
        wins_a3 = np.sum(diff_a3 < 0)
        t_a3, p_a3 = stats.ttest_rel(vals, a3_vals) if name != "A3TGCN (det ens)" else (0, 1.0)

        vs_f_str = f"{diff_floor.mean():+10.3f}" if name != "Persistence Floor" else "       ---"
        p_f_str = f"{p_f:14.4f}" if name != "Persistence Floor" else "           ---"
        w_f_str = f"{wins_floor:2d}/{n_org:2d}" if name != "Persistence Floor" else "        ---"

        vs_a3_str = f"{diff_a3.mean():+11.3f}" if name != "A3TGCN (det ens)" else "        ---"
        p_a3_str = f"{p_a3:12.4f}" if name != "A3TGCN (det ens)" else "         ---"

        print(f"{name:30s} {vals.mean():8.3f} {vs_f_str} {w_f_str:>12s} {p_f_str} {vs_a3_str} {p_a3_str}")

    out_file = OUT_DIR / "super_ensemble_results.json"
    out_file.write_text(json.dumps(results_by_origin, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to {out_file}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    epochs = 20 if args.quick else args.epochs
    seeds = tuple(range(1 if args.quick else args.seeds))
    origins, _ = make_origins(9, 0.50, 0.90)
    run_experiment(origins, seeds, epochs=epochs)
