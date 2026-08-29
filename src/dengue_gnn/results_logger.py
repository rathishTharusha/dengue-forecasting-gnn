import os
import csv
from datetime import datetime, timezone
import pandas as pd

def log_experiment_result(exp_id: str, cfg: object, df_metrics: pd.DataFrame, csv_path: str = "results/experiment_results.csv") -> str:
    """Appends a fully-configured experiment summary row to csv_path.
    
    Args:
        exp_id: Unique experiment identifier (e.g. 'EXP-002', 'EXP-003').
        cfg: Configuration object (e.g. CFG class or dict).
        df_metrics: DataFrame containing per-fold metric results.
        csv_path: Target CSV file path.
        
    Returns:
        Absolute path to written CSV file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    
    # Extract config parameters safely
    get_cfg = lambda attr, default: getattr(cfg, attr, default) if not isinstance(cfg, dict) else cfg.get(attr, default)
    
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exp_id": exp_id,
        "model": get_cfg("model", "GCN"),
        "residual": get_cfg("residual", True),
        "log_transform": get_cfg("log_transform", True),
        "use_adaptive": get_cfg("use_adaptive", False),
        "emb_dim": get_cfg("emb_dim", 10),
        "lambda_phys": get_cfg("lambda_phys", 0.0),
        "hidden": get_cfg("hidden", 64),
        "dropout": get_cfg("dropout", 0.1),
        "lr": get_cfg("lr", 1e-3),
        "weight_decay": get_cfg("weight_decay", 5e-4),
        "epochs": get_cfg("epochs", 120),
        "patience": get_cfg("patience", 25),
        "mean_rmse": float(df_metrics.get("AdaptiveGCN_RMSE", df_metrics.get("PhysicsSkeleton_RMSE", df_metrics.get("RMSE", [0]))).mean()) \
                    if not df_metrics.empty else 0.0,
        "mean_mae": float(df_metrics.get("AdaptiveGCN_MAE", df_metrics.get("PhysicsSkeleton_MAE", df_metrics.get("MAE", [0]))).mean()) \
                   if not df_metrics.empty else 0.0,
        "mean_smape": float(df_metrics.get("AdaptiveGCN_SMAPE", df_metrics.get("PhysicsSkeleton_SMAPE", df_metrics.get("SMAPE", [0]))).mean()) \
                     if not df_metrics.empty else 0.0,
        "mean_pte": float(df_metrics.get("AdaptiveGCN_PTE", df_metrics.get("PhysicsSkeleton_PTE", df_metrics.get("PeakTimingErr", [0]))).mean()) \
                   if not df_metrics.empty else 0.0,
        "fold1_rmse": float(df_metrics.iloc[0].get("AdaptiveGCN_RMSE", df_metrics.iloc[0].get("PhysicsSkeleton_RMSE", df_metrics.iloc[0].get("RMSE", 0.0)))) \
                      if len(df_metrics) > 0 else 0.0,
        "fold2_rmse": float(df_metrics.iloc[1].get("AdaptiveGCN_RMSE", df_metrics.iloc[1].get("PhysicsSkeleton_RMSE", df_metrics.iloc[1].get("RMSE", 0.0)))) \
                      if len(df_metrics) > 1 else 0.0,
        "fold3_rmse": float(df_metrics.iloc[2].get("AdaptiveGCN_RMSE", df_metrics.iloc[2].get("PhysicsSkeleton_RMSE", df_metrics.iloc[2].get("RMSE", 0.0)))) \
                      if len(df_metrics) > 2 else 0.0,
        "learned_gate_sig": float(df_metrics["learned_gate_sig"].mean()) if "learned_gate_sig" in df_metrics.columns else 1.0,
    }
    
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    headers = list(row.keys())
    
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        
    return os.path.abspath(csv_path)
