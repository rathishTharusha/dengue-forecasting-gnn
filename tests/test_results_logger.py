import os
import tempfile
import pandas as pd
import pytest
from dengue_gnn.results_logger import log_experiment_result

class DummyCFG:
    model = "AdaptiveGCN"
    residual = True
    log_transform = True
    use_adaptive = True
    emb_dim = 10
    lambda_phys = 0.1
    hidden = 64
    dropout = 0.1
    lr = 1e-3
    weight_decay = 5e-4
    epochs = 120
    patience = 25

def test_log_experiment_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "results.csv")
        
        df_metrics = pd.DataFrame([
            {"fold": 1, "RMSE": 27.2, "MAE": 13.5, "SMAPE": 53.7, "PeakTimingErr": 0.83, "learned_gate_sig": 0.499},
            {"fold": 2, "RMSE": 39.9, "MAE": 17.1, "SMAPE": 68.7, "PeakTimingErr": 0.94, "learned_gate_sig": 0.482},
            {"fold": 3, "RMSE": 67.4, "MAE": 16.5, "SMAPE": 83.2, "PeakTimingErr": 0.84, "learned_gate_sig": 0.499},
        ])
        
        # Log first experiment
        written_path = log_experiment_result("EXP-003", DummyCFG, df_metrics, csv_path=csv_file)
        assert os.path.exists(written_path)
        
        res_df = pd.read_csv(csv_file)
        assert len(res_df) == 1
        assert res_df.iloc[0]["exp_id"] == "EXP-003"
        assert res_df.iloc[0]["model"] == "AdaptiveGCN"
        assert pytest.approx(res_df.iloc[0]["mean_rmse"], 0.1) == 44.8
        assert pytest.approx(res_df.iloc[0]["fold2_rmse"], 0.1) == 39.9
        
        # Log second experiment (append)
        log_experiment_result("EXP-004", DummyCFG, df_metrics, csv_path=csv_file)
        res_df2 = pd.read_csv(csv_file)
        assert len(res_df2) == 2
        assert res_df2.iloc[1]["exp_id"] == "EXP-004"
