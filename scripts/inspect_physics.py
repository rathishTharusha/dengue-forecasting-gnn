import pandas as pd
from pathlib import Path

base = Path("analysis/results/beat_baseline")
for arch in ["A3TGCN", "STGAT", "AAGCN"]:
    p = base / f"physics_envelope_{arch}.csv"
    if not p.exists():
        continue
    df = pd.read_csv(p)
    print(f"\n==================== {arch} ====================")
    agg = df.groupby(["increment", "origin"])[["RMSE_clean", "MAE_clean"]].mean().reset_index()
    for inc in agg["increment"].unique():
        sub = agg[agg["increment"] == inc]
        vals = [round(x, 3) for x in sub["RMSE_clean"].tolist()]
        print(f"  {inc:16s} : mean={sub['RMSE_clean'].mean():.3f} | by origin={vals}")
