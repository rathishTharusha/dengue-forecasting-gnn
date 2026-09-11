"""In-depth analysis of beating persistence floor and A3TGCN baseline."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DIR = Path("analysis/results/beat_baseline")

def load_records():
    records = []
    for p in sorted(DIR.glob("beat_*.json")):
        data = json.load(open(p, encoding="utf-8"))
        for r in data:
            r.setdefault("head", "det")
            r.setdefault("features", "cases")
            r["source"] = p.stem
            records.append(r)
    return records

def main():
    records = load_records()
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records across {df['source'].nunique()} files.")
    
    # Filter for key arms
    floor_df = df[df["arch"] == "persistence"].groupby("origin")["RMSE_clean"].mean()
    print("\n=== Persistence Floor by Origin ===")
    for o, v in floor_df.items():
        print(f"  Origin {o:.2f}: {v:.3f}")
    print(f"  Mean Floor: {floor_df.mean():.3f}")

    # Compare key candidates against floor
    key_candidates = [
        ("ASTGCN", "det", "cases", "ens_blend_raw"),
        ("ASTGCN", "det", "cases", "ens_raw"),
        ("ASTGCN", "det", "cases", "blend_raw"),
        ("ASTGCN", "det", "cases", "raw"),
        ("A3TGCN", "det", "cases", "ens_raw"),
        ("A3TGCN", "det", "cases", "ens_blend_raw"),
        ("A3TGCN", "det", "cases", "raw"),
        ("A3TGCN", "det", "cases", "blend_raw"),
        ("A3TGCN+STGAT+ASTGCN", "det", "cases", "multi_raw"),
    ]

    print("\n=== Comparison of Key Candidates vs Floor ===")
    print(f"{'Arch':22s} {'Arm':16s} {'RMSE':>7s} {'vs Floor':>9s} {'Wins':>6s} {'p-val':>8s} {'Sig (<0.05)':>12s}")
    print("-" * 85)

    for arch, head, feat, arm in key_candidates:
        sub = df[(df["arch"] == arch) & (df["head"] == head) & (df["features"] == feat) & (df["arm"] == arm)]
        if sub.empty:
            continue
        by_origin = sub.groupby("origin")["RMSE_clean"].mean()
        common_origins = sorted(set(by_origin.index).intersection(floor_df.index))
        v = by_origin.loc[common_origins].values
        f = floor_df.loc[common_origins].values
        diff = v - f
        t, p = stats.ttest_rel(v, f)
        wins = np.sum(diff < 0)
        n = len(common_origins)
        sig = "YES (p<0.05)" if p < 0.05 and diff.mean() < 0 else "no"
        print(f"{arch:22s} {arm:16s} {v.mean():7.3f} {diff.mean():+9.3f} {wins:2d}/{n:2d} {p:8.4f} {sig:>12s}")

    # Now let's compare ASTGCN directly against A3TGCN!
    print("\n=== Direct Comparison: ASTGCN vs A3TGCN ===")
    astgcn_ens = df[(df["arch"] == "ASTGCN") & (df["head"] == "det") & (df["features"] == "cases") & (df["arm"] == "ens_raw")].groupby("origin")["RMSE_clean"].mean()
    a3tgcn_ens = df[(df["arch"] == "A3TGCN") & (df["head"] == "det") & (df["features"] == "cases") & (df["arm"] == "ens_raw")].groupby("origin")["RMSE_clean"].mean()
    
    common = sorted(set(astgcn_ens.index).intersection(a3tgcn_ens.index))
    v_ast = astgcn_ens.loc[common].values
    v_a3 = a3tgcn_ens.loc[common].values
    d = v_ast - v_a3
    t, p = stats.ttest_rel(v_ast, v_a3)
    wins = np.sum(d < 0)
    print(f"ASTGCN ens_raw mean : {v_ast.mean():.3f}")
    print(f"A3TGCN ens_raw mean : {v_a3.mean():.3f}")
    print(f"Difference (AST - A3): {d.mean():+.3f} RMSE")
    print(f"ASTGCN wins against A3TGCN: {wins}/{len(common)} origins (p = {p:.4f})")
    for o in common:
        print(f"  Origin {o:.2f}: ASTGCN={astgcn_ens[o]:.3f} vs A3TGCN={a3tgcn_ens[o]:.3f} (diff={astgcn_ens[o]-a3tgcn_ens[o]:+.3f})")

if __name__ == "__main__":
    main()
