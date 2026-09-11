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
        ("AAGCN", "det", "cases", "ens_raw"),
        ("AAGCN", "det", "cases", "ens_blend_raw"),
        ("ASTGCN", "det", "cases", "ens_blend_raw"),
        ("ASTGCN", "det", "cases", "ens_raw"),
        ("A3TGCN", "det", "cases", "ens_raw"),
        ("A3TGCN", "det", "cases", "ens_blend_raw"),
        ("A3TGCN+STGAT+ASTGCN", "det", "cases", "multi_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "cases", "multi_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "cases", "multi_blend_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "cases", "multi_opt_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "cases", "multi_super_raw"),
    ]

    print("\n=== Comparison of Key Candidates vs Floor ===")
    print(f"{'Arch':24s} {'Arm':16s} {'RMSE':>7s} {'vs Floor':>9s} {'Wins':>6s} {'p-val':>8s} {'Sig (<0.05)':>12s}")
    print("-" * 88)

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
        print(f"{arch:24s} {arm:16s} {v.mean():7.3f} {diff.mean():+9.3f} {wins:2d}/{n:2d} {p:8.4f} {sig:>12s}")

    # Now let's compare top contenders directly against A3TGCN baseline!
    a3tgcn_ens = df[(df["arch"] == "A3TGCN") & (df["head"] == "det") & (df["features"] == "cases") & (df["arm"] == "ens_raw")].groupby("origin")["RMSE_clean"].mean()

    print("\n=== Direct Pairwise Comparisons against A3TGCN Baseline (27.041 RMSE) ===")
    print(f"{'Challenger Arch':24s} {'Arm':16s} {'Challenger':>10s} {'A3TGCN':>8s} {'Diff':>8s} {'Wins':>6s} {'p-val':>8s}")
    print("-" * 88)

    challengers = [
        ("AAGCN", "det", "cases", "ens_raw"),
        ("AAGCN", "det", "cases", "ens_blend_raw"),
        ("ASTGCN", "det", "cases", "ens_raw"),
        ("ASTGCN", "det", "cases", "ens_blend_raw"),
        ("A3TGCN+STGAT+ASTGCN", "det", "cases", "multi_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "cases", "multi_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "cases", "multi_super_raw"),
    ]

    for arch, head, feat, arm in challengers:
        sub = df[(df["arch"] == arch) & (df["head"] == head) & (df["features"] == feat) & (df["arm"] == arm)]
        if sub.empty:
            continue
        c_by_origin = sub.groupby("origin")["RMSE_clean"].mean()
        common = sorted(set(c_by_origin.index).intersection(a3tgcn_ens.index))
        v_c = c_by_origin.loc[common].values
        v_a3 = a3tgcn_ens.loc[common].values
        d = v_c - v_a3
        t, p = stats.ttest_rel(v_c, v_a3)
        wins = np.sum(d < 0)
        n = len(common)
        print(f"{arch:24s} {arm:16s} {v_c.mean():10.3f} {v_a3.mean():8.3f} {d.mean():+8.3f} {wins:2d}/{n:2d} {p:8.4f}")

if __name__ == "__main__":
    main()
