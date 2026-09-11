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
        stem = p.stem.lower()
        default_head = "gauss" if "_gauss" in stem else ("nb" if "_nb" in stem else "det")
        default_feats = "climate" if "_climate" in stem else ("causal" if "_causal" in stem else "cases")
        default_phys = "spatial" if "_spatial" in stem else "none"
        for r in data:
            if not r.get("head"):
                r["head"] = default_head
            if not r.get("features"):
                r["features"] = default_feats
            if not r.get("physics") or r.get("physics") == "base":
                r["physics"] = default_phys
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
        ("AAGCN", "det", "none", "cases", "ens_raw"),
        ("AAGCN", "det", "none", "cases", "ens_blend_raw"),
        ("ASTGCN", "det", "none", "cases", "ens_blend_raw"),
        ("ASTGCN", "det", "none", "cases", "ens_raw"),
        ("A3TGCN", "det", "none", "cases", "ens_raw"),
        ("A3TGCN", "det", "none", "cases", "ens_blend_raw"),
        ("A3TGCN+STGAT+ASTGCN", "det", "none", "cases", "multi_raw"),
        ("AAGCN+ASTGCN", "det", "none", "cases", "multi_raw"),
        ("AAGCN+ASTGCN", "det", "none", "cases", "multi_opt_raw"),
        ("AAGCN+ASTGCN", "det", "none", "cases", "multi_blend_raw"),
        ("AAGCN+ASTGCN", "det", "none", "cases", "multi_super_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "none", "cases", "multi_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "none", "cases", "multi_blend_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "none", "cases", "multi_opt_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "none", "cases", "multi_super_raw"),
        # Quad-model ensemble (4 architectures)
        ("AAGCN+ASTGCN+A3TGCN+STGAT", "det", "none", "cases", "multi_raw"),
        ("AAGCN+ASTGCN+A3TGCN+STGAT", "det", "none", "cases", "multi_opt_raw"),
        ("AAGCN+ASTGCN+A3TGCN+STGAT", "det", "none", "cases", "multi_blend_raw"),
        ("AAGCN+ASTGCN+A3TGCN+STGAT", "det", "none", "cases", "multi_super_raw"),
        # Physics-informed candidates (Spatial Dirichlet regularizer)
        ("ASTGCN", "det", "spatial", "cases", "ens_raw"),
        ("A3TGCN", "det", "spatial", "cases", "ens_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "spatial", "cases", "multi_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "spatial", "cases", "multi_opt_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "spatial", "cases", "multi_blend_raw"),
    ]

    print("\n=== Comparison of Key Candidates vs Floor ===")
    print(f"{'Arch':24s} {'Phys':8s} {'Arm':16s} {'RMSE':>7s} {'vs Floor':>9s} {'Wins':>6s} {'p-val':>8s} {'Sig (<0.05)':>12s}")
    print("-" * 96)

    for arch, head, phys, feat, arm in key_candidates:
        sub = df[(df["arch"] == arch) & (df["head"] == head) & (df["physics"] == phys) & (df["features"] == feat) & (df["arm"] == arm)]
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
        label = f"{arch}"
        print(f"{label:24s} {phys:8s} {arm:16s} {v.mean():7.3f} {diff.mean():+9.3f} {wins:2d}/{n:2d} {p:8.4f} {sig:>12s}")

    # Now let's compare top contenders directly against A3TGCN unconstrained baseline (26.993 RMSE)!
    a3tgcn_sub = df[(df["arch"] == "A3TGCN") & (df["head"] == "det") & (df["physics"] == "none") & (df["features"] == "cases") & (df["arm"] == "ens_raw")]
    if a3tgcn_sub.empty:
        a3tgcn_sub = df[(df["arch"] == "A3TGCN") & (df["head"] == "det") & (df["features"] == "cases") & (df["arm"] == "ens_raw")]
    a3tgcn_ens = a3tgcn_sub.groupby("origin")["RMSE_clean"].mean()

    print(f"\n=== Direct Pairwise Comparisons against A3TGCN Baseline ({a3tgcn_ens.mean():.3f} RMSE) ===")
    print(f"{'Challenger Arch':24s} {'Phys':8s} {'Arm':16s} {'Challenger':>10s} {'A3TGCN':>8s} {'Diff':>8s} {'Wins':>6s} {'p-val':>8s}")
    print("-" * 96)

    challengers = [
        ("AAGCN", "det", "none", "cases", "ens_raw"),
        ("AAGCN", "det", "none", "cases", "ens_blend_raw"),
        ("ASTGCN", "det", "none", "cases", "ens_raw"),
        ("ASTGCN", "det", "none", "cases", "ens_blend_raw"),
        ("A3TGCN+STGAT+ASTGCN", "det", "none", "cases", "multi_raw"),
        ("AAGCN+ASTGCN", "det", "none", "cases", "multi_raw"),
        ("AAGCN+ASTGCN", "det", "none", "cases", "multi_opt_raw"),
        ("AAGCN+ASTGCN", "det", "none", "cases", "multi_blend_raw"),
        ("AAGCN+ASTGCN", "det", "none", "cases", "multi_super_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "none", "cases", "multi_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "none", "cases", "multi_opt_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "none", "cases", "multi_blend_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "none", "cases", "multi_super_raw"),
        # Quad-model challengers
        ("AAGCN+ASTGCN+A3TGCN+STGAT", "det", "none", "cases", "multi_raw"),
        ("AAGCN+ASTGCN+A3TGCN+STGAT", "det", "none", "cases", "multi_opt_raw"),
        ("AAGCN+ASTGCN+A3TGCN+STGAT", "det", "none", "cases", "multi_blend_raw"),
        ("AAGCN+ASTGCN+A3TGCN+STGAT", "det", "none", "cases", "multi_super_raw"),
        # Physics challengers
        ("ASTGCN", "det", "spatial", "cases", "ens_raw"),
        ("A3TGCN", "det", "spatial", "cases", "ens_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "spatial", "cases", "multi_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "spatial", "cases", "multi_opt_raw"),
        ("AAGCN+ASTGCN+A3TGCN", "det", "spatial", "cases", "multi_blend_raw"),
    ]

    for arch, head, phys, feat, arm in challengers:
        sub = df[(df["arch"] == arch) & (df["head"] == head) & (df["physics"] == phys) & (df["features"] == feat) & (df["arm"] == arm)]
        if sub.empty:
            continue
        by_origin = sub.groupby("origin")["RMSE_clean"].mean()
        common_origins = sorted(set(by_origin.index).intersection(a3tgcn_ens.index))
        v = by_origin.loc[common_origins].values
        b = a3tgcn_ens.loc[common_origins].values
        diff = v - b
        t, p = stats.ttest_rel(v, b)
        wins = np.sum(diff < 0)
        n = len(common_origins)
        print(f"{arch:24s} {phys:8s} {arm:16s} {v.mean():10.3f} {b.mean():8.3f} {diff.mean():+8.3f} {wins:2d}/{n:2d} {p:8.4f}")

if __name__ == "__main__":
    main()
