"""Measure a candidate regulariser against the data loss, before sweeping lambda.

Run this **before** any lambda sweep on a new loss term. It is the check that
Phase 2 skipped, and skipping it cost the whole spatial-regularisation result:
on raw counts the smoothness penalty was ~17,700x the data loss, so every
non-zero lambda switched the data term off entirely and the sweep measured
nothing but degrees of not-fitting (docs/PHASE2_REVIEW.md F5, lesson LL-019).

A regulariser is a *weight* only when its magnitude is within an order of
magnitude or two of the data loss. Outside that range lambda is a switch.

Usage:
    python scripts/measure_loss_scale.py                # built-in terms
    python scripts/measure_loss_scale.py --fold 4 --n 120

For a new term (the SEIR-SEI residual, say), add it to TERMS below as a callable
taking ``(pred_counts, pred_log, adj)`` and returning a scalar tensor.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from dengue_gnn.experiment import (  # noqa: E402
    Config,
    FoldData,
    _build_fold,
    fold_split,
    load_dataset,
)
from dengue_gnn.losses import nonnegativity_loss, smoothness_loss  # noqa: E402
from dengue_gnn.models import AdaptiveGCN  # noqa: E402

NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"

#: name -> f(pred_counts, pred_log, adj) -> scalar tensor.
#: Add the mechanistic residual here once it exists; see docs/PHASE3_PLAN.md 5.5.
TERMS = {
    "L_smooth (counts)": lambda c, lg, a: smoothness_loss(c, a),
    "L_smooth (log1p)": lambda c, lg, a: smoothness_loss(lg, a),
    "L_cons (counts)": lambda c, lg, a: nonnegativity_loss(c),
    "L_cons (log1p)": lambda c, lg, a: nonnegativity_loss(lg),
}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=4, help="fold index (0-based)")
    ap.add_argument("--n", type=int, default=80, help="training windows to average over")
    args = ap.parse_args()

    torch.set_num_threads(1)
    raw, adj, _ = load_dataset(NPY, ADJ)
    cfg = Config()
    train_ids, val_ids, _, origin = fold_split(raw, cfg, min(args.fold, len(cfg.origins) - 1))
    (xtr, ytr, ptr), (xva, yva, pva), tmean, tstd = _build_fold(raw, cfg, train_ids, val_ids)
    fold = FoldData(xtr, ytr, ptr, xva, yva, pva, xva, yva, pva, tmean, tstd, val_ids)

    torch.manual_seed(0)
    model = AdaptiveGCN(
        in_dim=xtr.shape[2],
        hidden=cfg.hidden,
        horizon=cfg.horizon,
        n_nodes=raw.shape[1],
        adj_fixed=adj,
    )
    mse = nn.MSELoss()

    data_losses: list[float] = []
    term_values: dict[str, list[float]] = {k: [] for k in TERMS}

    with torch.no_grad():
        for i in range(min(args.n, xtr.shape[0])):
            out = model(xtr[i].unsqueeze(0)).squeeze(0)
            data_losses.append(float(mse(out, ytr[i] - ptr[i])))
            counts = fold.to_counts(out + ptr[i]).unsqueeze(0)
            log_counts = torch.log1p(counts)
            a = model.blended_adjacency()
            for name, fn in TERMS.items():
                term_values[name].append(float(fn(counts, log_counts, a)))

    data = st.mean(data_losses)
    print(f"fold {args.fold} (origin {origin:.3f}), {len(data_losses)} windows, untrained model\n")
    print(f"  {'term':22s} {'magnitude':>12s} {'ratio to L_data':>17s}   verdict")
    print(f"  {'L_data (MSE)':22s} {data:12.4f} {1.0:17.1f}   —")
    for name, vals in term_values.items():
        m = st.mean(vals)
        ratio = m / data if data else float("inf")
        if m == 0.0:
            verdict = "INERT — cannot bind"
        elif ratio > 100:
            verdict = "TOO LARGE — lambda is a switch"
        elif ratio < 0.01:
            verdict = "TOO SMALL — lambda cannot bite"
        else:
            verdict = "usable"
        print(f"  {name:22s} {m:12.4f} {ratio:17.1f}   {verdict}")

    print("\n  Suggested lambda grid for a usable term: the values that put the")
    print("  regulariser at roughly 1%, 10% and 50% of the total loss:")
    for name, vals in term_values.items():
        m = st.mean(vals)
        if m <= 0:
            continue
        grid = [round(frac / (1 - frac) * data / m, 6) for frac in (0.01, 0.10, 0.50)]
        print(f"    {name:22s} {grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
