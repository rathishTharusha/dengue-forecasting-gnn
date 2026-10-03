"""Does any spatial structure help, once the border graph is replaced?

The project has tested the border graph twice and found it contributes nothing:
the adaptive-graph experiment, and the frozen-encoder control in EXP-052. Neither
tested a graph built on *movement*, which is what actually carries dengue between
districts -- *Aedes* mosquitoes travel only a few hundred metres.

Four arms, identical except for the spatial structure. Everything else is the v2
configuration of `run_s5_seir_gnn_v2.py`, and the SEIR simulator stays in the
prediction path throughout.

| arm | message passing | import term in lambda |
|---|---|---|
| `none` | border graph | none (the v2 default) |
| `geo` | border graph | border neighbours |
| `gravity` | border graph | gravity-weighted |
| `gravity_graph` | gravity top-k edges | gravity-weighted |

The import term is `lambda_i = beta_i * (I_i + alpha * sum_j W_ij I_j)` with
`alpha >= 0` learned, so a model that finds no use for its neighbours can drive
alpha to zero and recover the `none` arm exactly. That is the point: the arm can
refuse the structure.

Protocol: `rebuilt`, 9 disjoint origins, 3 seeds, window 3 -> horizon 3,
training-only normalisation, squared error, early stopping on validation RMSE.

Usage::

    python analysis/_build/build_gravity_graph.py     # writes the matrix first
    python analysis/_build/run_spatial_test.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

REPO = Path(__file__).resolve().parent.parent.parent
for _p in (REPO / "analysis" / "lib", REPO / "analysis" / "_build", REPO / "src"):
    sys.path.insert(0, str(_p))

import corrected_data as cd  # noqa: E402
import run_corrected_benchmark as rcb  # noqa: E402
import run_s5_seir_gnn_v2 as v2  # noqa: E402

GRAVITY = REPO / "data" / "external" / "district_gravity_matrix.csv"
ARMS = {
    "none":          dict(coupling="implicit", matrix="geo",     graph="geo"),
    "geo":           dict(coupling="explicit", matrix="geo",     graph="geo"),
    "gravity":       dict(coupling="explicit", matrix="gravity", graph="geo"),
    "gravity_graph": dict(coupling="explicit", matrix="gravity", graph="gravity"),
}
TOPK = 4


def gravity_matrix(names: list[str]) -> np.ndarray:
    if not GRAVITY.exists():
        raise SystemExit("run analysis/_build/build_gravity_graph.py first")
    df = pd.read_csv(GRAVITY, index_col=0)
    return df.loc[names, names].to_numpy(dtype=np.float64)


def topk_edges(w: np.ndarray, k: int = TOPK) -> np.ndarray:
    """Symmetric edge_index from each district's k strongest links."""
    n = w.shape[0]
    keep = np.zeros_like(w, dtype=bool)
    for i in range(n):
        for j in np.argsort(w[i])[::-1][:k]:
            if i != j:
                keep[i, j] = keep[j, i] = True
    src, dst = np.nonzero(keep)
    return np.stack([src, dst])


def run_job(job):
    arm, seed = job["arm"], job["seed"]
    spec, cfg = ARMS[arm], v2.ARMS["v2"]
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    ft = torch.tensor(job["features"], dtype=torch.float32)
    model = v2.SEIRGNNv2("STGAT", job["features"].shape[-1],
                         torch.tensor(job["edge_index"], dtype=torch.long),
                         torch.tensor(job["matrix"], dtype=torch.float32),
                         coupling=spec["coupling"], mass_action=True, per_week=True,
                         learn_rho=True, learn_rates=True, mod_clamp=cfg["mod_clamp"],
                         learn_state=True)
    opt = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    def prep(idx):
        st0, s_, oi, oe, pop, y = v2.initial_state(job["cases"], job["population"], idx,
                                                   True, 0.0)
        return ft[idx], st0, pop, y, s_, oi, oe

    tr, va, te = prep(job["train_idx"]), prep(job["val_idx"]), prep(job["test_idx"])
    fwd = lambda b: model.to_cases(model(b[0], model.build_state(b[4], b[5], b[6])), b[2])  # noqa: E731

    best, bw, wait = float("inf"), None, 0
    for e in range(cfg["epochs"]):
        model.train()
        opt.zero_grad()
        torch.mean((fwd(tr) - tr[3]) ** 2).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if e % 3 == 0:
            model.eval()
            with torch.no_grad():
                v = torch.sqrt(torch.mean((fwd(va) - va[3]) ** 2)).item()
            if v < best - 1e-5:
                best, wait = v, 0
                bw = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 20:
                    break
    if bw:
        model.load_state_dict(bw)
    model.eval()
    with torch.no_grad():
        tp = fwd(te)
        rec = {"arm": arm, "origin": job["origin"], "seed": seed, "val_RMSE": best,
               "test_RMSE": torch.sqrt(torch.mean((tp - te[3]) ** 2)).item(),
               "test_MAE": torch.mean((tp - te[3]).abs()).item(),
               "alpha": float(torch.relu(model.alpha)) if hasattr(model, "alpha") else 0.0,
               "elapsed": round(time.time() - t0, 1)}
    print(f"  [OK] {arm:14s} o{job['origin']} s{seed} | val {rec['val_RMSE']:6.2f} "
          f"| test {rec['test_RMSE']:6.2f} | alpha {rec['alpha']:.4f}", flush=True)
    return rec


def perm_p(x):
    x = np.asarray(x)
    f = np.array(list(itertools.product([-1, 1], repeat=len(x))))
    return float(np.mean(np.abs((f * x).mean(1)) >= abs(x.mean())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs="*", default=list(ARMS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(REPO / "analysis" / "results" / "seir_gnn" / "s5_v2"
                                        / "spatial_test_runs.csv"))
    args = ap.parse_args()

    data = cd.load()
    cases, adj, artifact, missing, _ = rcb.prepare("rebuilt")
    names = sorted(json.loads((REPO / "notebooks" / "baseline"
                               / "sri_lanka_adj_list.json").read_text(encoding="utf-8")))
    grav = gravity_matrix(names)
    src, dst = np.nonzero(adj)
    geo_edges = np.stack([src, dst])
    grav_edges = topk_edges(grav)
    print(f"border graph: {geo_edges.shape[1]} directed edges | "
          f"gravity top-{TOPK}: {grav_edges.shape[1]}")

    folds = v2.build_folds(cases, missing, "9origin")
    jobs = []
    for arm in args.arm:
        spec = ARMS[arm]
        for fold in folds:
            feats = v2.features_train_only(data, "cases", fold)
            for seed in (0, 1, 2):
                jobs.append({"arm": arm, "seed": seed, "origin": fold.origin,
                             "train_idx": np.asarray(fold.train_index),
                             "val_idx": np.asarray(fold.val_index),
                             "test_idx": np.asarray(fold.test_index),
                             "cases": cases, "features": feats,
                             "population": data.population,
                             "edge_index": grav_edges if spec["graph"] == "gravity" else geo_edges,
                             "matrix": grav if spec["matrix"] == "gravity" else adj})

    print(f"{len(jobs)} jobs / {args.workers} workers", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run_job, j) for j in jobs]):
            try:
                out.append(f.result())
            except Exception as exc:
                import traceback
                print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()

    df = pd.DataFrame(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\n=== 9 disjoint origins x 3 seeds ===")
    print(df.groupby("arm")[["val_RMSE", "test_RMSE", "test_MAE", "alpha"]]
            .mean().round(3).sort_values("val_RMSE").to_string())
    if "none" in df.arm.values:
        ref = df[df.arm == "none"].groupby("origin")["test_RMSE"].mean()
        print("\n=== against no spatial coupling, paired and clustered by origin ===")
        for arm, g in df.groupby("arm"):
            if arm == "none":
                continue
            d = g.groupby("origin")["test_RMSE"].mean() - ref
            print(f"  {arm:14s} mean {d.mean():+6.2f} | better {int((d < 0).sum())}/9 "
                  f"| p = {perm_p(d.values):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
