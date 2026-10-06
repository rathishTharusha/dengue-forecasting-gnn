"""EXP-063: the prospective test of docs/PROSPECTIVE_PLAN.md, in the plan's order.

    python seirgnn2/prospective.py dev --workers 6   # step 3: tuning on the purged nine origins
    python seirgnn2/prospective.py select            # step 3: write results/prospective_frozen.json
    # commit prospective_frozen.json, THEN parse the new weeks (step 4)
    python seirgnn2/prospective.py final --workers 6 # step 5: one run on 2024 W11 onward
    python seirgnn2/prospective.py stats             # step 6: P1, P2, Holm, sensitivity

``final`` refuses to run unless the frozen settings file is committed and clean,
so the settings provably predate the first look at the new weeks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import classical  # noqa: E402
import core  # noqa: E402
import corrected_data as cd  # noqa: E402

OUT = HERE / "results"
FROZEN = OUT / "prospective_frozen.json"
SEEDS = (0, 1, 2)

# ---- frozen arms and budget (docs/PROSPECTIVE_PLAN.md sections 5 and 6) ---------------------
_GWN = dict(backbone="adaptive", adj_init="gwn", adj_shared=True)
ARMS = {
    "Adaptive SEIR-GNN": dict(**_GWN, head="foi_res", lam_param="anchor", state_fit="encoder"),
    "Adaptive gated non-SEIR": dict(**_GWN, head="gated"),
    "Adaptive residual": dict(**_GWN, head="residual"),
}
FIXED = dict(loss="mse_z", dist="point", epochs=300, patience=40, batch_size=32, layers=2,
             dropout=0.1, weight_decay=1e-4)
GRID = tuple(dict(lr=lr, hidden=h) for lr in (1e-3, 3e-3) for h in (32, 64))
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
PRIMARY = (("P1", "Adaptive SEIR-GNN", "Adaptive gated non-SEIR"),
           ("P2", "Adaptive SEIR-GNN", "Persistence"))
BLOCK, SENSITIVITY_BLOCKS, REPS = 8, (4, 12), 10_000


def _tag(arm: str, g: dict) -> str:
    return f"{arm} | lr={g['lr']:g} hidden={g['hidden']}"


def _sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()


def _truth(cases: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.stack([cases[idx + h] for h in range(core.HORIZON)], -1)


# ---- step 3: development tuning -------------------------------------------------------------
def dev_configs() -> list[dict]:
    return [dict(name=_tag(arm, g), **cfg, **FIXED, **g, origins="frozen9_purged")
            for arm, cfg in ARMS.items() for g in GRID]


def dev_classical(data: cd.CorrectedData) -> list[dict]:
    """Seasonal naive and AR(3) at every alpha on the purged nine origins."""
    rows = []
    folds = core.build_folds(data.cases, data.missing, core.WINDOW, core.ORIGINS_F9,
                             core.TEST_FRAC_F9, purge=core.HORIZON - 1)
    for fold in folds:
        va, te = fold.idx["val"], fold.idx["test"]
        yv, yt = _truth(data.cases, va), _truth(data.cases, te)
        sv, _ = classical.seasonal_naive(data.cases, va)
        st, _ = classical.seasonal_naive(data.cases, te)
        rows.append(dict(name="Seasonal naive", origin=fold.origin, seed=-1,
                         val_RMSE=core.rmse(sv, yv), RMSE=core.rmse(st, yt)))
        for a in ALPHAS:
            coef = classical.ar3_fit(data.cases, fold.idx["train"], a)
            rows.append(dict(name=f"AR(3) ridge | alpha={a:g}", alpha=a, origin=fold.origin,
                             seed=-1,
                             val_RMSE=core.rmse(classical.ar3_forecast(data.cases, va, coef), yv),
                             RMSE=core.rmse(classical.ar3_forecast(data.cases, te, coef), yt)))
    return rows


def run_dev(workers: int) -> None:
    import sweep

    sweep.run(dev_configs(), "prospective_dev", workers)
    rows = dev_classical(cd.load())
    (OUT / "prospective_dev_classical.json").write_text(json.dumps(rows, indent=1), "utf-8")


def select() -> dict:
    """Validation-only choice of each learned arm's setting (plan section 6)."""
    neural = json.loads((OUT / "prospective_dev.json").read_text("utf-8"))
    clas = json.loads((OUT / "prospective_dev_classical.json").read_text("utf-8"))
    n_runs = len(core.ORIGINS_F9) * len(SEEDS)
    frozen = {"arms": {}, "dev_val_RMSE": {}}
    for arm in ARMS:
        best = None
        for g in GRID:
            vals = [r["val_RMSE"] for r in neural if r["name"] == _tag(arm, g)]
            if len(vals) != n_runs:
                raise SystemExit(f"{_tag(arm, g)}: {len(vals)} runs, expected {n_runs}")
            score = float(np.mean(vals))
            frozen["dev_val_RMSE"][_tag(arm, g)] = score
            if best is None or score < best[0]:
                best = (score, g)
        frozen["arms"][arm] = best[1]
    best = None
    for a in ALPHAS:
        vals = [r["val_RMSE"] for r in clas if r.get("alpha") == a]
        score = float(np.mean(vals))
        frozen["dev_val_RMSE"][f"AR(3) ridge | alpha={a:g}"] = score
        if best is None or score < best[0]:
            best = (score, a)
    frozen["ar_alpha"] = best[1]
    frozen["selected_at_commit"] = _sha()
    FROZEN.write_text(json.dumps(frozen, indent=1), "utf-8")
    print(json.dumps(frozen, indent=1))
    return frozen


# ---- step 5: the one prospective run ---------------------------------------------------------
def _require_committed() -> dict:
    rel = FROZEN.relative_to(REPO).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO,
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO).returncode == 0
    if not (tracked and clean):
        raise SystemExit(f"{rel} must be committed, unchanged, before the prospective run")
    return json.loads(FROZEN.read_text("utf-8"))


def _final_job(args):
    import torch
    import train
    arm, seed, setting, new_dir, epochs = args
    torch.set_num_threads(1)
    data, last_dev = cd.load_with_new_weeks(new_dir)
    fold = core.build_final_fold(data.cases, data.missing, last_dev)
    edge, fixed = core.adjacency(data.names)
    cfg = dict(ARMS[arm], **FIXED, **setting)
    if epochs is not None:
        cfg["epochs"] = epochs
    out, extra = train.run_fold(data, fold, seed=seed, edge=edge, fixed=fixed, keep=True, **cfg)
    return arm, seed, out, extra["test"]["pred"], extra["test"]["idx"]


def run_final(frozen: dict, new_dir: Path = cd.NEW_WEEKS, workers: int = 6,
              epochs: int | None = None, out_name: str = "prospective_final") -> dict:
    """Every arm once on the final split. ``epochs`` overrides only for smoke tests."""
    from dengue_gnn.metrics import score

    data, last_dev = cd.load_with_new_weeks(new_dir)
    fold = core.build_final_fold(data.cases, data.missing, last_dev)
    te = fold.idx["test"]
    if len(te) == 0:
        raise SystemExit("no test window survives the missing-week rule")
    truth = _truth(data.cases, te)

    preds: dict[str, list[np.ndarray]] = {}
    jobs = [(arm, s, frozen["arms"][arm], new_dir, epochs) for arm in ARMS for s in SEEDS]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_final_job, jobs))
    else:
        results = [_final_job(j) for j in jobs]
    for arm, _seed, _out, pred, idx in sorted(results, key=lambda r: (r[0], r[1])):
        if not np.array_equal(idx, te):
            raise RuntimeError(f"{arm}: test starts differ from the fold's")
        preds.setdefault(arm, []).append(pred)

    seasonal, fallback = classical.seasonal_naive(data.cases, te)
    coef = classical.ar3_fit(data.cases, fold.idx["train"], frozen["ar_alpha"])
    preds["Persistence"] = [classical.persistence(data.cases, te)]
    preds["Seasonal naive"] = [seasonal]
    preds["AR(3) ridge"] = [classical.ar3_forecast(data.cases, te, coef)]

    years = data.week_start.iloc[te].dt.year.to_numpy()
    rows, losses = [], {}
    for arm, plist in preds.items():
        per_seed = []
        for k, p in enumerate(plist):
            m = score(p, truth)
            m.update({f"RMSE_h{h + 1}": core.rmse(p[..., h], truth[..., h])
                      for h in range(core.HORIZON)})
            m.update({f"RMSE_{y}": core.rmse(p[years == y], truth[years == y])
                      for y in sorted(set(years.tolist()))})
            per_seed.append(m)
            rows.append(dict(name=arm, seed=SEEDS[k] if len(plist) > 1 else -1, **m))
        losses[arm] = np.mean([classical.per_start_loss(p, truth) for p in plist], 0).tolist()
    result = {"rows": rows, "losses": losses, "test_starts": te.tolist(),
              "n_train": len(fold.idx["train"]), "n_val": len(fold.idx["val"]),
              "n_test": len(te), "last_dev": last_dev, "seasonal_fallback_cells": fallback,
              "frozen": frozen, "commit": _sha()}
    OUT.mkdir(exist_ok=True)
    (OUT / f"{out_name}.json").write_text(json.dumps(result, indent=1), "utf-8")
    return result


# ---- step 6: statistics ----------------------------------------------------------------------
def stats(result: dict) -> dict:
    """P1 and P2 with Holm at block 8; block 4 and 12 and every other pair as secondary."""
    L = {k: np.asarray(v) for k, v in result["losses"].items()}
    out = {"primary": [], "sensitivity": [], "secondary": [],
           "underpowered": result["n_test"] < 52}
    prim = [classical.block_bootstrap(L[a], L[b], BLOCK, REPS) for _, a, b in PRIMARY]
    for (tag, a, b), r, padj in zip(PRIMARY, prim, classical.holm([r["p"] for r in prim]),
                                    strict=True):
        out["primary"].append(dict(test=tag, a=a, b=b, p_holm=padj, **r))
        for blk in SENSITIVITY_BLOCKS:
            out["sensitivity"].append(dict(test=tag, a=a, b=b,
                                           **classical.block_bootstrap(L[a], L[b], blk, REPS)))
    names = list(L)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out["secondary"].append(dict(a=a, b=b, **classical.block_bootstrap(L[a], L[b], BLOCK,
                                                                                REPS)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("dev", "select", "final", "stats"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.step == "dev":
        run_dev(args.workers)
    elif args.step == "select":
        select()
    elif args.step == "final":
        run_final(_require_committed(), workers=args.workers)
    else:
        res = json.loads((OUT / "prospective_final.json").read_text("utf-8"))
        st = stats(res)
        (OUT / "prospective_stats.json").write_text(json.dumps(st, indent=1), "utf-8")
        print(json.dumps(st["primary"], indent=1))


if __name__ == "__main__":
    main()
