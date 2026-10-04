"""D7: a simple 95% interval built from persistence residuals, on the legacy-array folds.

The Gaussian-head intervals (improved_sweep.json: PICP, MPIW) were produced on the
legacy benchmark array, three folds (origins 0.55 / 0.70 / 0.85, 68 test windows
each). To compare like with like we rebuild exactly those folds from the array and
form two intervals that need no network:

  additive   persistence forecast + the 2.5 / 97.5 percentiles of the training-window
             residuals (y[t+h] - y[t-1]), one pair per horizon, pooled over districts
  scaled     the same on residuals divided by sqrt(1 + y[t-1]), multiplied back

The lower end is clipped at zero. Coverage is the share of test cells inside, width the
mean upper minus lower. The fold logic mirrors analysis/lib/adaptive.py::build_folds.
The script checks that its persistence RMSE equals the saved one (improved_sweep.json).
"""
from __future__ import annotations

import numpy as np
from common import IMPROVED, LEGACY, RES, load_json

WINDOW, HORIZON, VAL_WEEKS = 3, 3, 30
ORIGINS, TEST_FRAC = (0.55, 0.70, 0.85), 0.15


def folds(n_weeks: int):
    ids = list(range(WINDOW, n_weeks - HORIZON))
    for origin in ORIGINS:
        cut = int(origin * len(ids))
        end = int(min(origin + TEST_FRAC, 1.0) * len(ids))
        yield origin, ids[: cut - VAL_WEEKS], ids[cut:end]


def pairs(cases: np.ndarray, ids):
    last = np.stack([cases[i - 1] for i in ids])                      # (K, N)
    truth = np.stack([cases[i : i + HORIZON].T for i in ids])         # (K, N, H)
    return last, truth


def main() -> None:
    cases = np.load(LEGACY)[:, :, 5].astype(float)
    saved = {r["origin"]: r["RMSE"] for r in load_json(IMPROVED) if r["arch"] == "persistence"}
    out = {"folds": []}
    for origin, train, test in folds(cases.shape[0]):
        ltr, ytr = pairs(cases, train)
        lte, yte = pairs(cases, test)
        pred = np.repeat(lte[:, :, None], HORIZON, axis=2)
        rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
        assert abs(rmse - saved[origin]) < 1e-6, (origin, rmse, saved[origin])
        row = {"origin": origin, "n_test_windows": len(test), "persistence_rmse": rmse}
        res_tr = ytr - ltr[:, :, None]
        for kind in ("additive", "scaled"):
            cov, wid = [], []
            for h in range(HORIZON):
                if kind == "additive":
                    lo, hi = np.percentile(res_tr[:, :, h], [2.5, 97.5])
                    lower = np.clip(pred[:, :, h] + lo, 0, None)
                    upper = pred[:, :, h] + hi
                else:
                    z = res_tr[:, :, h] / np.sqrt(1.0 + ltr)
                    lo, hi = np.percentile(z, [2.5, 97.5])
                    s = np.sqrt(1.0 + lte)
                    lower = np.clip(pred[:, :, h] + lo * s, 0, None)
                    upper = pred[:, :, h] + hi * s
                cov.append(np.mean((yte[:, :, h] >= lower) & (yte[:, :, h] <= upper)))
                wid.append(np.mean(upper - lower))
            row[kind] = {"coverage": float(np.mean(cov)), "width": float(np.mean(wid))}
        out["folds"].append(row)
    for kind in ("additive", "scaled"):
        out[kind] = {k: float(np.mean([f[kind][k] for f in out["folds"]])) for k in ("coverage", "width")}

    # Gaussian head (saved): mean over origins and seeds per architecture.
    rec = [r for r in load_json(IMPROVED) if r["increment"] == "probabilistic" and "PICP" in r]
    gauss = {}
    for a in sorted({r["arch"] for r in rec}):
        rows = [r for r in rec if r["arch"] == a]
        gauss[a] = {"coverage": float(np.mean([r["PICP"] for r in rows])),
                    "width": float(np.mean([r["MPIW"] for r in rows])), "n": len(rows)}
    out["gaussian_head"] = gauss
    out["persistence_check"] = "persistence RMSE per fold equals improved_sweep.json to 1e-6"
    RES.mkdir(exist_ok=True)
    (RES / "interval_baseline.json").write_text(__import__("json").dumps(out, indent=2), encoding="utf-8")
    print(__import__("json").dumps(out, indent=2))


if __name__ == "__main__":
    main()
