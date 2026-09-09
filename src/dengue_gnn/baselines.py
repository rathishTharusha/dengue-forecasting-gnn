"""Non-graph baselines for the Comparative Analysis.

The handout's paper structure requires a comparison against existing methods.
The Phase-2 paper declined to compare numerically against Weng et al., correctly,
because their static 70/30 split is not commensurable with rolling-origin. The
answer to that is not to skip the comparison but to **re-run the competitors under
our protocol**, which is what this module exists for.

Everything here shares the graph models' machinery exactly: the same
``_build_fold`` windows, the same residual-over-persistence target in normalised
log space, the same ``to_counts`` inverse transform, the same ``_rows_from``
scoring. A baseline that differed in any of those would be measuring a different
quantity, which is finding F6 in a different costume.

Models, cheapest first:

``seasonal_naive``
    Last year's value for the same week. Dengue is strongly seasonal, so this is
    the second naive competitor persistence cannot express, and it is free.
``ridge``
    Linear multi-output regression on the same 33-dimensional node window.
    Establishes how much of the signal is linear before any deep model is credited.
``random_forest`` / ``xgboost``
    Tabular learners over pooled district-weeks. These are what the proposal
    promised and what Weng et al. report as strong on this feature set.
``lstm``
    Per-node sequence model with no graph at all -- isolates what the *graph*
    contributes, as distinct from what the deep model contributes.

None of these see the adjacency. That is the point: together with the dense
fixed-graph control they bracket the graph's contribution from below.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from dengue_gnn.experiment import Config, FoldData, _build_fold, _rows_from, fold_split
from dengue_gnn.metrics import score

__all__ = ["BASELINE_KINDS", "run_baseline_single"]

BASELINE_KINDS = ("seasonal_naive", "ridge", "random_forest", "xgboost", "lstm")


def _flatten(x: torch.Tensor, y: torch.Tensor, p: torch.Tensor):
    """(windows, nodes, feat) -> (windows*nodes, feat) for tabular learners.

    Districts are pooled into one training set rather than fitted separately.
    With ~180 windows per fold a per-district model would see ~180 samples, which
    is not enough; pooling is also what the reference implementation does.
    """
    n_win, n_nodes, n_feat = x.shape
    xf = x.reshape(n_win * n_nodes, n_feat).numpy()
    yf = (y - p).reshape(n_win * n_nodes, y.shape[-1]).numpy()  # residual target
    return xf, yf


def _unflatten(pred_flat: np.ndarray, n_win: int, n_nodes: int, horizon: int) -> torch.Tensor:
    return torch.tensor(pred_flat.reshape(n_win, n_nodes, horizon), dtype=torch.float)


def _seasonal_naive(raw: np.ndarray, cfg: Config, test_ids: list[int], period: int = 52):
    """Forecast week ``t+h`` as the observed value at ``t+h-period``.

    Falls back to the persistence value when the lag reaches before the start of
    the series, which only affects the earliest origins.
    """
    cases = raw[..., cfg.cases_idx]
    preds, truths = [], []
    for i in test_ids:
        row = []
        for h in range(cfg.horizon):
            src = i + h - period
            row.append(cases[src] if src >= 0 else cases[i - 1])
        preds.append(np.stack(row, axis=1))
        truths.append(cases[i : i + cfg.horizon].T)
    return np.stack(preds), np.stack(truths)


class _LSTM(nn.Module):
    """Per-node LSTM over the window. No graph, by design."""

    def __init__(self, n_feat: int, window: int, hidden: int, horizon: int, dropout: float):
        super().__init__()
        self.window, self.n_feat = window, n_feat
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, nodes, window*feat) laid out feature-major per timestep,
        # matching _build_fold's transpose((1, 2, 0)).reshape(N, F*W).
        b, n, _ = x.shape
        seq = x.reshape(b * n, self.n_feat, self.window).transpose(1, 2)
        out, _ = self.lstm(seq)
        return self.head(out[:, -1]).reshape(b, n, -1)


def _fit_lstm(fold: FoldData, cfg: Config, n_feat: int, device) -> _LSTM:
    model = _LSTM(n_feat, cfg.window, cfg.hidden, cfg.horizon, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()
    best, best_state, wait = float("inf"), None, 0

    for _ in range(cfg.epochs):
        model.train()
        for i in torch.randperm(fold.x_train.shape[0]):
            opt.zero_grad()
            out = model(fold.x_train[i].unsqueeze(0).to(device)).squeeze(0)
            target = fold.y_train[i] - fold.p_train[i] if cfg.residual else fold.y_train[i]
            loss = mse(out, target.to(device))
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(fold.x_val.shape[0]):
                o = model(fold.x_val[i].unsqueeze(0).to(device)).squeeze(0).cpu()
                preds.append(fold.to_counts(o + fold.p_val[i] if cfg.residual else o))
            truth = torch.stack([fold.to_counts(fold.y_val[i]) for i in range(fold.y_val.shape[0])])
        val = score(torch.stack(preds).numpy(), truth.numpy())["RMSE"]

        if val < best - 1e-4:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def run_baseline_single(
    raw: np.ndarray,
    cfg: Config,
    fold_idx: int,
    seed: int,
    kind: str,
    device: torch.device | None = None,
) -> list[dict]:
    """Train and evaluate one non-graph baseline on one ``(fold, seed)``.

    Mirrors :func:`dengue_gnn.experiment.run_single` so both write the same row
    schema and both can be dispatched by the same worker pool.

    Raises:
        ValueError: If ``kind`` is not in :data:`BASELINE_KINDS`.
    """
    if kind not in BASELINE_KINDS:
        raise ValueError(f"unknown baseline {kind!r}; expected one of {BASELINE_KINDS}")

    import time

    device = device or torch.device("cpu")
    train_ids, val_ids, test_ids, origin = fold_split(raw, cfg, fold_idx)
    if not test_ids:
        return []

    base = {
        "label": kind,
        "fold": fold_idx + 1,
        "origin": origin,
        "seed": seed,
        "model": kind,
        "lambda_phys": 0.0,
        "gate_sigma": float("nan"),
        # These belong to the graph models but must be present so baseline rows
        # share one schema with them -- results/make_tables.py refuses to merge
        # files whose columns differ, which is the guard that stops incomparable
        # quantities being averaged together (review F9).
        "shrink_gamma": 1.0,  # no residual shrinkage is applied to baselines
        "temporal": "none",
        "augment": "none",
        "n_train_windows": 0,
        "n_real_windows": 0,
        "n_test_weeks": len(test_ids),
    }

    # Seasonal naive needs no fold machinery and no training.
    if kind == "seasonal_naive":
        pred, truth = _seasonal_naive(raw, cfg, test_ids)
        base |= {"n_params": 0, "train_seconds": 0.0, "infer_ms_per_window": 0.0}
        return _rows_from(pred, truth, base)

    (xtr, ytr, ptr), (xva, yva, pva), tmean, tstd = _build_fold(raw, cfg, train_ids, val_ids)
    _, (xte, yte, pte), _, _ = _build_fold(raw, cfg, train_ids, test_ids)
    fold = FoldData(xtr, ytr, ptr, xva, yva, pva, xte, yte, pte, tmean, tstd, test_ids)

    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    n_win, n_nodes, n_in = xte.shape
    n_feat = raw.shape[2]

    t0 = time.perf_counter()
    if kind == "lstm":
        model = _fit_lstm(fold, cfg, n_feat, device)
        train_seconds = time.perf_counter() - t0
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        t0 = time.perf_counter()
        model.eval()
        with torch.no_grad():
            out = torch.stack(
                [model(xte[i].unsqueeze(0).to(device)).squeeze(0).cpu() for i in range(n_win)]
            )
        infer_seconds = time.perf_counter() - t0
    else:
        xf, yf = _flatten(xtr, ytr, ptr)
        if kind == "ridge":
            from sklearn.linear_model import Ridge

            est = Ridge(alpha=1.0)  # deterministic; seed is irrelevant here
        elif kind == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            est = RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1
            )
        else:  # xgboost
            from xgboost import XGBRegressor

            est = XGBRegressor(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                n_jobs=1,
                multi_strategy="one_output_per_tree",
                verbosity=0,
            )
        est.fit(xf, yf)
        train_seconds = time.perf_counter() - t0
        n_params = _estimator_size(est, xf.shape[1], yf.shape[1])

        t0 = time.perf_counter()
        xtf = xte.reshape(n_win * n_nodes, n_in).numpy()
        out = _unflatten(np.asarray(est.predict(xtf)), n_win, n_nodes, cfg.horizon)
        infer_seconds = time.perf_counter() - t0

    if cfg.residual:
        out = out + pte
    pred = torch.stack([fold.to_counts(out[i]) for i in range(n_win)]).numpy()
    truth = torch.stack([fold.to_counts(yte[i]) for i in range(n_win)]).numpy()

    base |= {
        "n_params": n_params,
        "train_seconds": round(train_seconds, 3),
        "infer_ms_per_window": round(1000 * infer_seconds / max(n_win, 1), 4),
    }
    return _rows_from(pred, truth, base)


def _estimator_size(est, n_in: int, n_out: int) -> int:
    """A parameter count comparable to a network's, for the compute table.

    Linear models count coefficients. Tree ensembles have no weights, so total
    node count is reported instead -- the closest honest analogue of capacity,
    and labelled as such in the paper.
    """
    if hasattr(est, "coef_"):
        return int(np.asarray(est.coef_).size + np.asarray(getattr(est, "intercept_", 0)).size)
    if hasattr(est, "estimators_"):
        try:
            return int(sum(e.tree_.node_count for e in np.ravel(est.estimators_)))
        except AttributeError:
            pass
    booster = getattr(est, "get_booster", None)
    if booster is not None:
        try:
            return int(sum(t.count("\n") for t in booster().get_dump()))
        except Exception:
            pass
    return 0
