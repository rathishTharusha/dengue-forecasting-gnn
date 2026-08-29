"""Rolling-origin evaluation harness.

Ports the protocol from ``notebooks/baseline/dengue_baseline_GNN_v2.ipynb`` into
version-controlled code, so results can be regenerated instead of trusted.

The protocol is deliberately unchanged from Phase 1 -- same origins, same
window/horizon, same train-only normalisation, same batch-size-1 updates, same
early stopping. If any of that moved, Phase-1 and Phase-2 numbers would stop
being comparable and every row of the ablation table would need re-running.

One thing *is* changed, and it is a fix rather than a protocol change. The
notebook's ``rolling_origin`` averaged RMSE across seeds inside each fold and
stored only the mean, so the seed dimension was destroyed before it was ever
written down. That is why no Phase-2 result has a standard deviation and why no
significance test is possible on them (review finding F7). :func:`rolling_origin`
here emits one record per ``(fold, seed, horizon)`` and aggregates later, which
is also what ``results/README.md`` asks for.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dengue_gnn.losses import spatial_regularisation
from dengue_gnn.metrics import peak_week_error, score
from dengue_gnn.models import AdaptiveGCN, build_fixed_adjacency

__all__ = ["Config", "FoldData", "load_dataset", "rolling_origin", "train_fold"]


@dataclass
class Config:
    """Everything that defines a run. Serialised into every result row."""

    # data / protocol -- do not change without re-running every ablation row
    window: int = 3
    horizon: int = 3
    cases_idx: int = 5
    self_loops: bool = True
    origins: tuple[float, ...] = (0.55, 0.70, 0.85)
    test_frac: float = 0.15
    val_weeks: int = 30
    seeds: tuple[int, ...] = (0, 1, 2)

    # target handling
    residual: bool = True
    log_transform: bool = True

    # model
    hidden: int = 64
    dropout: float = 0.1
    emb_dim: int = 10
    use_adaptive: bool = True
    gate_init: float = 1.5

    # objective
    lambda_phys: float = 0.0
    cons_weight: float = 10.0
    reg_space: str = "log"  # "log" | "count" -- see _regularisation_target

    # optimisation
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 120
    patience: int = 25
    grad_clip: float = 5.0

    label: str = "run"
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("extra", None)
        d["origins"] = ",".join(str(o) for o in self.origins)
        d["seeds"] = ",".join(str(s) for s in self.seeds)
        return d


@dataclass
class FoldData:
    """One fold's tensors plus the inverse transform back to case counts."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    p_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    p_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    p_test: torch.Tensor
    target_mean: float
    target_std: float
    test_ids: list[int]

    def to_counts(self, normalised: torch.Tensor) -> torch.Tensor:
        """Invert normalisation and log1p, differentiably.

        Mirrors the notebook's ``inv``: ``expm1(clip(x*std + mean, 0, 12))``. The
        clip bounds predictions to ``e^12`` cases, which suppresses the occasional
        hallucinated spike; it is differentiable inside the range.
        """
        return torch.expm1(torch.clamp(normalised * self.target_std + self.target_mean, 0, 12))


def load_dataset(
    npy_path: str | Path,
    adj_path: str | Path,
    self_loops: bool = True,
) -> tuple[np.ndarray, torch.Tensor, list[str]]:
    """Load the feature array and geographic adjacency.

    Node order is ``sorted(adj_list)`` -- districts alphabetically -- matching the
    Phase-1 notebook. Axis 1 of the array is assumed to follow that same order;
    a mismatch silently scrambles the graph without raising anything.

    Returns:
        ``(raw, adj_fixed, district_names)`` where ``raw`` is
        ``(T, N, F)`` float32 with NaNs zeroed.
    """
    raw = np.nan_to_num(np.load(npy_path, allow_pickle=True)).astype(np.float32)
    with open(adj_path) as fh:
        adj_list = json.load(fh)

    names = sorted(adj_list)
    if raw.shape[1] != len(names):
        raise ValueError(f"array has {raw.shape[1]} nodes but adjacency has {len(names)} districts")
    adj = build_fixed_adjacency(adj_list, names, self_loops=self_loops, normalize=True)
    return raw, adj, names


def _build_fold(raw: np.ndarray, cfg: Config, train_ids: list[int], eval_ids: list[int]):
    """Window and normalise one train/eval split using train-only statistics."""
    n_nodes, n_feat = raw.shape[1], raw.shape[2]
    train_weeks = raw[: train_ids[-1]]

    fmean = train_weeks.reshape(-1, n_feat).mean(0)
    fstd = train_weeks.reshape(-1, n_feat).std(0) + 1e-6
    feat = (raw - fmean) / fstd

    cases = raw[..., cfg.cases_idx]
    if cfg.log_transform:
        base = np.log1p(cases)
        tbase = np.log1p(train_weeks[..., cfg.cases_idx])
        tmean, tstd = float(tbase.mean()), float(tbase.std() + 1e-6)
    else:
        base = cases
        tmean = float(train_weeks[..., cfg.cases_idx].mean())
        tstd = float(train_weeks[..., cfg.cases_idx].std() + 1e-6)
    tgt = (base - tmean) / tstd

    def make(ids):
        xs, ys, ps = [], [], []
        for i in ids:
            xi = np.transpose(feat[i - cfg.window : i], (1, 2, 0)).reshape(
                n_nodes, n_feat * cfg.window
            )
            xs.append(xi)
            ys.append(tgt[i : i + cfg.horizon].T)
            ps.append(np.repeat(tgt[i - 1].reshape(n_nodes, 1), cfg.horizon, axis=1))
        t = lambda a: torch.tensor(np.stack(a), dtype=torch.float)  # noqa: E731
        return t(xs), t(ys), t(ps)

    return make(train_ids), make(eval_ids), tmean, tstd


def _evaluate(model, x, y, p, fold: FoldData, cfg: Config, device) -> tuple:
    """Return (pred_counts, true_counts) as numpy, shape (n_windows, n_nodes, H)."""
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for i in range(x.shape[0]):
            out = model(x[i].unsqueeze(0).to(device)).squeeze(0).cpu()
            if cfg.residual:
                out = out + p[i]
            preds.append(fold.to_counts(out))
            truths.append(fold.to_counts(y[i]))
    return torch.stack(preds).numpy(), torch.stack(truths).numpy()


def _regularisation_target(fold: FoldData, normalised: torch.Tensor, space: str) -> torch.Tensor:
    """Put predictions into the space the spatial regulariser acts on.

    ``"count"`` -- raw cases. Semantically what the constraints describe, but
    measurement shows it is unusable as specified: on this data ``L_smooth`` on
    counts is ~4,800 against a data loss of ~0.27, a ratio of about 17,700. At
    the paper's smallest non-zero weight (λ=0.01) the regulariser is 99.4% of the
    total loss and the model stops fitting the data at all. Since the published
    λ sweep moved RMSE by only 0.07, the lost original cannot have been applying
    the constraint on this scale -- which corroborates review finding F5.

    ``"log"`` (default) -- ``log1p`` of the counts. Two reasons. Magnitudes land
    within an order of magnitude of the data loss, so λ behaves like a
    regularisation weight rather than a switch. And on a target with median 13
    and maximum 2,631, smoothness on raw counts is dominated entirely by the
    largest districts; in log space it penalises *relative* disagreement between
    neighbours, which is the property the constraint is actually appealing to.

    Note that the non-negativity term is inert either way: ``to_counts`` clamps
    its argument at 0 before ``expm1``, so predicted counts are non-negative by
    construction and ``L_cons`` is identically zero. The ``10.0 · L_cons`` term
    in the paper's equation (4) contributes nothing under any configuration.
    It is kept so the objective matches the published specification.
    """
    counts = fold.to_counts(normalised)
    if space == "count":
        return counts
    if space == "log":
        return torch.log1p(counts)
    raise ValueError(f"reg_space must be 'log' or 'count', got {space!r}")


def train_fold(fold: FoldData, cfg: Config, adj_fixed: torch.Tensor, device) -> AdaptiveGCN:
    """Train one model on one fold, early-stopping on validation RMSE."""
    n_nodes = fold.x_train.shape[1]
    model = AdaptiveGCN(
        in_dim=fold.x_train.shape[2],
        hidden=cfg.hidden,
        horizon=cfg.horizon,
        n_nodes=n_nodes,
        adj_fixed=adj_fixed,
        emb_dim=cfg.emb_dim,
        dropout=cfg.dropout,
        use_adaptive=cfg.use_adaptive,
        gate_init=cfg.gate_init,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()

    best, best_state, wait = float("inf"), None, 0
    for _ in range(cfg.epochs):
        model.train()
        for i in torch.randperm(fold.x_train.shape[0]):
            opt.zero_grad()
            out = model(fold.x_train[i].unsqueeze(0).to(device))
            target = fold.y_train[i] - fold.p_train[i] if cfg.residual else fold.y_train[i]
            loss = mse(out.squeeze(0), target.to(device))

            if cfg.lambda_phys > 0.0:
                normalised = out.squeeze(0)
                if cfg.residual:
                    normalised = normalised + fold.p_train[i].to(device)
                target = _regularisation_target(fold, normalised, cfg.reg_space)
                loss = loss + spatial_regularisation(
                    target.unsqueeze(0),
                    model.blended_adjacency(),
                    lambda_phys=cfg.lambda_phys,
                    cons_weight=cfg.cons_weight,
                )

            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        pred, truth = _evaluate(model, fold.x_val, fold.y_val, fold.p_val, fold, cfg, device)
        val_rmse = score(pred, truth)["RMSE"]
        if val_rmse < best - 1e-4:
            best = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def _persistence_counts(raw: np.ndarray, cfg: Config, test_ids: list[int]) -> tuple:
    """Naive baseline: next h weeks all equal this week's observed count."""
    pred = np.stack(
        [
            np.repeat(raw[i - 1, :, cfg.cases_idx : cfg.cases_idx + 1], cfg.horizon, axis=1)
            for i in test_ids
        ]
    )
    truth = np.stack([raw[i : i + cfg.horizon, :, cfg.cases_idx].T for i in test_ids])
    return pred, truth


def _rows_from(pred, truth, base: dict) -> list[dict]:
    """Emit one record per horizon plus an ``overall`` record."""
    rows = []
    n_h = pred.shape[-1]
    for h in range(n_h):
        s = score(pred[..., h], truth[..., h])
        rows.append(
            {
                **base,
                "horizon": h + 1,
                **{k.lower(): v for k, v in s.items()},
                "peak_week_err": peak_week_error(pred[:, :, h], truth[:, :, h]),
                "n_obs": int(pred[..., h].size),
            }
        )
    s = score(pred, truth)
    rows.append(
        {
            **base,
            "horizon": 0,  # 0 == pooled across horizons
            **{k.lower(): v for k, v in s.items()},
            "peak_week_err": float("nan"),
            "n_obs": int(pred.size),
        }
    )
    return rows


def rolling_origin(
    raw: np.ndarray,
    adj_fixed: torch.Tensor,
    cfg: Config,
    device: torch.device | None = None,
    include_persistence: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Run the full rolling-origin protocol.

    Returns:
        One record per ``(fold, seed, horizon)``, plus persistence records where
        requested. ``horizon=0`` means pooled across horizons. Nothing is
        averaged here -- aggregation happens at reporting time so that
        mean ± std and paired significance tests remain possible.
    """
    device = device or torch.device("cpu")
    n_weeks = raw.shape[0]
    ids = list(range(cfg.window, n_weeks - cfg.horizon))
    n = len(ids)
    rows: list[dict] = []

    for k, origin in enumerate(cfg.origins, start=1):
        cut = int(origin * n)
        tend = int(min(origin + cfg.test_frac, 1.0) * n)
        train_pool, test_ids = ids[:cut], ids[cut:tend]
        if not test_ids:
            continue
        train_ids, val_ids = train_pool[: -cfg.val_weeks], train_pool[-cfg.val_weeks :]

        (xtr, ytr, ptr), (xva, yva, pva), tmean, tstd = _build_fold(raw, cfg, train_ids, val_ids)
        _, (xte, yte, pte), _, _ = _build_fold(raw, cfg, train_ids, test_ids)

        fold = FoldData(xtr, ytr, ptr, xva, yva, pva, xte, yte, pte, tmean, tstd, test_ids)

        for seed in cfg.seeds:
            torch.manual_seed(seed)
            # Legacy global seeding on purpose: the Phase-1 notebook's set_seed()
            # did the same, and matching its RNG state is what keeps re-runs
            # comparable to the published baseline. Nothing here draws from numpy
            # today, but a future change that does must land on the same stream.
            np.random.seed(seed)  # noqa: NPY002
            model = train_fold(fold, cfg, adj_fixed, device)
            pred, truth = _evaluate(model, xte, yte, pte, fold, cfg, device)

            base = {
                "label": cfg.label,
                "fold": k,
                "origin": origin,
                "seed": seed,
                "model": "AdaptiveGCN" if cfg.use_adaptive else "DenseGCN",
                "lambda_phys": cfg.lambda_phys,
                "gate_sigma": model.gate_value(),
                "n_test_weeks": len(test_ids),
            }
            rows.extend(_rows_from(pred, truth, base))

            if verbose:
                overall = score(pred, truth)
                print(
                    f"  fold {k} (origin {origin:.2f}) seed {seed}: "
                    f"RMSE {overall['RMSE']:.2f}  MAE {overall['MAE']:.2f}  "
                    f"gate {model.gate_value():.3f}"
                )

        if include_persistence:
            pred, truth = _persistence_counts(raw, cfg, test_ids)
            base = {
                "label": "persistence",
                "fold": k,
                "origin": origin,
                "seed": 0,  # deterministic
                "model": "Persistence",
                "lambda_phys": 0.0,
                "gate_sigma": float("nan"),
                "n_test_weeks": len(test_ids),
            }
            rows.extend(_rows_from(pred, truth, base))

    return rows
