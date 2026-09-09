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
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dengue_gnn.augment import augment_training_set
from dengue_gnn.losses import spatial_regularisation
from dengue_gnn.mechanistic import (
    MAX_WEEKLY_LOG_GROWTH,
    curriculum_weight,
    mechanistic_regularisation,
)
from dengue_gnn.metrics import peak_week_error, score
from dengue_gnn.models import AdaptiveGCN, build_fixed_adjacency

__all__ = [
    "Config",
    "FoldData",
    "fold_split",
    "load_dataset",
    "persistence_rows",
    "rolling_origin",
    "run_single",
    "select_shrinkage",
    "train_fold",
]


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
    temporal: str = "none"  # none | gtcn | gru | lstm | gru_attn
    spatial: str = "gcn"  # gcn | gat -- graph attention (Velickovic et al.)
    gat_heads: int = 8

    # objective
    lambda_phys: float = 0.0
    cons_weight: float = 10.0
    reg_space: str = "log"  # "log" | "count" -- see _regularisation_target

    # Stage-3 mechanistic constraints on observables (dengue_gnn.mechanistic).
    lambda_mech: float = 0.0
    mech_mode: str = "both"  # "band" | "smooth" | "both"
    # Growth ceiling for the band term. An axis rather than a constant so the
    # old guessed 0.70 and the SEIR-derived value can be compared as two arms of
    # one run, changing nothing else (D7). See dengue_gnn.seir.
    max_growth: float = MAX_WEEKLY_LOG_GROWTH
    curriculum: float = 0.0  # >0 ramps constraint weights over this fraction of training

    # Residual shrinkage. The network predicts a correction to persistence, so
    # scaling that correction by gamma interpolates between persistence (0) and
    # the unshrunk model (1). gamma is chosen on the VALIDATION fold and then
    # applied to test -- selecting it on test would be choosing the answer.
    shrink: bool = False
    shrink_grid: tuple[float, ...] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    # Contribution (b): training-set augmentation. Applied to the TRAINING split
    # only -- augmenting validation or test would leak synthetic structure into
    # the reported numbers.
    augment: str = "none"  # none | jitter | window_warp | gan
    augment_ratio: float = 0.5
    gan_epochs: int = 200

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


def _evaluate(model, x, y, p, fold: FoldData, cfg: Config, device, gamma: float = 1.0) -> tuple:
    """Return (pred_counts, true_counts) as numpy, shape (n_windows, n_nodes, H).

    ``gamma`` scales the predicted residual before it is added back to
    persistence: 0 reproduces persistence exactly, 1 is the raw model. It only
    applies when ``cfg.residual`` is set, because otherwise the output is a level
    rather than a correction and scaling it means nothing.
    """
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for i in range(x.shape[0]):
            out = model(x[i].unsqueeze(0).to(device)).squeeze(0).cpu()
            if cfg.residual:
                out = gamma * out + p[i]
            preds.append(fold.to_counts(out))
            truths.append(fold.to_counts(y[i]))
    return torch.stack(preds).numpy(), torch.stack(truths).numpy()


def select_shrinkage(model, fold: FoldData, cfg: Config, device) -> float:
    """Pick the residual scale that minimises RMSE on the validation fold.

    Returns 1.0 (no shrinkage) when disabled or when the target is not a
    residual. The validation fold is the same one early stopping uses, so no
    additional data is consumed and the test fold stays untouched.
    """
    if not cfg.shrink or not cfg.residual:
        return 1.0
    best_gamma, best_rmse = 1.0, float("inf")
    for gamma in cfg.shrink_grid:
        pred, truth = _evaluate(model, fold.x_val, fold.y_val, fold.p_val, fold, cfg, device, gamma)
        rmse = score(pred, truth)["RMSE"]
        if rmse < best_rmse:
            best_gamma, best_rmse = gamma, rmse
    return best_gamma


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
    n_feat = fold.x_train.shape[2] // cfg.window
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
        temporal=cfg.temporal,
        spatial=cfg.spatial,
        gat_heads=cfg.gat_heads,
        n_feat=n_feat if cfg.temporal != "none" else None,
        window=cfg.window if cfg.temporal != "none" else None,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()

    best, best_state, wait = float("inf"), None, 0
    for epoch in range(cfg.epochs):
        model.train()
        for i in torch.randperm(fold.x_train.shape[0]):
            opt.zero_grad()
            out = model(fold.x_train[i].unsqueeze(0).to(device))
            target = fold.y_train[i] - fold.p_train[i] if cfg.residual else fold.y_train[i]
            loss = mse(out.squeeze(0), target.to(device))

            if cfg.lambda_phys > 0.0 or cfg.lambda_mech > 0.0:
                normalised = out.squeeze(0)
                if cfg.residual:
                    normalised = normalised + fold.p_train[i].to(device)

                # Krishnapriyan et al.: a constraint imposed at full strength from
                # step one deforms the loss landscape. Ramping it lets the data
                # term establish a solution first.
                scale = (
                    curriculum_weight(epoch, cfg.epochs, 1.0, cfg.curriculum)
                    if cfg.curriculum > 0
                    else 1.0
                )

            if cfg.lambda_phys > 0.0:
                target = _regularisation_target(fold, normalised, cfg.reg_space)
                loss = loss + spatial_regularisation(
                    target.unsqueeze(0),
                    model.blended_adjacency(),
                    lambda_phys=cfg.lambda_phys * scale,
                    cons_weight=cfg.cons_weight,
                )

            if cfg.lambda_mech > 0.0:
                counts = fold.to_counts(normalised).unsqueeze(0)
                last_obs = fold.to_counts(fold.p_train[i][:, 0].to(device)).unsqueeze(0)
                loss = loss + mechanistic_regularisation(
                    counts,
                    last_obs,
                    lambda_mech=cfg.lambda_mech * scale,
                    mode=cfg.mech_mode,
                    max_growth=cfg.max_growth,
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


def fold_split(raw: np.ndarray, cfg: Config, fold_idx: int):
    """Return ``(train_ids, val_ids, test_ids, origin)`` for one fold.

    Split arithmetic lives here so the sequential and parallel runners cannot
    drift apart -- if these differ by one window, two "identical" protocols
    silently stop being comparable.
    """
    ids = list(range(cfg.window, raw.shape[0] - cfg.horizon))
    n = len(ids)
    origin = cfg.origins[fold_idx]
    cut = int(origin * n)
    tend = int(min(origin + cfg.test_frac, 1.0) * n)
    train_pool, test_ids = ids[:cut], ids[cut:tend]
    return train_pool[: -cfg.val_weeks], train_pool[-cfg.val_weeks :], test_ids, origin


def run_single(
    raw: np.ndarray,
    adj_fixed: torch.Tensor,
    cfg: Config,
    fold_idx: int,
    seed: int,
    device: torch.device | None = None,
) -> list[dict]:
    """Train and evaluate one ``(fold, seed)`` and return its rows.

    The unit of parallelism. Every run is independent, so a pool of these
    saturates the available cores -- see ``scripts/run_phase3.py`` for why that
    beats a GPU on this workload.

    Rows carry computational measurements (parameter count, training wall-clock,
    inference latency) alongside accuracy, because the Phase-3 paper requires a
    Computational Analysis section and measuring it afterwards means running
    everything twice.
    """
    device = device or torch.device("cpu")
    train_ids, val_ids, test_ids, origin = fold_split(raw, cfg, fold_idx)
    if not test_ids:
        return []

    (xtr, ytr, ptr), (xva, yva, pva), tmean, tstd = _build_fold(raw, cfg, train_ids, val_ids)
    _, (xte, yte, pte), _, _ = _build_fold(raw, cfg, train_ids, test_ids)

    n_real = xtr.shape[0]
    if cfg.augment != "none" and cfg.augment_ratio > 0:
        xtr, ytr, ptr = augment_training_set(
            xtr,
            ytr,
            ptr,
            kind=cfg.augment,
            ratio=cfg.augment_ratio,
            window=cfg.window,
            n_feat=raw.shape[2],
            seed=seed,
            gan_epochs=cfg.gan_epochs,
        )

    fold = FoldData(xtr, ytr, ptr, xva, yva, pva, xte, yte, pte, tmean, tstd, test_ids)

    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- see rolling_origin

    t0 = time.perf_counter()
    model = train_fold(fold, cfg, adj_fixed, device)
    train_seconds = time.perf_counter() - t0

    gamma = select_shrinkage(model, fold, cfg, device)

    t0 = time.perf_counter()
    pred, truth = _evaluate(model, xte, yte, pte, fold, cfg, device, gamma)
    infer_seconds = time.perf_counter() - t0

    # How far the model actually moved from persistence, relative to how far it
    # needed to move. Recorded on every row because RMSE alone cannot tell a
    # forecast apart from a copy of the baseline: under the residual
    # parameterisation (ADR-0001) a model whose output collapses to zero *is*
    # persistence and inherits its score, which on this data is a competitive
    # one. That is how STGAT came top of the Phase-3 architecture table while
    # moving 2-5% of the required distance (EXP-013). A value near 0 means the
    # arm is not forecasting; a value near 1 means it is making corrections of
    # the right magnitude, and says nothing about their direction.
    persist_pred, _ = _persistence_counts(raw, cfg, test_ids)
    needed = float(np.mean(np.abs(truth - persist_pred)))
    moved = float(np.mean(np.abs(pred - persist_pred)))
    resid_ratio = moved / needed if needed else float("nan")

    base = {
        "label": cfg.label,
        "fold": fold_idx + 1,
        "origin": origin,
        "seed": seed,
        "model": "AdaptiveGCN" if cfg.use_adaptive else "DenseGCN",
        "lambda_phys": cfg.lambda_phys,
        "gate_sigma": model.gate_value(),
        "shrink_gamma": gamma,
        "resid_ratio": round(resid_ratio, 4),
        "max_growth": cfg.max_growth,
        "temporal": cfg.temporal,
        "spatial": cfg.spatial,
        "window": cfg.window,
        "hidden": cfg.hidden,
        "lr": cfg.lr,
        "augment": cfg.augment,
        "n_train_windows": int(xtr.shape[0]),
        "n_real_windows": int(n_real),
        "n_test_weeks": len(test_ids),
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_seconds": round(train_seconds, 3),
        "infer_ms_per_window": round(1000 * infer_seconds / max(len(test_ids), 1), 4),
    }
    return _rows_from(pred, truth, base)


def persistence_rows(raw: np.ndarray, cfg: Config, fold_idx: int) -> list[dict]:
    """Persistence records for one fold. Deterministic, so one seed only."""
    _, _, test_ids, origin = fold_split(raw, cfg, fold_idx)
    if not test_ids:
        return []
    pred, truth = _persistence_counts(raw, cfg, test_ids)
    base = {
        "label": "persistence",
        "fold": fold_idx + 1,
        "origin": origin,
        "seed": 0,
        "model": "Persistence",
        "lambda_phys": 0.0,
        "gate_sigma": float("nan"),
        "shrink_gamma": 0.0,  # persistence IS gamma=0
        "resid_ratio": 0.0,  # persistence moves nowhere, by definition
        "max_growth": float("nan"),  # no constraint applies to the baseline
        "temporal": "none",
        "spatial": "none",
        "window": cfg.window,
        "hidden": 0,
        "lr": 0.0,
        "augment": "none",
        "n_train_windows": 0,
        "n_real_windows": 0,
        "n_test_weeks": len(test_ids),
        "n_params": 0,
        "train_seconds": 0.0,
        "infer_ms_per_window": 0.0,
    }
    return _rows_from(pred, truth, base)


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
    rows: list[dict] = []

    # Delegates to run_single so the sequential and parallel runners share one
    # definition of a run. Two copies of split arithmetic drift by a window and
    # silently stop being the same protocol.
    for fold_idx in range(len(cfg.origins)):
        for seed in cfg.seeds:
            fold_rows = run_single(raw, adj_fixed, cfg, fold_idx, seed, device)
            rows.extend(fold_rows)
            if verbose and fold_rows:
                pooled = next(r for r in fold_rows if r["horizon"] == 0)
                print(
                    f"  fold {pooled['fold']} (origin {pooled['origin']:.2f}) "
                    f"seed {seed}: RMSE {pooled['rmse']:.2f}  MAE {pooled['mae']:.2f}  "
                    f"gate {pooled['gate_sigma']:.3f}"
                )
        if include_persistence:
            rows.extend(persistence_rows(raw, cfg, fold_idx))

    return rows
