"""Mini-batch training loop, scored on raw counts in the frozen protocol.

The loss operates in fold-scaled log1p space, the metric on raw counts, and
early stopping watches the metric -- so a run is selected on the quantity it is
reported by. The original implementation took one full-batch gradient step per
epoch (about 60 steps total) and minimised SMAPE while being scored by RMSE;
SMAPE is dominated by districts averaging 2-5 cases a week, RMSE by Colombo and
Gampaha at 226 and 136, so the objective pulled away from the metric.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from torch import nn

import count_loss as cl

import augment as aug
import core
import models

LOSSES = ("mse_z", "huber_z", "mse_raw", "smape", "nb")
AUGMENTS = ("none", "jitter", "timegan", "timegan_only", "seir_mix", "seir_pretrain",
            "seir_mix_cal", "seir_pretrain_cal", "lds")
#: Fixed a priori in docs/AUGMENTATION_PLAN.md.
JIT_NOISE, JIT_SCALE, PRETRAIN_EPOCHS = 0.05, 0.1, 50


def _loss(kind: str, pred_z: torch.Tensor, batch: dict, mean: float, std: float,
          disp: torch.Tensor | None = None) -> torch.Tensor:
    if kind == "nb":
        # The head already emits the count scale; the likelihood makes that
        # quantity the conditional *mean*, which is what RMSE scores. No
        # smearing correction, because there is no retransformation left to bias.
        mu = torch.expm1(torch.clamp(pred_z * std + mean, -1.0, cl.LOG_MU_MAX)).clamp_min(1e-6)
        alpha = cl.ALPHA_MIN + (cl.ALPHA_MAX - cl.ALPHA_MIN) * torch.sigmoid(disp)
        return cl.nb_nll(mu, alpha, batch["y_raw"], batch.get("w"))
    if kind == "mse_z":
        return nn.functional.mse_loss(pred_z, batch["y_z"])
    if kind == "huber_z":
        return nn.functional.huber_loss(pred_z, batch["y_z"], delta=1.0)
    counts = torch.expm1(torch.clamp(pred_z * std + mean, -1.0, 12.0))
    if kind == "mse_raw":
        return nn.functional.mse_loss(counts, batch["y_raw"])
    denom = (counts.abs() + batch["y_raw"].abs() + 1e-5) / 2.0
    return ((counts - batch["y_raw"]).abs() / denom).mean()


def _jitter(batch: dict, fold: core.Fold, g: torch.Generator) -> dict:
    """G1: noise on the log1p inputs, and one magnitude shift per window.

    The shift moves inputs *and* targets together in log1p space -- a scaled
    epidemic, not a mislabelled one. Resampled every batch.
    """
    b = dict(batch)
    w = fold.window
    shape = b["x"].shape[:2] + (1,)
    shift = torch.randn(shape, generator=g) * JIT_SCALE
    noise = torch.randn(b["x"][..., :w].shape, generator=g) * JIT_NOISE
    x = b["x"].clone()
    x[..., :w] = x[..., :w] + (noise + shift) / fold.std
    b["x"] = x
    b["p_z"] = b["p_z"] + shift / fold.std
    b["y_z"] = b["y_z"] + shift / fold.std
    b["y_raw"] = torch.expm1(torch.log1p(b["y_raw"]) + shift).clamp_min(0.0)
    return b


def to_counts(pred_z: torch.Tensor, mean: float, std: float) -> np.ndarray:
    return torch.expm1(torch.clamp(pred_z * std + mean, -1.0, 12.0)).clamp_min(0.0).detach().numpy()


def run_fold(data, fold: core.Fold, *, backbone: str, head: str, loss: str = "mse_z",
             use_climate: bool = False, use_ndvi: bool = False, use_season: bool = False,
             seed: int = 0, hidden: int = 64, layers: int = 2, dropout: float = 0.1,
             lr: float = 3e-3, weight_decay: float = 1e-4, epochs: int = 300,
             batch_size: int = 32, patience: int = 40, edge=None, fixed=None,
             lam_param: str = "sigmoid", state_fit: bool = False,
             state_seed: str = "lagged", dist: str = "point",
             norm: str = "fold", node_emb: int = 0, aux_phys: float = 0.0,
             train_frac: float = 1.0, augment: str = "none",
             clim_blocks=(), clim_anom: bool = False, case_window: int | None = None,
             keep: bool = False) -> dict:
    """Train one (fold, seed) and return its test scores plus the raw predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if train_frac < 1.0:
        # Learning curve: a random subset of training windows, drawn with its own
        # generator so the model's initialisation stream is untouched. Validation,
        # test and the fold's normalisation are unchanged -- only how much the
        # model gets to learn from varies.
        tr = fold.idx["train"]
        keep_n = max(1, int(round(train_frac * len(tr))))
        sub = np.sort(np.random.default_rng(1000 + seed).choice(tr, keep_n, replace=False))
        fold = dataclasses.replace(fold, idx={**fold.idx, "train": sub})

    if case_window and case_window != fold.window:
        # docs/CLIMATE_PLAN.md K5: a longer case history with the fold boundaries
        # left exactly as they are, so the arm pairs with its control window for
        # window. (EXP-036 rebuilt the folds instead, which is why its arms could
        # not be paired.) Windows whose longer history would touch a missing week
        # or run off the start of the series are dropped.
        bad = set(np.where(data.missing)[0].tolist())
        keep_ = lambda ids: np.array([i for i in ids if i - case_window >= 0 and  # noqa: E731
                                      not any(t in bad for t in range(i - case_window, i))])
        fold = dataclasses.replace(fold, window=case_window,
                                   idx={k: keep_(v) for k, v in fold.idx.items()})
    if clim_blocks:
        fold = core.with_history(fold, max(b for _, b in clim_blocks))
    packs = {s: core.build_tensors(data, fold, s, use_climate, use_ndvi, use_season,
                                   clim_blocks, clim_anom)
             for s in ("train", "val", "test")}
    cum = {}
    for s, pack in packs.items():
        cases = np.nan_to_num(data.cases)
        cum[s] = torch.tensor(np.stack([cases[:i].sum(0) for i in pack["idx"]]), dtype=torch.float32)

    needs_state = head in ("foi", "foi_res") or aux_phys > 0
    if augment not in AUGMENTS:
        raise ValueError(f"unknown augment {augment!r}; expected one of {AUGMENTS}")
    if augment != "none" and needs_state:
        raise ValueError("augmentation is defined for heads without SEIR state")
    # docs/AUGMENTATION_PLAN.md. Synthetic windows only ever join the TRAINING
    # pack; validation and test are the real packs built above, untouched.
    pretrain = None
    n_real = len(packs["train"]["idx"])
    if augment in ("timegan", "timegan_only"):
        syn = aug.synth_timegan(data, fold, n_real, seed)
        packs["train"] = syn if augment == "timegan_only" else aug.concat(packs["train"], syn)
    elif augment in ("seir_mix", "seir_mix_cal"):
        syn = aug.synth_seir(data, fold, n_real, seed, calibrated=augment.endswith("_cal"))
        packs["train"] = aug.concat(packs["train"], syn)
    elif augment in ("seir_pretrain", "seir_pretrain_cal"):
        pretrain = aug.synth_seir(data, fold, n_real, seed, calibrated=augment.endswith("_cal"))
    elif augment == "lds":
        packs["train"]["w"] = aug.lds_weights(packs["train"]["y_raw"])
    state = {s: models.seir_state(packs[s]["x_raw"], packs[s]["pop"], cum[s],
                                  mode=state_seed) if needs_state else None
             for s in packs}

    net = models.Net(packs["train"]["x"].shape[-1], packs["train"]["x"].shape[1],
                     horizon=core.HORIZON, hidden=hidden, backbone=backbone, head=head,
                     layers=layers, dropout=dropout, lam_param=lam_param, state_fit=state_fit,
                     window=fold.window, edge_index=edge, dist=dist, norm=norm,
                     node_emb=node_emb, aux_phys=aux_phys)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    stopper = core.EarlyStop(net, patience=patience)

    def predict(split: str) -> torch.Tensor:
        pack = packs[split]
        return net(pack["x"], fixed, pack["p_z"], state[split], pack["pop"], fold.mean,
                   fold.std, edge)[0]

    if pretrain is not None:
        # G3b: learn from simulated epidemics first, then from real data as
        # usual. No early stopping here -- validation is real, and this phase
        # is not what is being selected.
        m = len(pretrain["idx"])
        for _ in range(PRETRAIN_EPOCHS):
            net.train()
            for sl in torch.randperm(m).split(batch_size):
                b = {k: v[sl] for k, v in pretrain.items() if k != "idx"}
                opt.zero_grad()
                pred, disp, _ = net(b["x"], fixed, b["p_z"], None, b["pop"],
                                    fold.mean, fold.std, edge)
                _loss(loss, pred, b, fold.mean, fold.std, disp).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    jit = torch.Generator().manual_seed(2000 + seed)
    n = len(packs["train"]["idx"])
    ran, halted = 0, False
    for ep in range(epochs):
        ran = ep + 1
        net.train()
        for sl in torch.randperm(n).split(batch_size):
            pack = packs["train"]
            batch = {k: v[sl] for k, v in pack.items() if k != "idx"}
            st = state["train"][sl] if needs_state else None
            if augment == "jitter":
                batch = _jitter(batch, fold, jit)
            opt.zero_grad()
            pred, disp, aux = net(batch["x"], fixed, batch["p_z"], st, batch["pop"],
                                  fold.mean, fold.std, edge)
            out = _loss(loss, pred, batch, fold.mean, fold.std, disp)
            if aux is not None:
                # The SEIR head is a constraint on the shared representation,
                # not the forecast: fitted in the same fold-z space, with a
                # squared error so its weight means the same for every loss.
                out = out + aux_phys * nn.functional.mse_loss(aux, batch["y_z"])
            out.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = core.rmse(to_counts(predict("val"), fold.mean, fold.std),
                          packs["val"]["y_raw"].numpy())
        if stopper.step(v):
            halted = True
            break

    stopper.restore()
    net.eval()
    with torch.no_grad():
        pred = to_counts(predict("test"), fold.mean, fold.std)
        vpred = to_counts(predict("val"), fold.mean, fold.std)
    truth = packs["test"]["y_raw"].numpy()
    out = core.score(pred, truth)
    out.update(val_RMSE=core.rmse(vpred, packs["val"]["y_raw"].numpy()),
               origin=fold.origin, seed=seed, backbone=backbone, head=head, loss=loss,
               climate=use_climate, ndvi=use_ndvi, season=use_season,
               lam_param=lam_param, state_fit=state_fit, state_seed=state_seed, dist=dist,
               norm=norm, node_emb=node_emb, aux_phys=aux_phys, train_frac=train_frac,
               augment=augment, clim_blocks=[list(b) for b in clim_blocks],
               clim_anom=clim_anom, case_window=case_window or fold.window,
               n_train=len(fold.idx["train"]), n_val=len(fold.idx["val"]),
               n_test=len(fold.idx["test"]),
               epochs_ran=ran, best_epoch=ran - stopper.waited, stopped_early=halted)
    if keep:
        # Everything a post-hoc diagnosis needs, per split: the forecast, the
        # truth, the persistence anchor and the last observed week. Test is
        # included for completeness, but diagnoses should read "val" -- studying
        # test errors and then designing around them spends the held-out set.
        with torch.no_grad():
            tpred = to_counts(predict("train"), fold.mean, fold.std)
        extra = {s_: {"pred": p_, "truth": packs[s_]["y_raw"].numpy(),
                      "persist": packs[s_]["p_raw"].numpy(),
                      "last": packs[s_]["x_raw"].numpy()[..., -1],
                      "idx": packs[s_]["idx"]}
                 for s_, p_ in (("train", tpred), ("val", vpred), ("test", pred))}
        return out, extra
    return out, pred, truth
