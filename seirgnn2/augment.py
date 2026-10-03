"""Synthetic training data -- the arms of ``docs/AUGMENTATION_PLAN.md``.

Two generators and two non-generative treatments, all producing *training*
data only. Validation and test windows are never synthetic.

``timegan``  Yoon, Jarrett & van der Schaar (NeurIPS 2019): embedder, recovery,
             generator, supervisor and discriminator GRUs, trained in the paper's
             three phases on 6-week x 25-district log1p sequences plus seasonal
             features.
``seir``     synthetic epidemics from the harness's own SEIR simulator, started
             from real training-window states, with a force of infection that
             random-walks around the value inverted from training data, observed
             through negative-binomial noise. This is the mechanistic route that
             helped in DEFSI (2019) and Osthus et al. (2026).
``jitter``   noise and magnitude scaling, resampled every batch (in ``train.py``).
``lds``      label-distribution-smoothed loss weights (Yang et al. 2021).

Leakage rule. Every generator is fitted to, or seeded from, only the fold's
analogue-safe training windows -- those whose targets end before the first
validation origin (``knn.library_idx``). ``tests/test_augment_leakage.py``
perturbs every week from the first validation origin onward and checks the
synthetic data does not move.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

import core
import knn
import models
import seir_sim

N_SEASON = 4


# ------------------------------------------------------------------ packing --

def to_pack(x_raw: np.ndarray, y_raw: np.ndarray, season: np.ndarray, pop: np.ndarray,
            src_idx: np.ndarray, fold: core.Fold) -> dict[str, torch.Tensor]:
    """Synthetic windows in exactly the layout ``core.build_tensors`` produces for B.

    B's features are the fold-z log1p case window followed by the seasonal
    features, so that is what is built -- the model cannot tell a synthetic
    window from a real one by its shape.
    """
    z = lambda a: (np.log1p(a) - fold.mean) / fold.std  # noqa: E731
    horizon = y_raw.shape[-1]
    p_raw = np.repeat(x_raw[..., -1:], horizon, axis=-1)
    feats = np.concatenate([z(x_raw), np.repeat(season[:, None, :], x_raw.shape[1], 1)], -1)
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    return {"x": t(feats), "y_raw": t(y_raw), "p_raw": t(p_raw), "y_z": t(z(y_raw)),
            "p_z": t(z(p_raw)), "x_raw": t(x_raw), "pop": t(pop), "idx": src_idx}


def concat(a: dict, b: dict) -> dict:
    out = {k: torch.cat([a[k], b[k]]) for k in a if k != "idx"}
    out["idx"] = np.concatenate([a["idx"], b["idx"]])
    return out


def _real_sequences(data, fold: core.Fold):
    """Analogue-safe training windows as (K, W+H, N) log1p cases and (K, W+H, 4) season."""
    lib = knn.library_idx(fold)
    w, h = fold.window, core.HORIZON
    cases = np.nan_to_num(data.cases)
    seq = np.stack([np.log1p(cases[i - w : i + h]) for i in lib])                   # (K,T,N)
    season = np.stack([core.seasonal_features(data.week_start, np.arange(i - w, i + h))
                       for i in lib])                                                # (K,T,4)
    return lib, seq, season


# ------------------------------------------------------------------ TimeGAN --

class _Rnn(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, layers: int, act: bool = True):
        super().__init__()
        self.rnn = nn.GRU(d_in, d_hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(d_hidden, d_out)
        self.act = act

    def forward(self, x):
        h, _ = self.rnn(x)
        y = self.fc(h)
        return torch.sigmoid(y) if self.act else y


def timegan(seqs: np.ndarray, n_samples: int, seed: int, hidden: int = 32, layers: int = 2,
            iters: int = 1500, batch: int = 64, gamma: float = 1.0) -> np.ndarray:
    """Fit TimeGAN to ``seqs`` (K, T, F) and draw ``n_samples`` sequences.

    Follows Yoon et al.'s reference implementation: min-max scaling, (1) embedder
    + recovery trained to reconstruct, (2) supervisor trained on next-step
    embeddings, (3) joint training -- two generator steps per embedder step,
    discriminator updated only while its loss exceeds 0.15. Loss weights are the
    paper's (reconstruction x10, supervised x100, moments x100, gamma = 1).
    """
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    lo, hi = seqs.min((0, 1), keepdims=True), seqs.max((0, 1), keepdims=True)
    x_all = torch.tensor((seqs - lo) / (hi - lo + 1e-7), dtype=torch.float32)
    k, t, f = x_all.shape

    emb = _Rnn(f, hidden, hidden, layers)
    rec = _Rnn(hidden, hidden, f, layers)
    gen = _Rnn(f, hidden, hidden, layers)
    sup = _Rnn(hidden, hidden, hidden, max(1, layers - 1))
    dis = _Rnn(hidden, hidden, 1, layers, act=False)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    def batch_x():
        return x_all[torch.randint(0, k, (min(batch, k),), generator=g)]

    def noise(n):
        return torch.rand(n, t, f, generator=g)

    opt_er = torch.optim.Adam([*emb.parameters(), *rec.parameters()], lr=1e-3)
    for _ in range(iters):                                   # phase 1: autoencoder
        x = batch_x()
        loss = 10 * torch.sqrt(mse(rec(emb(x)), x))
        opt_er.zero_grad(); loss.backward(); opt_er.step()

    opt_s = torch.optim.Adam(sup.parameters(), lr=1e-3)
    for _ in range(iters):                                   # phase 2: supervisor
        h = emb(batch_x()).detach()
        loss = mse(sup(h)[:, :-1], h[:, 1:])
        opt_s.zero_grad(); loss.backward(); opt_s.step()

    opt_g = torch.optim.Adam([*gen.parameters(), *sup.parameters()], lr=1e-3)
    opt_e = torch.optim.Adam([*emb.parameters(), *rec.parameters()], lr=1e-3)
    opt_d = torch.optim.Adam(dis.parameters(), lr=1e-3)
    for _ in range(iters):                                   # phase 3: joint
        for _ in range(2):
            x = batch_x(); z = noise(len(x))
            h = emb(x); e_hat = gen(z); h_hat = sup(e_hat); x_hat = rec(h_hat)
            y_fake, y_fake_e = dis(h_hat), dis(e_hat)
            g_u = bce(y_fake, torch.ones_like(y_fake)) + gamma * bce(y_fake_e, torch.ones_like(y_fake_e))
            g_s = mse(sup(h)[:, :-1], h[:, 1:])
            g_v = (torch.abs(x_hat.std(0) - x.std(0)).mean()
                   + torch.abs(x_hat.mean(0) - x.mean(0)).mean())
            loss = g_u + 100 * torch.sqrt(g_s) + 100 * g_v
            opt_g.zero_grad(); loss.backward(); opt_g.step()

            x = batch_x(); h = emb(x)
            e_loss = 10 * torch.sqrt(mse(rec(h), x)) + 0.1 * mse(sup(h)[:, :-1], h[:, 1:])
            opt_e.zero_grad(); e_loss.backward(); opt_e.step()

        x = batch_x(); z = noise(len(x))
        h = emb(x).detach(); e_hat = gen(z).detach(); h_hat = sup(e_hat).detach()
        y_real, y_fake, y_fake_e = dis(h), dis(h_hat), dis(e_hat)
        d_loss = (bce(y_real, torch.ones_like(y_real)) + bce(y_fake, torch.zeros_like(y_fake))
                  + gamma * bce(y_fake_e, torch.zeros_like(y_fake_e)))
        if d_loss.item() > 0.15:
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

    with torch.no_grad():
        out = rec(sup(gen(noise(n_samples)))).numpy()
    return out * (hi - lo + 1e-7) + lo


def synth_timegan(data, fold: core.Fold, n: int, seed: int, iters: int = 1500) -> dict:
    lib, seq, season = _real_sequences(data, fold)
    w = fold.window
    fake = timegan(np.concatenate([seq, season], -1), n, seed, iters=iters)          # (n,T,N+4)
    n_nodes = seq.shape[-1]
    counts = np.expm1(np.clip(fake[..., :n_nodes], 0.0, None))                       # (n,T,N)
    x_raw = counts[:, :w].transpose(0, 2, 1)
    y_raw = counts[:, w:].transpose(0, 2, 1)
    seas = np.clip(fake[:, w, n_nodes:], -1.0, 1.0)                                  # origin week
    src = np.random.default_rng(seed).choice(lib, n)
    pop = data.population[src - 1]
    return to_pack(x_raw, y_raw, seas, pop, src, fold)


# --------------------------------------------------------------- SEIR sampler --

def _lambda_stats(data, fold: core.Fold) -> tuple[float, float]:
    """Median log lambda and sd of its week-to-week change, from inverted training windows.

    Inverted with ``diagnose_foi.invert`` on analogue-safe training windows only.
    Cells pinned at lambda = 0 (the target sits below the maintenance floor) are
    excluded: their log is a numerical artefact, not a transmission intensity.
    """
    import diagnose_foi as dg

    lib = knn.library_idx(fold)
    cases = np.nan_to_num(data.cases)
    x_raw = torch.tensor(np.stack([cases[i - fold.window : i].T for i in lib]), dtype=torch.float32)
    y = torch.tensor(np.stack([cases[i : i + core.HORIZON].T for i in lib]), dtype=torch.float32)
    pop = torch.tensor(data.population[lib - 1], dtype=torch.float32)
    cum = torch.tensor(np.stack([cases[:i].sum(0) for i in lib]), dtype=torch.float32)
    st = models.seir_state(x_raw, pop, cum)
    lams = []
    for h in range(core.HORIZON):
        lam = dg.invert(st, pop, y[..., h])
        lams.append(lam)
        st = dg.step(st, lam)
    lam = torch.stack(lams, -1).numpy()
    ok = lam > 1e-8
    log_lam = np.log(np.where(ok, lam, np.nan))
    step = np.diff(log_lam, axis=-1)
    return float(np.nanmedian(log_lam)), float(np.nanstd(step))


#: Calibration grid (logged deviation, EXP-042): a shift of the log-lambda
#: centre and a multiplier on the random-walk step.
CAL_OFFSETS = tuple(np.arange(-4.0, 1.01, 0.25))
CAL_STEPS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def _growth_stats(counts: np.ndarray) -> tuple[float, float]:
    """Mean and sd of week-to-week log1p growth over (window, district, week) cells."""
    g = np.diff(np.log1p(counts), axis=-1)
    return float(g.mean()), float(g.std())


def calibrate_seir(data, fold: core.Fold, n: int = 400, seed: int = 12345) -> tuple[float, float]:
    """Pick (offset, step multiplier) so simulated growth matches training growth.

    Fitted on analogue-safe training windows only -- the same data a GAN would be
    fitted to -- and on week-to-week *growth*, not levels: levels come from the
    real starting states, so growth is the part the lambda process controls.
    Matching its mean and spread leaves the tails free to exceed what history
    shows, which is the reason for simulating at all.
    """
    _, seq, _ = _real_sequences(data, fold)
    target = _growth_stats(np.expm1(seq).transpose(0, 2, 1))
    med, sd = _lambda_stats(data, fold)
    best = None
    for off in CAL_OFFSETS:
        for mult in CAL_STEPS:
            c = _simulate(data, fold, n, seed, 0.2, med + off, sd / np.sqrt(2.0) * mult)
            m, s = _growth_stats(np.concatenate([c[0], c[1]], -1))
            err = (m - target[0]) ** 2 + (s - target[1]) ** 2
            if best is None or err < best[0]:
                best = (err, off, mult)
    return best[1], best[2]


def synth_seir(data, fold: core.Fold, n: int, seed: int, alpha: float = 0.2,
               calibrated: bool = False) -> dict:
    """``n`` synthetic 6-week epidemics: real start, random-walk lambda, NB noise.

    ``calibrated=False`` is the simulator exactly as pre-registered.
    ``calibrated=True`` first fits the lambda centre and step to training growth
    (``calibrate_seir``) -- a logged deviation, added after the pre-registered
    version was found to produce targets ~10x the real scale.
    """
    off, mult = calibrate_seir(data, fold) if calibrated else (0.0, 1.0)
    med, sd = _lambda_stats(data, fold)
    x_raw, y_raw, season, pop, src = _simulate(data, fold, n, seed, alpha, med + off,
                                               sd / np.sqrt(2.0) * mult)
    return to_pack(x_raw, y_raw, season, pop, src, fold)


def _simulate(data, fold: core.Fold, n: int, seed: int, alpha: float, med: float, step: float):
    """Raw simulation: log-lambda centre ``med`` with random-walk step sd ``step``."""
    rng = np.random.default_rng(seed)
    lib = knn.library_idx(fold)
    w, h = fold.window, core.HORIZON
    cases = np.nan_to_num(data.cases)

    src = rng.choice(lib, n)
    x0 = torch.tensor(np.stack([cases[i - w : i].T for i in src]), dtype=torch.float32)
    pop = data.population[src - 1]
    popt = torch.tensor(pop, dtype=torch.float32)
    cum = torch.tensor(np.stack([cases[:i].sum(0) for i in src]), dtype=torch.float32)
    state0 = models.seir_state(x0, popt, cum)                                       # (n,N,4)

    n_nodes = x0.shape[1]
    national = np.cumsum(rng.normal(0, step, (n, 1, w + h)), axis=-1)
    district = np.cumsum(rng.normal(0, step, (n, n_nodes, w + h)), axis=-1)
    log_lam = np.clip(med + national + district, -25.0, models.LOG_LAMBDA_MAX)
    lam = torch.tensor(np.exp(log_lam), dtype=torch.float32)
    _, inc = seir_sim.simulate_weeks(state0, lam, omega=0.7 / 7.0, gamma=1.0 / 7.0, substeps=7)
    mu = (inc.numpy() * (1.0 / 11.0) * pop[..., None]).clip(1e-9, None)
    rate = rng.gamma(shape=1.0 / alpha, scale=mu * alpha)                            # NB2 via Gamma-Poisson
    counts = rng.poisson(rate).astype(np.float64)

    x_raw, y_raw = counts[..., :w], counts[..., w:]
    season = core.seasonal_features(data.week_start, src + w)                       # origin = start + W
    return x_raw, y_raw, season, pop, src


# ------------------------------------------------------------------- LDS --

def lds_weights(y_raw: torch.Tensor, bins: int = 50, kernel_sd: float = 2.0) -> torch.Tensor:
    """Label-distribution-smoothed weights, density^-0.5, normalised to mean 1."""
    y = np.log1p(y_raw.numpy())
    edges = np.linspace(y.min(), y.max() + 1e-9, bins + 1)
    counts, _ = np.histogram(y, edges)
    ker = np.exp(-0.5 * (np.arange(-3 * kernel_sd, 3 * kernel_sd + 1) / kernel_sd) ** 2)
    smooth = np.convolve(counts, ker / ker.sum(), mode="same")
    b = np.clip(np.digitize(y, edges) - 1, 0, bins - 1)
    wgt = 1.0 / np.sqrt(np.maximum(smooth[b], 1e-6))
    return torch.tensor(wgt / wgt.mean(), dtype=torch.float32)
