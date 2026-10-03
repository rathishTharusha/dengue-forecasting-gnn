# %% [markdown]
# ## 6. Training
#
# One run = one configuration × one fold × one seed. Adam (learning rate 3e-3, weight
# decay 1e-4, batch 32, cosine schedule), gradient clipping at 5, early stopping on
# **validation RMSE on counts** with patience 40, and the best-validation weights
# restored. Test is scored once, at the end, and never used to choose anything.
# Every run is single-threaded, so a configuration is reproducible from its seed.

# %%
def cumulative(idx):
    return cumulative_before(idx)


def run_fold(fold, *, backbone, head, loss="mse_z", use_climate=False, use_season=False, seed=0,
             hidden=64, layers=2, dropout=0.1, lr=3e-3, weight_decay=1e-4, epochs=300, batch_size=32,
             patience=40, lam_param="sigmoid", state_fit=False, dist="point", norm="fold", node_emb=0,
             aux_phys=0.0, train_frac=1.0, augment="none", clim_blocks=(), head_mlp=0, dseason=False,
             global_ctx=False, spatial=0.0, spatial_kind="ratio", keep_train=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if train_frac < 1.0:
        tr = fold.idx["train"]
        sub = np.sort(np.random.default_rng(1000 + seed).choice(tr, max(1, int(round(train_frac * len(tr)))),
                                                               replace=False))
        fold = dc_replace(fold, idx={**fold.idx, "train": sub})
    if clim_blocks:
        fold = with_history(fold, max(b for _, b in clim_blocks))
    packs = {s: build_tensors(fold, s, use_climate, use_season, clim_blocks) for s in ("train", "val", "test")}
    needs_state = head in ("foi", "foi_res", "foi_meta") or aux_phys > 0
    pretrain = None
    n_real = len(packs["train"]["idx"])
    tail = {}
    if augment == "timegan":
        syn = synth_timegan(fold, n_real, seed)
        tail = {"synthetic_max": float(torch.cat([syn["x_raw"].flatten(), syn["y_raw"].flatten()]).max()),
                "real_train_max": float(torch.cat([packs["train"]["x_raw"].flatten(),
                                                   packs["train"]["y_raw"].flatten()]).max())}
        packs["train"] = concat_packs(packs["train"], syn)
    elif augment == "seir_pretrain_cal":
        pretrain = synth_seir(fold, n_real, seed)
    state = {s: seir_state(p["x_raw"], p["pop"], cumulative(p["idx"])) if needs_state else None
             for s, p in packs.items()}
    net = Net(packs["train"]["x"].shape[-1], N, horizon=HORIZON, hidden=hidden, backbone=backbone, head=head,
              layers=layers, dropout=dropout, lam_param=lam_param, state_fit=state_fit, window=fold.window,
              dist=dist, norm=norm, node_emb=node_emb, aux_phys=aux_phys, head_mlp=head_mlp, dseason=dseason,
              global_ctx=global_ctx)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def predict(split):
        p = packs[split]
        return net(p["x"], FIXED, p["p_z"], state[split], p["pop"], fold.mean, fold.std)[0]

    if pretrain is not None:           # learn from simulated epidemics first; no early stopping here
        m = len(pretrain["idx"])
        for _ in range(50):
            net.train()
            for sl in torch.randperm(m).split(batch_size):
                b = {k: v[sl] for k, v in pretrain.items() if k != "idx"}
                opt.zero_grad()
                pred, disp, _ = net(b["x"], FIXED, b["p_z"], None, b["pop"], fold.mean, fold.std)
                loss_fn(loss, pred, b, fold.mean, fold.std, disp).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if spatial:
        adj_bin = ((FIXED > 0) & ~torch.eye(N, dtype=torch.bool)).float()
        scales = torch.tensor(np.nanmean(CASES[: int(fold.idx["train"].max())], axis=0), dtype=torch.float32)

    best, best_state, waited, ran = float("inf"), None, 0, 0
    n = len(packs["train"]["idx"])
    for ep in range(epochs):
        ran = ep + 1
        net.train()
        for sl in torch.randperm(n).split(batch_size):
            batch = {k: v[sl] for k, v in packs["train"].items() if k != "idx"}
            st = state["train"][sl] if needs_state else None
            opt.zero_grad()
            pred, disp, aux = net(batch["x"], FIXED, batch["p_z"], st, batch["pop"], fold.mean, fold.std)
            out = loss_fn(loss, pred, batch, fold.mean, fold.std, disp)
            if spatial:
                counts = torch.expm1(torch.clamp(pred * fold.std + fold.mean, -1.0, 12.0)).clamp_min(0.0)
                out = out + spatial * spatial_penalty(counts, adj_bin, scales, spatial_kind)
            if aux is not None:
                out = out + aux_phys * F.mse_loss(aux, batch["y_z"])
            out.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = rmse(to_counts(predict("val"), fold.mean, fold.std), packs["val"]["y_raw"].numpy())
        if v < best - 1e-6:
            best, waited = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    splits = ("train", "val", "test") if keep_train else ("val", "test")
    with torch.no_grad():
        preds = {s: to_counts(predict(s), fold.mean, fold.std) for s in splits}
    truth = {s: packs[s]["y_raw"].numpy() for s in splits}
    row = score(preds["test"], truth["test"])
    row.update(val_RMSE=rmse(preds["val"], truth["val"]), best_epoch=ran - waited, epochs_ran=ran,
               n_train=len(fold.idx["train"]), **tail)
    return row, {"pred": preds, "truth": truth, "persist": {s: packs[s]["p_raw"].numpy() for s in splits},
                 "idx": {s: np.asarray(packs[s]["idx"]) for s in splits}}


# %% [markdown]
# ### 6.1 Forecasters without a network
#
# * **k-nearest-neighbour analogues**: each (district, window) is matched to the k most
#   similar training windows by the shape of its recent log-cases, optionally its level
#   and season; the forecast applies the neighbours' realised growth. k and the two
#   feature weights are chosen on validation.
# * **Gradient-boosted trees** on log-growth from last week, recent history, season,
#   district identity and optionally climate lag blocks, one model per horizon week.

# %%
from itertools import product

GRID_K, GRID_LEVEL, GRID_SEASON = (10, 20, 40, 80, 160, 320), (0.0, 0.5, 1.0), (0.0, 0.5)


def library_idx(fold):
    """Training origins whose whole target span ends before the first validation origin."""
    tr = fold.idx["train"]
    return tr[tr + HORIZON - 1 < int(fold.idx["val"].min())]


def _knn_windows(idx, window, with_target):
    c = np.nan_to_num(CASES)
    hist = np.stack([np.log1p(c[i - window: i]).T for i in idx])
    last = hist[..., -1]
    season = np.repeat(seasonal_features(idx)[:, None, :], N, 1)
    growth = (np.stack([np.log1p(c[i: i + HORIZON]).T for i in idx]) - last[..., None]) if with_target else None
    return hist, last, season, growth


def _knn_predict(lib, query, k, wl, ws):
    feats = lambda h, l, s: np.concatenate([h - l[..., None], wl * l[..., None], ws * s], -1)  # noqa: E731
    lf = feats(*lib[:3]).reshape(-1, lib[0].shape[-1] + 5)
    qf = feats(*query[:3]).reshape(-1, query[0].shape[-1] + 5)
    lg, q_last = lib[3].reshape(-1, HORIZON), query[1].reshape(-1)
    lsq, out = (lf ** 2).sum(1)[None, :], np.empty((len(qf), HORIZON))
    for a in range(0, len(qf), 1024):
        q = qf[a: a + 1024]
        d2 = (q ** 2).sum(1)[:, None] - 2 * q @ lf.T + lsq
        nn_idx = np.argpartition(d2, kth=min(k, d2.shape[1] - 1), axis=1)[:, :k]
        out[a: a + 1024] = np.expm1(q_last[a: a + 1024, None, None] + lg[nn_idx]).mean(1)
    return out.clip(min=0.0).reshape(*query[1].shape, -1)


def run_knn(fold, seed=0, **_):
    lib = _knn_windows(library_idx(fold), fold.window, True)
    q = {s: _knn_windows(fold.idx[s], fold.window, False) for s in ("val", "test")}
    truth = {s: np.stack([CASES[i: i + HORIZON].T for i in fold.idx[s]]) for s in q}
    best = min(((rmse(_knn_predict(lib, q["val"], k, wl, ws), truth["val"]), k, wl, ws)
                for k, wl, ws in product(GRID_K, GRID_LEVEL, GRID_SEASON)))
    preds = {s: _knn_predict(lib, q[s], *best[1:]) for s in q}
    row = score(preds["test"], truth["test"])
    row.update(val_RMSE=best[0])
    persist = {s: np.repeat(CASES[fold.idx[s] - 1][:, :, None], HORIZON, 2) for s in q}
    return row, {"pred": preds, "truth": truth, "persist": persist, "idx": {s: fold.idx[s] for s in q}}


def run_gbm(fold, seed=0, clim_blocks=(), **_):
    from sklearn.ensemble import HistGradientBoostingRegressor

    if clim_blocks:
        fold = with_history(fold, max(b for _, b in clim_blocks))
    lc = np.log1p(np.nan_to_num(CASES))

    def design(idx):
        last = lc[idx - 1]
        parts = [last[..., None], np.stack([lc[idx - k] - last for k in (2, 3)], -1),
                 np.repeat(seasonal_features(idx)[:, None, :], N, 1), np.repeat(np.eye(N)[None], len(idx), 0)]
        if clim_blocks:
            blk, ref = climate_blocks(idx, clim_blocks, CLIMATE), climate_blocks(fold.idx["train"], clim_blocks, CLIMATE)
            parts.append((blk - ref.mean((0, 1))) / (ref.std((0, 1)) + 1e-8))
        return np.concatenate(parts, -1)

    x = {s: design(fold.idx[s]) for s in ("train", "val", "test")}
    xt = x["train"].reshape(-1, x["train"].shape[-1])
    preds = {s: np.zeros((len(fold.idx[s]), N, HORIZON)) for s in ("val", "test")}
    for k in range(HORIZON):
        yt = (lc[fold.idx["train"] + k] - lc[fold.idx["train"] - 1]).reshape(-1)
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=40, random_state=0).fit(xt, yt)
        smear = float(np.mean(np.exp(yt - m.predict(xt))))
        for s in preds:
            g = m.predict(x[s].reshape(-1, x[s].shape[-1])).reshape(len(fold.idx[s]), N)
            preds[s][..., k] = np.clip((np.nan_to_num(CASES)[fold.idx[s] - 1] + 1.0) * np.exp(g) * smear - 1.0, 0, None)
    truth = {s: np.stack([CASES[i: i + HORIZON].T for i in fold.idx[s]]) for s in preds}
    row = score(preds["test"], truth["test"])
    row.update(val_RMSE=rmse(preds["val"], truth["val"]))
    persist = {s: np.repeat(CASES[fold.idx[s] - 1][:, :, None], HORIZON, 2) for s in preds}
    return row, {"pred": preds, "truth": truth, "persist": persist}


# %% [markdown]
# ### 6.2 Synthetic training data
#
# * **TimeGAN** (Yoon et al. 2019), fitted to the analogue-safe training windows of each
#   fold: embedder/recovery, supervisor, then joint adversarial training with the
#   paper's loss weights. Its samples join the training set only.
# * **SEIR-simulated epidemics**: real starting states, a random-walk force of
#   infection whose centre and step are calibrated to training-window growth, and
#   negative-binomial reporting noise. The model pre-trains on them for 50 epochs.

# %%
def to_pack(x_raw, y_raw, season, pop, src, fold):
    z = lambda a: (np.log1p(a) - fold.mean) / fold.std  # noqa: E731
    p_raw = np.repeat(x_raw[..., -1:], y_raw.shape[-1], -1)
    feats = np.concatenate([z(x_raw), np.repeat(season[:, None, :], x_raw.shape[1], 1)], -1)
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    return {"x": t(feats), "y_raw": t(y_raw), "p_raw": t(p_raw), "y_z": t(z(y_raw)), "p_z": t(z(p_raw)),
            "x_raw": t(x_raw), "pop": t(pop), "idx": src}


def concat_packs(a, b):
    out = {k: torch.cat([a[k], b[k]]) for k in a if k != "idx"}
    out["idx"] = np.concatenate([a["idx"], b["idx"]])
    return out


def _real_sequences(fold):
    lib, w = library_idx(fold), fold.window
    c = np.nan_to_num(CASES)
    seq = np.stack([np.log1p(c[i - w: i + HORIZON]) for i in lib])
    season = np.stack([seasonal_features(np.arange(i - w, i + HORIZON)) for i in lib])
    return lib, seq, season


class _Rnn(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, layers, act=True):
        super().__init__()
        self.rnn, self.fc, self.act = nn.GRU(d_in, d_hidden, num_layers=layers, batch_first=True), \
            nn.Linear(d_hidden, d_out), act

    def forward(self, x):
        y = self.fc(self.rnn(x)[0])
        return torch.sigmoid(y) if self.act else y


def timegan(seqs, n_samples, seed, hidden=32, layers=2, iters=1500, batch=64, gamma=1.0):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    lo, hi = seqs.min((0, 1), keepdims=True), seqs.max((0, 1), keepdims=True)
    x_all = torch.tensor((seqs - lo) / (hi - lo + 1e-7), dtype=torch.float32)
    k, t, f = x_all.shape
    emb, rec, gen = _Rnn(f, hidden, hidden, layers), _Rnn(hidden, hidden, f, layers), _Rnn(f, hidden, hidden, layers)
    sup, dis = _Rnn(hidden, hidden, hidden, max(1, layers - 1)), _Rnn(hidden, hidden, 1, layers, act=False)
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
    batch_x = lambda: x_all[torch.randint(0, k, (min(batch, k),), generator=g)]  # noqa: E731
    noise = lambda m: torch.rand(m, t, f, generator=g)  # noqa: E731
    opt_er = torch.optim.Adam([*emb.parameters(), *rec.parameters()], lr=1e-3)
    for _ in range(iters):
        x = batch_x()
        loss = 10 * torch.sqrt(mse(rec(emb(x)), x))
        opt_er.zero_grad(); loss.backward(); opt_er.step()
    opt_s = torch.optim.Adam(sup.parameters(), lr=1e-3)
    for _ in range(iters):
        h = emb(batch_x()).detach()
        loss = mse(sup(h)[:, :-1], h[:, 1:])
        opt_s.zero_grad(); loss.backward(); opt_s.step()
    opt_g = torch.optim.Adam([*gen.parameters(), *sup.parameters()], lr=1e-3)
    opt_e = torch.optim.Adam([*emb.parameters(), *rec.parameters()], lr=1e-3)
    opt_d = torch.optim.Adam(dis.parameters(), lr=1e-3)
    for _ in range(iters):
        for _ in range(2):
            x = batch_x(); zz = noise(len(x))
            h, e_hat = emb(x), gen(zz)
            h_hat = sup(e_hat); x_hat = rec(h_hat)
            y_fake, y_fake_e = dis(h_hat), dis(e_hat)
            g_u = bce(y_fake, torch.ones_like(y_fake)) + gamma * bce(y_fake_e, torch.ones_like(y_fake_e))
            g_s = mse(sup(h)[:, :-1], h[:, 1:])
            g_v = (x_hat.std(0) - x.std(0)).abs().mean() + (x_hat.mean(0) - x.mean(0)).abs().mean()
            loss = g_u + 100 * torch.sqrt(g_s) + 100 * g_v
            opt_g.zero_grad(); loss.backward(); opt_g.step()
            x = batch_x(); h = emb(x)
            e_loss = 10 * torch.sqrt(mse(rec(h), x)) + 0.1 * mse(sup(h)[:, :-1], h[:, 1:])
            opt_e.zero_grad(); e_loss.backward(); opt_e.step()
        x = batch_x(); zz = noise(len(x))
        h, e_hat = emb(x).detach(), gen(zz).detach()
        h_hat = sup(e_hat).detach()
        y_real, y_fake, y_fake_e = dis(h), dis(h_hat), dis(e_hat)
        d_loss = (bce(y_real, torch.ones_like(y_real)) + bce(y_fake, torch.zeros_like(y_fake))
                  + gamma * bce(y_fake_e, torch.zeros_like(y_fake_e)))
        if d_loss.item() > 0.15:
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
    with torch.no_grad():
        out = rec(sup(gen(noise(n_samples)))).numpy()
    return out * (hi - lo + 1e-7) + lo


def synth_timegan(fold, n, seed, iters=1500):
    lib, seq, season = _real_sequences(fold)
    w = fold.window
    fake = timegan(np.concatenate([seq, season], -1), n, seed, iters=iters)
    counts = np.expm1(np.clip(fake[..., :N], 0.0, None))
    src = np.random.default_rng(seed).choice(lib, n)
    return to_pack(counts[:, :w].transpose(0, 2, 1), counts[:, w:].transpose(0, 2, 1),
                   np.clip(fake[:, w, N:], -1.0, 1.0), POPULATION[src - 1], src, fold)


def _lambda_stats(fold):
    lib = library_idx(fold)
    c = np.nan_to_num(CASES)
    x_raw = torch.tensor(np.stack([c[i - fold.window: i].T for i in lib]), dtype=torch.float32)
    y = torch.tensor(np.stack([c[i: i + HORIZON].T for i in lib]), dtype=torch.float32)
    pop = torch.tensor(POPULATION[lib - 1], dtype=torch.float32)
    st = seir_state(x_raw, pop, cumulative(lib))
    lams = []
    for h in range(HORIZON):
        lam = invert(st, pop, y[..., h])
        lams.append(lam)
        st = advance(st, lam)
    lam = torch.stack(lams, -1).numpy()
    log_lam = np.log(np.where(lam > 1e-8, lam, np.nan))
    return float(np.nanmedian(log_lam)), float(np.nanstd(np.diff(log_lam, axis=-1)))


def _simulate(fold, n, seed, alpha, med, step):
    rng = np.random.default_rng(seed)
    lib, w = library_idx(fold), fold.window
    c = np.nan_to_num(CASES)
    src = rng.choice(lib, n)
    x0 = torch.tensor(np.stack([c[i - w: i].T for i in src]), dtype=torch.float32)
    pop = POPULATION[src - 1]
    state0 = seir_state(x0, torch.tensor(pop, dtype=torch.float32), cumulative(src))
    walk = (np.cumsum(rng.normal(0, step, (n, 1, w + HORIZON)), -1)
            + np.cumsum(rng.normal(0, step, (n, N, w + HORIZON)), -1))
    lam = torch.tensor(np.exp(np.clip(med + walk, -25.0, LOG_LAMBDA_MAX)), dtype=torch.float32)
    _, inc = simulate_weeks(state0, lam)
    mu = (inc.numpy() * RHO * pop[..., None]).clip(1e-9, None)
    counts = rng.poisson(rng.gamma(shape=1.0 / alpha, scale=mu * alpha)).astype(np.float64)
    return counts[..., :w], counts[..., w:], seasonal_features(src + w), pop, src


def synth_seir(fold, n, seed, alpha=0.2):
    """Calibrated: the log-lambda centre and step are chosen so simulated growth matches training growth."""
    _, seq, _ = _real_sequences(fold)
    g = np.diff(np.log1p(np.expm1(seq).transpose(0, 2, 1)), axis=-1)
    target = (g.mean(), g.std())
    med, sd = _lambda_stats(fold)
    best = None
    for off in np.arange(-4.0, 1.01, 0.25):
        for mult in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            xs, ys, *_ = _simulate(fold, 400, 12345, 0.2, med + off, sd / np.sqrt(2.0) * mult)
            gs = np.diff(np.log1p(np.concatenate([xs, ys], -1)), axis=-1)
            err = (gs.mean() - target[0]) ** 2 + (gs.std() - target[1]) ** 2
            if best is None or err < best[0]:
                best = (err, off, mult)
    x_raw, y_raw, season, pop, src = _simulate(fold, n, seed, alpha, med + best[1], sd / np.sqrt(2.0) * best[2])
    return to_pack(x_raw, y_raw, season, pop, src, fold)
