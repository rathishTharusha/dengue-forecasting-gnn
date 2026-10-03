# %% [markdown]
# ### 5.4 Why a pure SEIR decoder cannot work here (measured before any training)
#
# For a given initial state, the simulator's weekly incidence rises monotonically with
# the force of infection λ, so λ can be **inverted by bisection** for the value that
# reproduces each observed count exactly. That answers two questions without training
# anything:
#
# * **Reach.** With transmission switched off (λ = 0), the exposed people already seeded
#   from recent cases still produce some cases. Any target below that floor is
#   unreachable at any network output.
# * **Learnability.** How much of the required λ (in logs) is predictable from last
#   week's cases, compared with how predictable the target itself is.

# %%
def incidence(state, lam):
    _, inc = simulate_weeks(state, lam.unsqueeze(-1))
    return inc[..., 0]


def advance(state, lam):
    st, _ = simulate_weeks(state, lam.unsqueeze(-1))
    return st[..., 1, :]


def invert(state, pop, target, iters=60):
    lo, hi = torch.zeros_like(target), torch.full_like(target, 10.0 / 7.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        low = incidence(state, mid) * (RHO * pop) < target
        lo, hi = torch.where(low, mid, lo), torch.where(low, hi, mid)
    return 0.5 * (lo + hi)


def r2(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    a = np.stack([x[ok], np.ones(ok.sum())], 1)
    res = y[ok] - a @ np.linalg.lstsq(a, y[ok], rcond=None)[0]
    return float(1.0 - res.var() / y[ok].var())


def cumulative_before(idx):
    c = np.nan_to_num(CASES)
    return torch.tensor(np.stack([c[:i].sum(0) for i in idx]), dtype=torch.float32)


diag = []
for fold in build_folds():
    pack = build_tensors(fold, "train")
    st = seir_state(pack["x_raw"], pack["pop"], cumulative_before(pack["idx"]))
    y, pop = pack["y_raw"], pack["pop"]
    floor, s_lo, lam_star, s_hi = [], st, [], st
    for h in range(HORIZON):
        z = torch.zeros(st.shape[:-1])
        floor.append(incidence(s_lo, z) * (RHO * pop))
        s_lo = advance(s_lo, z)
        lam = invert(s_hi, pop, y[..., h])
        lam_star.append(lam)
        s_hi = advance(s_hi, lam)
    floor, lam_star = torch.stack(floor, -1), torch.stack(lam_star, -1)
    last = pack["x_raw"][..., -1].numpy()
    lg = np.log(np.clip(lam_star[..., 0].numpy(), 1e-8, None))
    diag.append({"origin": fold.origin,
                 "targets below the lambda=0 floor (%)": 100 * float((y < floor).float().mean()),
                 "cells needing lambda = 0 (%)": 100 * float((lam_star <= 1e-6 / 7).float().mean()),
                 "r2 of log lambda* on log cases[t-1]": r2(np.log1p(last).ravel(), lg.ravel()),
                 "r2 of log cases[t] on log cases[t-1]": r2(np.log1p(last).ravel(),
                                                             np.log1p(y[..., 0].numpy()).ravel()),
                 "S at its 0.01 clamp (%)": 100 * float((st[..., 0] <= 0.0100001).float().mean())})
DIAG = pd.DataFrame(diag).set_index("origin")
print(DIAG.round(3).to_string())
