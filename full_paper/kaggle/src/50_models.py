# %% [markdown]
# ## 5. The SEIR model and the forecast heads
#
# ### 5.1 A differentiable SEIR simulator
#
# $\dot S = -\lambda S,\ \dot E = \lambda S - \omega E,\ \dot I = \omega E - \gamma I,\ \dot R = \gamma I$,
# stepped **daily** (7 sub-steps per week) with exponential flows — each step moves
# $S(1-e^{-\lambda\,dt})$ out of S, and so on — which is the exact outflow for rates held
# over a step and keeps every compartment non-negative. Rates per day: incubation
# $\omega = 0.1$, recovery $\gamma = 1/7$. Weekly incidence is the flow E → I over the
# week. The metapopulation variant recomputes the force of infection every day as
# $\beta_i \sum_j C_{ij} I_j / N_j$.

# %%
DAYS_PER_WEEK = 7
OMEGA, GAMMA, RHO = 0.7 / 7.0, 1.0 / 7.0, 1.0 / 11.0


def seir_step(state, lam, omega, gamma, dt):
    s, e, i, r = state.unbind(-1)
    infect = s * (1.0 - torch.exp(-lam * dt))
    onset = e * (1.0 - torch.exp(torch.as_tensor(-omega * dt, dtype=state.dtype)))
    recover = i * (1.0 - torch.exp(torch.as_tensor(-gamma * dt, dtype=state.dtype)))
    return torch.stack([s - infect, e + infect - onset, i + onset - recover, r + recover], -1), onset


def simulate_weeks(state0, foi, omega=OMEGA, gamma=GAMMA, substeps=7):
    """One force of infection per district per week; returns (states, weekly incidence)."""
    dt = DAYS_PER_WEEK / substeps
    states, weekly, state = [state0], [], state0
    for w in range(foi.shape[-1]):
        lam, total = foi[..., w], torch.zeros_like(foi[..., w])
        for _ in range(substeps):
            state, onset = seir_step(state, lam, omega, gamma, dt)
            total = total + onset
        states.append(state)
        weekly.append(total)
    return torch.stack(states, -2), torch.stack(weekly, -1)


def simulate_closed_loop(state0, beta, omega=OMEGA, gamma=GAMMA, substeps=7, coupling=None):
    """Mass action, force of infection recomputed every day (optionally coupled across districts)."""
    dt = DAYS_PER_WEEK / substeps
    state, weekly = state0, []
    for w in range(beta.shape[-1]):
        total = torch.zeros(state.shape[:-1], dtype=state.dtype)
        for _ in range(substeps):
            prevalence = state[..., 2] / state.sum(-1).clamp_min(1.0)
            if coupling is not None:
                prevalence = prevalence @ coupling.T
            state, onset = seir_step(state, beta[..., w] * prevalence, omega, gamma, dt)
            total = total + onset
        weekly.append(total)
    return None, torch.stack(weekly, -1), None


def seir_state(cases_raw, pop, cum, rho=RHO, s0=1.0 - 0.682):
    """Initial (S, E, I, R) fractions from observed counts: I0 from cases[t-1], E0 from cases[t-2].

    Reporting covers a fraction rho of infections; S starts at 1 - 0.682 (a 2013-14 Colombo
    serosurvey, before the study period) less cumulative infections implied by reported cases.
    """
    scale = rho * pop
    i0 = (cases_raw[..., -1] / scale).clamp(1e-6, 0.5)
    e0 = (cases_raw[..., -2] / scale).clamp(1e-6, 0.5)
    s = (s0 - cum / scale).clamp(0.01, 1.0)
    return torch.stack([s, e0, i0, (1.0 - s - e0 - i0).clamp(0.0, 1.0)], -1)


# %% [markdown]
# ### 5.2 The network: encoder → head
#
# Every model is an encoder (section 4, or the toy graph layers used by the first
# screen) followed by one of six heads. All heads output fold-standardised
# $\log(1+y)$ for the three horizon weeks; $p$ is last week's value in the same space.
#
# | head | forecast | role |
# |---|---|---|
# | `direct` | $f(h)$ | the published GNNs |
# | `residual` | $p + f(h)$ | persistence anchor |
# | `gated` | $p + \sigma(a)\,(f(h) - p)$ | anchor + learned gate, **no physics** |
# | `foi` | SEIR$(\lambda = e^{f(h)+b})$ | the pure SEIR decoder |
# | `foi_res` | $p + \sigma(a)\,(\mathrm{SEIR}(\lambda) - p)$ | gated SEIR: our SEIR-GNN, or SEIR-LSTM with the LSTM |
# | `foi_meta` | as `foi_res`, metapopulation SEIR with learned coupling | physics inside the dynamics |
#
# Options (all off unless an experiment sets them): negative-binomial output
# (`dist="nb"`), reversible instance normalisation (`norm`), a district identity
# embedding (`node_emb`), a nonlinear head (`head_mlp`), district seasonal curves
# (`dseason`), a national context vector (`global_ctx`) and an auxiliary SEIR loss
# (`aux_phys`).

# %%
LAM_LOG0 = -9.36          # log of the median force of infection that reproduces observed counts
LOG_LAMBDA_MAX = -1.95    # log(1/7), a hard clamp
META_LOG_BETA0 = -0.69    # log(0.5/day): the metapopulation transmission rate at initialisation
N_SEASON = 4


class GraphLayer(nn.Module):
    """One message-passing step: `none` (identity), `gcn` (fixed graph), `adaptive` (learned), `hybrid`."""

    def __init__(self, dim, n_nodes, mode, emb=16):
        super().__init__()
        self.mode = mode
        self.lin = nn.Linear(dim, dim)
        self.e1 = nn.Parameter(torch.randn(n_nodes, emb) * 0.05)
        self.e2 = nn.Parameter(torch.randn(n_nodes, emb) * 0.05)

    def forward(self, h, fixed):
        if self.mode == "none":
            a = torch.eye(fixed.shape[0])
        elif self.mode == "gcn":
            a = fixed
        else:
            learned = torch.softmax(torch.relu(self.e1 @ self.e2.T), -1)
            a = learned if self.mode == "adaptive" else 0.5 * fixed + 0.5 * learned
        return torch.relu(self.lin(a @ h))


class Net(nn.Module):
    def __init__(self, in_dim, n_nodes, horizon=3, hidden=64, backbone="gcn", head="residual", layers=2,
                 dropout=0.1, lam_param="sigmoid", state_fit=False, window=3, dist="point", norm="fold",
                 node_emb=0, aux_phys=0.0, head_mlp=0, dseason=False, global_ctx=False):
        super().__init__()
        self.head, self.backbone, self.dist, self.norm, self.window = head, backbone, dist, norm, window
        self.lam_param, self.state_fit, self.aux_phys, self.global_ctx = lam_param, state_fit, aux_phys, global_ctx
        if backbone in ENCODERS:
            self.real = Encoder(backbone, n_nodes, window, in_dim - window, hidden, EDGE_INDEX)
            self.in_proj, self.graph = None, nn.ModuleList()
            feat = self.real.out_dim
        elif backbone == "linear":
            self.real, self.in_proj, self.graph = None, None, nn.ModuleList()
            feat = in_dim
        else:
            self.real = None
            self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
            self.graph = nn.ModuleList(GraphLayer(hidden, n_nodes, backbone) for _ in range(layers))
            feat = hidden
        if global_ctx:
            feat *= 2
        self.node_emb = nn.Parameter(torch.randn(n_nodes, node_emb) * 0.1) if node_emb else None
        feat += node_emb
        if norm == "revin":
            self.rev_gamma = nn.Parameter(torch.ones(1))
            self.rev_beta = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(dropout)
        self.out = (nn.Sequential(nn.Linear(feat, head_mlp), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_mlp, horizon))
                    if head_mlp else nn.Linear(feat, horizon))
        self.disp = nn.Linear(feat, horizon) if dist == "nb" else None
        self.out_phys = nn.Linear(feat, horizon) if aux_phys else None
        if head == "gated":
            self.alpha = nn.Parameter(torch.tensor(-2.0))
        if head in ("foi", "foi_res", "foi_meta") or aux_phys:
            self.alpha = nn.Parameter(torch.tensor(-2.0))            # gate, through a sigmoid: starts at 0.12
            self.beta = nn.Parameter(torch.tensor(0.0))              # log-scale spatial import weight
            self.lam_bias = nn.Parameter(torch.tensor(LAM_LOG0))
            self.e_scale = nn.Parameter(torch.tensor(0.0))           # learned scale on E0
            self.rho_scale = nn.Parameter(torch.tensor(0.0))         # learned scale on reporting
        self.dseason = nn.Parameter(torch.zeros(n_nodes, N_SEASON, horizon)) if dseason else None
        if head == "foi_meta":
            self.beta_bias = nn.Parameter(torch.tensor(float(META_LOG_BETA0)))
            self.c_logit = nn.Parameter(torch.zeros(n_nodes, n_nodes))

    def encode(self, x, fixed):
        if self.real is not None:
            h = self.drop(self.real(x))
        elif self.backbone == "linear":
            h = x
        else:
            h = self.in_proj(x)
            for layer in self.graph:
                h = h + self.drop(layer(h, fixed))
        if self.global_ctx:
            h = torch.cat([h, h.mean(1, keepdim=True).expand_as(h)], -1)
        if self.node_emb is not None:
            h = torch.cat([h, self.node_emb.expand(h.shape[0], -1, -1)], -1)
        return h

    def _fit_state(self, st0):
        s, e, i, r = st0.unbind(-1)
        scaled = e * torch.sigmoid(self.e_scale) * 2.0
        return torch.stack([s, scaled, i, (r + e - scaled).clamp_min(0.0)], -1)

    def _physics(self, raw, st0, pop, mean, std):
        lam = (torch.exp((raw + self.lam_bias).clamp(-25.0, LOG_LAMBDA_MAX)) if self.lam_param == "log"
               else (1.0 / 7.0) * torch.sigmoid(raw))
        if self.state_fit:
            st0 = self._fit_state(st0)
        imp = st0[..., 2].clamp_min(1e-9).log().unsqueeze(-1)       # spatial import on the log infected fraction
        lam = lam * torch.exp(self.beta * (imp - imp.mean()) * 0.1)
        _, inc = simulate_weeks(st0, lam)
        rho = RHO * torch.exp(self.rho_scale) if self.state_fit else RHO
        return (torch.log1p((inc * rho * pop.unsqueeze(-1)).clamp_min(0.0)) - mean) / std

    def _meta_physics(self, raw, st0, pop, mean, std, fixed):
        beta = torch.exp((raw + self.beta_bias).clamp(-12.0, 2.0))
        if self.state_fit:
            st0 = self._fit_state(st0)
        coupling = torch.softmax(torch.log(fixed.clamp_min(1e-9)) + self.c_logit, -1)
        _, inc, _ = simulate_closed_loop(st0, beta, coupling=coupling)
        rho = RHO * torch.exp(self.rho_scale) if self.state_fit else RHO
        return (torch.log1p((inc * rho * pop.unsqueeze(-1)).clamp_min(0.0)) - mean) / std

    def forward(self, x, fixed, p_z, st0=None, pop=None, mean=0.0, std=1.0):
        season = x[..., -N_SEASON:] if self.dseason is not None else None
        if self.norm != "fold":
            cases = x[..., : self.window]
            m = cases.mean(-1, keepdim=True)
            s = torch.sqrt(cases.var(-1, unbiased=False, keepdim=True) + 0.01)
            cases = cases - m
            if self.norm == "revin":
                cases = cases / s * self.rev_gamma + self.rev_beta
            x = torch.cat([cases, x[..., self.window:]], -1)
        h = self.encode(x, fixed)
        raw = self.out(h)
        if self.dseason is not None:
            raw = raw + torch.einsum("bnk,nkh->bnh", season, self.dseason)
        disp = self.disp(h) if self.disp is not None else None
        aux = self._physics(self.out_phys(h), st0, pop, mean, std) if self.out_phys is not None else None
        if self.head == "direct":
            if self.norm == "revin_mean":
                raw = raw + m
            elif self.norm == "revin":
                raw = (raw - self.rev_beta) / (self.rev_gamma.abs() + 1e-3) * s + m
            return raw, disp, aux
        if self.head == "residual":
            return p_z + raw, disp, aux
        if self.head == "gated":
            return p_z + torch.sigmoid(self.alpha) * (raw - p_z), disp, aux
        if self.head == "foi_meta":
            phys = self._meta_physics(raw, st0, pop, mean, std, fixed)
            return p_z + torch.sigmoid(self.alpha) * (phys - p_z), disp, aux
        phys = self._physics(raw, st0, pop, mean, std)
        if self.head == "foi":
            return phys, disp, aux
        return p_z + torch.sigmoid(self.alpha) * (phys - p_z), disp, aux


# %% [markdown]
# ### 5.3 Losses
#
# Squared error on the standardised $\log(1+y)$ (`mse_z`), or the **negative-binomial**
# likelihood (`nb`): with $\mu = e^{\hat z \sigma + m} - 1$ and a learned dispersion
# $\alpha \in [10^{-4}, 2]$, it minimises
# $-\log \mathrm{NB2}(y; \mu, \alpha)$. Squared error on $\log(1+y)$ makes
# $e^{\hat z}$ a conditional *median*, while RMSE on counts rewards the conditional
# *mean*; the NB's $\mu$ is the mean. The spatial physics penalty adds
# $\lambda_s \sum_{(i,j)} A_{ij}(\hat y_i/s_i - \hat y_j/s_j)^2$ over bordering districts,
# with $s_i$ a district's training-mean cases (or the same in $\log(1+\cdot)$).

# %%
ALPHA_MIN, ALPHA_MAX, LOG_MU_MAX = 1e-4, 2.0, 12.0


def nb_nll(mu, alpha, target, weight=None):
    y, r = target.clamp_min(0.0), 1.0 / alpha
    log_r_mu = torch.log(r + mu)
    nll = -(torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0)
            + r * (torch.log(r) - log_r_mu) + y * (torch.log(mu) - log_r_mu))
    return nll.mean() if weight is None else (nll * weight).sum() / weight.sum()


def loss_fn(kind, pred_z, batch, mean, std, disp=None):
    if kind == "nb":
        mu = torch.expm1(torch.clamp(pred_z * std + mean, -1.0, LOG_MU_MAX)).clamp_min(1e-6)
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * torch.sigmoid(disp)
        return nb_nll(mu, alpha, batch["y_raw"], batch.get("w"))
    if kind == "mse_z":
        return F.mse_loss(pred_z, batch["y_z"])
    raise ValueError(kind)


def spatial_penalty(counts, adj, scales, kind):
    n = counts.shape[1]
    if kind == "ratio":
        rel = counts / scales.view(1, n, 1).clamp_min(1.0)
    else:
        rel = torch.log1p(counts.clamp_min(0.0)) - torch.log1p(scales.clamp_min(0.0)).view(1, n, 1)
    diff = rel.unsqueeze(2) - rel.unsqueeze(1)
    weighted = adj.unsqueeze(0).unsqueeze(-1) * diff.pow(2)
    return weighted.sum() / (weighted.shape[0] * weighted.shape[-1] * adj.abs().sum().clamp_min(1e-8))


def to_counts(pred_z, mean, std):
    return torch.expm1(torch.clamp(pred_z * std + mean, -1.0, 12.0)).clamp_min(0.0).detach().numpy()
