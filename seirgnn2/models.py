"""Graph backbones and forecast heads, with the graph and the head independent.

A run is ``(backbone, head, loss, features)``. Holding three fixed and varying
the fourth is what makes a difference attributable, so every combination shares
one encoder shape, one training loop and one normalisation.

Heads
-----
``direct``    predict fold-scaled log1p cases for each of the 3 horizon weeks.
``residual``  predict a *correction* to last week's value in the same space.
              ``docs/decisions/0001`` records this as the single change that
              moved the project's own baseline from RMSE 66 to 45.
``foi``       predict a per-week force of infection and let the SEIR simulator
              turn it into incidence. Unlike the original implementation the
              force of infection varies across the horizon rather than being
              one scalar repeated three times.
``foi_res``   the same, but the simulator's incidence is blended onto the
              persistence anchor with a learned gate, so the physics supplies
              the *departure* from persistence rather than the whole signal.
``gated``     the no-physics twin of ``foi_res``: the encoder's direct forecast is
              blended onto the persistence anchor through the same learned gate,
              initialised the same way. ``foi_res`` minus ``gated`` is the effect
              of routing the departure through SEIR, with the anchor and the gate
              held fixed (docs/RESCUE_PLAN.md).
``foi_meta``  metapopulation SEIR (docs/PHYSICS_GNN_PLAN.md, P4): the encoder
              predicts each district's weekly transmission rate beta_i(t), and
              the simulator recomputes the force of infection every day as
              beta_i * sum_j C_ij I_j, with C a learned row-stochastic coupling
              initialised to the border graph -- districts are coupled inside
              the dynamics, as in MepoGNN and CausalGNN, instead of through an
              import term on the initial state. Gated onto persistence like
              ``foi_res``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
import backbones  # noqa: E402
import seir_sim  # noqa: E402

HEADS = ("direct", "residual", "gated", "foi", "foi_res", "foi_meta")
BACKBONES = ("none", "gcn", "gat", "adaptive", "hybrid", "linear", *backbones.REAL)
LAM_PARAMS = ("sigmoid", "log")
SEEDS = ("lagged", "recent", "decon")

#: Output distribution. ``point`` trains a squared error on log1p and scores
#: counts, which is biased low by construction: for ``log(1+y) = m + e``,
#: ``expm1(m)`` is the conditional *median* while RMSE is minimised by the
#: conditional *mean*. ``error_diagnosis.json`` measures that bias at -12.92 on
#: outbreak windows. ``nb`` trains a negative-binomial likelihood instead, whose
#: ``mu`` **is** the mean, so the point forecast needs no retransformation. It
#: is orthogonal to the head: the physics path can emit an NB mean too.
DISTS = ("point", "nb")

#: Input normalisation: the fold statistics as before, or reversible instance
#: normalisation per district-window (see ``Net``).
NORMS = ("fold", "revin_mean", "revin")

#: Seasonal features are the last four input channels whenever ``use_season`` is on
#: (``core.build_tensors`` appends them last); ``dseason`` reads them from there.
N_SEASON = 4

#: log of the median force of infection that reproduces the observed counts when
#: the simulator is inverted -- see ``diagnose_foi.py`` section 3. Used to centre
#: the ``log`` parameterisation so training starts at the data's own scale.
LAM_LOG0 = -9.36
LOG_LAMBDA_MAX = -1.95  # log(1/7), the original bound, kept as a hard clamp
META_LOG_BETA0 = -0.69  # log(0.5/day): the foi_meta transmission-rate centre at initialisation


class GraphLayer(nn.Module):
    """One message-passing step over a chosen adjacency.

    ``none`` is the control -- 25 independent series. ``adaptive`` is the Graph
    WaveNet construction ``softmax(ReLU(E1 E2^T))``; ``hybrid`` averages it with
    the hand-built district graph, which is how Graph WaveNet actually uses it.
    The node embeddings are allocated in every mode so the arms consume
    identical RNG draws and differ only by the graph.
    """

    def __init__(self, dim: int, n_nodes: int, mode: str, emb: int = 16):
        super().__init__()
        self.mode = mode
        self.lin = nn.Linear(dim, dim)
        self.e1 = nn.Parameter(torch.randn(n_nodes, emb) * 0.05)
        self.e2 = nn.Parameter(torch.randn(n_nodes, emb) * 0.05)

    def adjacency(self, fixed: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return torch.eye(fixed.shape[0], device=fixed.device, dtype=fixed.dtype)
        if self.mode == "gcn" or self.mode == "gat":
            return fixed
        learned = torch.softmax(torch.relu(self.e1 @ self.e2.T), dim=-1)
        return learned if self.mode == "adaptive" else 0.5 * fixed + 0.5 * learned

    def forward(self, h: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(self.adjacency(fixed) @ h))


class Net(nn.Module):
    """Encoder -> graph layers -> head.

    ``in_proj`` keeps a full hidden vector per node. The original implementation
    used ``nn.Linear(in_dim, 1)``, collapsing every covariate channel to one
    scalar before the graph ever saw it, which is why its input-level factor
    moved results by less than 1% RMSE.

    Remedies from ``docs/REMEDIES_PLAN.md``, each off by default so every
    earlier grid reproduces unchanged:

    ``norm``      ``fold`` (as before) or reversible instance normalisation --
                  ``revin_mean`` subtracts each district-window's mean log level
                  and adds it back on the output, ``revin`` also divides by the
                  window's spread with a learned affine (Kim et al. 2022).
    ``node_emb``  a learned identity per district joined at the head
                  (spatial identity, Shao et al. 2022).
    ``linear``    backbone that passes the inputs straight to a linear head: with
                  ``dist="nb"`` and ``node_emb`` this is a negative-binomial GLM
                  with district effects.
    ``aux_phys``  the SEIR decoder as an auxiliary head trained alongside a
                  direct forecast, rather than as the forecast itself
                  (Rodriguez et al. 2023). Its weight in the loss is this value.

    Architecture arms of ``docs/ARCH_PLAN.md``, also off by default:

    ``head_mlp``    hidden width of a 2-layer MLP output head in place of the single
                    linear layer shared by all districts, so exogenous inputs that
                    bypass the encoder can act nonlinearly.
    ``dseason``     a learned (district x 4 Fourier terms x horizon) table added to
                    the output, initialised to zero: a separate seasonal curve per
                    district, which a shared linear head cannot express.
    ``global_ctx``  each district's encoding is concatenated with the mean encoding
                    over all districts -- a virtual node carrying the national picture.
    """

    def __init__(self, in_dim: int, n_nodes: int, horizon: int = 3, hidden: int = 64,
                 backbone: str = "gcn", head: str = "residual", layers: int = 2,
                 dropout: float = 0.1, lambda_max: float = 1.0 / 7.0, rho: float = 1.0 / 11.0,
                 lam_param: str = "sigmoid", state_fit: bool = False,
                 window: int = 3, edge_index: torch.Tensor | None = None,
                 dist: str = "point", norm: str = "fold", node_emb: int = 0,
                 aux_phys: float = 0.0, head_mlp: int = 0, dseason: bool = False,
                 global_ctx: bool = False):
        super().__init__()
        if norm not in NORMS:
            raise ValueError(f"unknown norm {norm!r}; expected one of {NORMS}")
        if (norm != "fold" or aux_phys) and head != "direct":
            raise ValueError("norm and aux_phys are defined for the direct head only")
        if (head_mlp or global_ctx) and head not in ("direct", "residual"):
            raise ValueError("head_mlp and global_ctx are defined for direct/residual heads")
        if dseason and head not in ("direct", "residual", "foi_meta"):
            raise ValueError("dseason is defined for direct, residual and foi_meta heads")
        self.head, self.horizon, self.lambda_max, self.rho = head, horizon, lambda_max, rho
        self.lam_param, self.state_fit, self.backbone = lam_param, state_fit, backbone
        self.dist, self.norm, self.window, self.aux_phys = dist, norm, window, aux_phys
        self.global_ctx = global_ctx
        if backbone in backbones.REAL:
            # A published architecture supplies the representation; the toy
            # encoder is bypassed entirely rather than stacked underneath it.
            self.real = backbones.Real(backbone, n_nodes, window, in_dim - window,
                                       hidden, edge_index)
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
        self.node_emb = (nn.Parameter(torch.randn(n_nodes, node_emb) * 0.1)
                         if node_emb else None)
        feat += node_emb
        if norm == "revin":
            self.rev_gamma = nn.Parameter(torch.ones(1))
            self.rev_beta = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(dropout)
        self.out = (nn.Sequential(nn.Linear(feat, head_mlp), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(head_mlp, horizon))
                    if head_mlp else nn.Linear(feat, horizon))
        self.disp = nn.Linear(feat, horizon) if dist == "nb" else None
        self.out_phys = nn.Linear(feat, horizon) if aux_phys else None
        if head == "gated":
            self.alpha = nn.Parameter(torch.tensor(-2.0))   # same gate and init as foi_res
        if head in ("foi", "foi_res", "foi_meta") or aux_phys:
            self.alpha = nn.Parameter(torch.tensor(-2.0))   # gate, through sigmoid
            self.beta = nn.Parameter(torch.tensor(0.0))     # log-scale spatial import weight
            # ``diagnose_foi.py`` inverts the simulator for the lambda that
            # reproduces each true count: the median is 6e-4 of lambda_max, so
            # sigmoid starts 809x too high and 57% of cells need |raw| > 6,
            # where the sigmoid's gradient is 150x below its maximum. LAM_LOG0
            # is that measured median, and ``log`` parameterisation makes raw
            # an offset in log-lambda -- unit gradient everywhere, with the
            # upper bound kept by a clamp instead of a squash.
            self.lam_bias = nn.Parameter(torch.tensor(LAM_LOG0))
            # The lambda=0 floor overshoots 14% of targets because E0 is seeded
            # from cases[t-2]/rho and half of it matures within the week. These
            # let the fit lower that floor rather than fight it.
            self.e_scale = nn.Parameter(torch.tensor(0.0))
            self.rho_scale = nn.Parameter(torch.tensor(0.0))
        self.dseason = (nn.Parameter(torch.zeros(n_nodes, N_SEASON, horizon))
                        if dseason else None)
        if head == "foi_meta":
            # beta ~ lambda / I: the inverted median lambda (~9e-5/day) over a typical
            # infectious fraction (~2e-4) puts beta near 0.5/day at initialisation.
            self.beta_bias = nn.Parameter(torch.tensor(float(META_LOG_BETA0)))
            self.c_logit = nn.Parameter(torch.zeros(n_nodes, n_nodes))

    def _instance_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and spread of each district-window's case channels, ``(B, N, 1)``.

        The spread is floored: three weekly points can be nearly equal, and
        dividing by a spread of ~0 would turn noise into a large input.
        """
        cases = x[..., : self.window]
        m = cases.mean(-1, keepdim=True)
        s = torch.sqrt(cases.var(-1, unbiased=False, keepdim=True) + 0.01)
        return m, s

    def encode(self, x: torch.Tensor, fixed: torch.Tensor,
               edge: torch.Tensor | None = None) -> torch.Tensor:
        if self.real is not None:
            h = self.drop(self.real(x, edge))
        elif self.backbone == "linear":
            h = x
        else:
            h = self.in_proj(x)
            for layer in self.graph:
                h = h + self.drop(layer(h, fixed))
        if self.global_ctx:
            h = torch.cat([h, h.mean(1, keepdim=True).expand_as(h)], dim=-1)
        if self.node_emb is not None:
            h = torch.cat([h, self.node_emb.expand(h.shape[0], -1, -1)], dim=-1)
        return h

    def _physics(self, raw: torch.Tensor, st0: torch.Tensor, pop: torch.Tensor,
                 mean: float, std: float) -> torch.Tensor:
        """Head output -> per-week force of infection -> SEIR incidence, in fold-z log1p."""
        if self.lam_param == "log":
            lam = torch.exp((raw + self.lam_bias).clamp(-25.0, LOG_LAMBDA_MAX))
        else:
            lam = self.lambda_max * torch.sigmoid(raw)
        if self.state_fit and st0 is not None:
            s, e, i, r = st0.unbind(-1)
            scaled = e * torch.sigmoid(self.e_scale) * 2.0
            st0 = torch.stack([s, scaled, i, (r + e - scaled).clamp_min(0.0)], dim=-1)
        if st0 is not None:
            # Spatial import, scaled by the *log* of the neighbour infected
            # fraction so the term is not numerically inert. The original added
            # alpha * mean(I/N) directly, with I/N ~ 1e-4 against lambda ~ 7e-2,
            # a relative contribution of ~1e-4 -- which is why its explicit and
            # implicit coupling arms agreed to five decimal places.
            imp = st0[..., 2].clamp_min(1e-9).log().unsqueeze(-1)
            lam = lam * torch.exp(self.beta * (imp - imp.mean()) * 0.1)
        _, inc = seir_sim.simulate_weeks(st0, lam, omega=0.7 / 7.0, gamma=1.0 / 7.0, substeps=7)
        rho = self.rho * torch.exp(self.rho_scale) if self.state_fit else self.rho
        counts = (inc * rho * pop.unsqueeze(-1)).clamp_min(0.0)
        return (torch.log1p(counts) - mean) / std

    def _meta_physics(self, raw: torch.Tensor, st0: torch.Tensor, pop: torch.Tensor,
                      mean: float, std: float, fixed: torch.Tensor) -> torch.Tensor:
        """Metapopulation SEIR: daily force of infection beta_i * sum_j C_ij I_j.

        C starts as the row-normalised border graph with self-loops; log of it is
        the prior and ``c_logit`` a learned correction, so pairs with no shared
        border start at ~0 weight (log 1e-9) but can be learned if the data want it.
        """
        beta = torch.exp((raw + self.beta_bias).clamp(-12.0, 2.0))               # (B,N,H) per day
        if self.state_fit and st0 is not None:
            s, e, i, r = st0.unbind(-1)
            scaled = e * torch.sigmoid(self.e_scale) * 2.0
            st0 = torch.stack([s, scaled, i, (r + e - scaled).clamp_min(0.0)], dim=-1)
        coupling = torch.softmax(torch.log(fixed.clamp_min(1e-9)) + self.c_logit, dim=-1)
        _, inc, _ = seir_sim.simulate_closed_loop(st0, beta, omega=0.7 / 7.0, gamma=1.0 / 7.0,
                                                  substeps=7, coupling=coupling)
        rho = self.rho * torch.exp(self.rho_scale) if self.state_fit else self.rho
        counts = (inc * rho * pop.unsqueeze(-1)).clamp_min(0.0)
        return (torch.log1p(counts) - mean) / std

    def forward(self, x: torch.Tensor, fixed: torch.Tensor, p_z: torch.Tensor,
                st0: torch.Tensor | None = None, pop: torch.Tensor | None = None,
                mean: float = 0.0, std: float = 1.0, edge: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """``(pred_z, raw_dispersion, aux_z)`` -- fold-scaled log1p ``(B, N, H)``.

        Keeping every head in one output space is what lets a single loss and a
        single metric compare them; the physics heads convert their raw-count
        incidence back into that space before returning. ``raw_dispersion`` is
        ``None`` unless ``dist == "nb"``; ``aux_z`` is the auxiliary SEIR
        forecast when ``aux_phys`` is set, and ``None`` otherwise.
        """
        season = x[..., -N_SEASON:] if self.dseason is not None else None
        if self.norm != "fold":
            m, s = self._instance_stats(x)
            cases = x[..., : self.window] - m
            if self.norm == "revin":
                cases = cases / s * self.rev_gamma + self.rev_beta
            x = torch.cat([cases, x[..., self.window :]], dim=-1)

        h = self.encode(x, fixed, edge)
        raw = self.out(h)
        if self.dseason is not None:
            raw = raw + torch.einsum("bnk,nkh->bnh", season, self.dseason)
        disp = self.disp(h) if self.disp is not None else None
        aux = (self._physics(self.out_phys(h), st0, pop, mean, std)
               if self.out_phys is not None else None)

        if self.head == "direct":
            if self.norm == "revin_mean":
                raw = raw + m
            elif self.norm == "revin":
                gamma = self.rev_gamma.abs() + 1e-3
                raw = (raw - self.rev_beta) / gamma * s + m
            return raw, disp, aux
        if self.head == "residual":
            return p_z + raw, disp, aux
        if self.head == "gated":
            return p_z + torch.sigmoid(self.alpha) * (raw - p_z), disp, aux
        if self.head == "foi_meta":
            phys_z = self._meta_physics(raw, st0, pop, mean, std, fixed)
            gate = torch.sigmoid(self.alpha)
            return p_z + gate * (phys_z - p_z), disp, aux
        phys_z = self._physics(raw, st0, pop, mean, std)
        if self.head == "foi":
            return phys_z, disp, aux
        gate = torch.sigmoid(self.alpha)
        return p_z + gate * (phys_z - p_z), disp, aux


def seir_state(cases_raw: torch.Tensor, pop: torch.Tensor, cum: torch.Tensor,
               rho: float = 1.0 / 11.0, s0: float = 1.0 - 0.682,
               mode: str = "lagged", omega: float = 0.7 / 7.0,
               gamma: float = 1.0 / 7.0) -> torch.Tensor:
    """``(B, N, 4)`` initial ``(S, E, I, R)`` fractions from observed counts.

    ``cases_raw`` is the window's case history, newest last; ``cum`` the
    cumulative reported cases strictly before the forecast origin. Reporting
    covers a fraction ``rho`` of true infections, so counts are inflated by
    ``1/rho`` before being turned into fractions.

    The seeding is what sets the head's reachable floor: with ``lambda = 0`` the
    week's incidence is whatever ``E`` matures, so ``E0`` alone decides how low
    the head can predict. ``diagnose_foi.py`` measures 14-16% of targets already
    below that floor under ``lagged``.

    ``lagged``  the original: ``I0`` from ``cases[t-1]``, ``E0`` from
                ``cases[t-2]`` -- the *stalest* week in the window, so the floor
                lags hardest exactly when the series turns.
    ``recent``  both from ``cases[t-1]``, dropping the staleness but keeping the
                flow-as-stock reading.
    ``decon``   stocks from residence times: a weekly report of ``c`` implies
                ``c / (7 rho)`` onsets a day, so ``I = onsets / gamma`` and
                ``E = onsets / omega``. Dimensionally right, and the one of the
                three that is not an ansatz.
    """
    scale = rho * pop
    newest = cases_raw[..., -1]
    if mode == "lagged":
        i_raw, e_raw = newest, cases_raw[..., -2]
    elif mode == "recent":
        i_raw = e_raw = newest
    elif mode == "decon":
        onsets = newest / 7.0
        i_raw, e_raw = onsets / gamma, onsets / omega
    else:
        raise ValueError(f"unknown seeding mode {mode!r}; expected one of {SEEDS}")
    i0 = (i_raw / scale).clamp(1e-6, 0.5)
    e0 = (e_raw / scale).clamp(1e-6, 0.5)
    s = (s0 - cum / scale).clamp(0.01, 1.0)
    r = (1.0 - s - e0 - i0).clamp(0.0, 1.0)
    return torch.stack([s, e0, i0, r], dim=-1)
