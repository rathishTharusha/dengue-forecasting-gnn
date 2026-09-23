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

HEADS = ("direct", "residual", "foi", "foi_res")
BACKBONES = ("none", "gcn", "gat", "adaptive", "hybrid", *backbones.REAL)
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

#: log of the median force of infection that reproduces the observed counts when
#: the simulator is inverted -- see ``diagnose_foi.py`` section 3. Used to centre
#: the ``log`` parameterisation so training starts at the data's own scale.
LAM_LOG0 = -9.36
LOG_LAMBDA_MAX = -1.95  # log(1/7), the original bound, kept as a hard clamp


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
    """

    def __init__(self, in_dim: int, n_nodes: int, horizon: int = 3, hidden: int = 64,
                 backbone: str = "gcn", head: str = "residual", layers: int = 2,
                 dropout: float = 0.1, lambda_max: float = 1.0 / 7.0, rho: float = 1.0 / 11.0,
                 lam_param: str = "sigmoid", state_fit: bool = False,
                 window: int = 3, edge_index: torch.Tensor | None = None,
                 dist: str = "point"):
        super().__init__()
        self.head, self.horizon, self.lambda_max, self.rho = head, horizon, lambda_max, rho
        self.lam_param, self.state_fit, self.backbone = lam_param, state_fit, backbone
        self.dist = dist
        if backbone in backbones.REAL:
            # A published architecture supplies the representation; the toy
            # encoder is bypassed entirely rather than stacked underneath it.
            self.real = backbones.Real(backbone, n_nodes, window, in_dim - window,
                                       hidden, edge_index)
            self.in_proj, self.graph = None, nn.ModuleList()
            feat = self.real.out_dim
        else:
            self.real = None
            self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
            self.graph = nn.ModuleList(GraphLayer(hidden, n_nodes, backbone) for _ in range(layers))
            feat = hidden
        self.drop = nn.Dropout(dropout)
        out_dim = horizon
        self.out = nn.Linear(feat, out_dim)
        self.disp = nn.Linear(feat, horizon) if dist == "nb" else None
        if head in ("foi", "foi_res"):
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

    def encode(self, x: torch.Tensor, fixed: torch.Tensor,
               edge: torch.Tensor | None = None) -> torch.Tensor:
        if self.real is not None:
            return self.drop(self.real(x, edge))
        h = self.in_proj(x)
        for layer in self.graph:
            h = h + self.drop(layer(h, fixed))
        return h

    def forward(self, x: torch.Tensor, fixed: torch.Tensor, p_z: torch.Tensor,
                st0: torch.Tensor | None = None, pop: torch.Tensor | None = None,
                mean: float = 0.0, std: float = 1.0,
                edge: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(pred_z, raw_dispersion)`` -- fold-scaled log1p predictions ``(B, N, H)``.

        Keeping every head in one output space is what lets a single loss and a
        single metric compare them; the physics heads convert their raw-count
        incidence back into that space before returning. The second element is
        ``None`` unless ``dist == "nb"``, in which case it carries the
        unconstrained dispersion logit the likelihood needs.
        """
        h = self.encode(x, fixed, edge)
        raw = self.out(h)
        disp = self.disp(h) if self.disp is not None else None

        if self.head == "direct":
            return raw, disp
        if self.head == "residual":
            return p_z + raw, disp

        # Physics heads: raw -> per-week force of infection -> SEIR incidence.
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
        phys_z = (torch.log1p(counts) - mean) / std
        if self.head == "foi":
            return phys_z, disp
        gate = torch.sigmoid(self.alpha)
        return p_z + gate * (phys_z - p_z), disp


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
