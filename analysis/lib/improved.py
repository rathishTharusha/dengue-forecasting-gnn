"""Architecture increments, each justified by a paper or an EDA finding.

Every option here has a citation. Nothing is included because it is fashionable
or because it might help; if an increment cannot be traced to something we read
or measured, it is not in this module.

Increments
----------
``per_horizon_heads``
    One output layer per forecast step instead of one layer emitting all of
    them. GulMohamed et al. Eq. (11): "separate output layers can produce
    predictions for 1-week, 2-week and 4-week ahead intervals". Motivated here by
    EDA F6 -- autocorrelation falls from r=0.92 at lag 1 to 0.82 at lag 3, so the
    three horizons are not the same problem and a shared head averages over that.

``temporal_attention``
    Additive attention over the window before the head. GulMohamed et al.
    Eq. (6)-(8). Their Table 6 ablation attributes +0.5 RMSE to removing it --
    the smallest of their three, so this is the increment we expect least from.

``probabilistic``
    Gaussian head emitting mean and log-variance, with 95% intervals from
    Eq. (14). GulMohamed et al. Eq. (12)-(13). ``docs/ROADMAP.md`` requires
    CRPS/PICP/MPIW from Phase 3 onward, and a point forecast cannot supply them.

``mask_artifact`` -- **removed, it was a no-op**
    Intended to drop windows overlapping week 395 from *training*. Measured: the
    artifact falls in the **test** set of the origin-0.85 fold and in **no**
    training set under this protocol, so masking training changes nothing. The
    increment is gone; :func:`artifact_windows` replaces it, on the evaluation
    side where the artifact actually lives.

``huber``
    Huber loss in place of MSE. EDA: the target has skewness 8.6 and the top 1%
    of district-weeks carry 18.3% of all cases, so squared error is dominated by
    a handful of windows.

Reporting
---------
:func:`artifact_windows` flags evaluation windows overlapping week 395 so results
can be reported **with and without** them. This is not optional bookkeeping. On
the origin-0.85 fold, 6 of 68 test windows (9%) carry **90%** of the squared
error, and persistence RMSE there moves from 68.62 to 22.80 when they are
excluded. A pooled number that does not say which side of that split it is on is
mostly a measurement of a reporting backlog.

They are flagged, never silently dropped: removing them from evaluation flatters
every model equally and measures nothing.

Deliberately excluded
---------------------
``week_of_year``
    EDA F9 claimed bimodal seasonality and recommended this. **F9 was retracted.**
    The seasonal shape does not repeat once amplitude is normalised (r = -0.065,
    at chance), no district shows a week-of-year effect at p<0.05, and it explains
    R^2 = 0.03 against 0.86 from the previous week. EXP-012 reached the same
    conclusion first. Not implemented, on purpose.

``wider_window``
    EXP-012 tested W in {3, 8, 16, 26, 39} at 8 folds x 3 seeds and found no
    improvement, after retracting an interim claim that there was one. Not
    re-tested.

``learned_adjacency``
    Tested in EXP-018 and in the head-to-head benchmark. The effect flips sign
    between two implementations on identical folds and seeds (-0.47 in the repo's
    model, +0.45 in ours), which is what a null effect looks like. Available in
    ``adaptive.py``; not carried forward as an improvement.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

__all__ = [
    "ARTIFACT_WEEK",
    "Increments",
    "TemporalAttention",
    "artifact_windows",
    "gaussian_nll",
    "make_head",
]

#: EDA F4. The reporting artifact, as a week index into the 459-week series.
ARTIFACT_WEEK = 395


class Increments:
    """Which increments are enabled. Defaults are the plain baseline."""

    def __init__(
        self,
        per_horizon_heads: bool = False,
        temporal_attention: bool = False,
        probabilistic: bool = False,
        huber: bool = False,
    ) -> None:
        self.per_horizon_heads = per_horizon_heads
        self.temporal_attention = temporal_attention
        self.probabilistic = probabilistic
        self.huber = huber

    def label(self) -> str:
        """Short name for results tables; ``base`` when nothing is enabled."""
        on = [n for n in ("per_horizon_heads", "temporal_attention", "probabilistic",
                          "huber") if getattr(self, n)]
        return "+".join(on) if on else "base"

    def __repr__(self) -> str:
        return f"Increments({self.label()})"


class TemporalAttention(nn.Module):
    """Additive attention over the window, GulMohamed et al. Eq. (6)-(8).

    Scores each of the ``window`` positions and returns their convex
    combination, so the model can weight the most recent week differently from
    three weeks back rather than treating the window as an unordered vector.
    """

    def __init__(self, window: int, hidden: int = 16) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, nodes, window)`` -> ``(batch, nodes, window)``, reweighted."""
        weights = torch.softmax(self.score(x.unsqueeze(-1)).squeeze(-1), dim=-1)
        return x * weights * x.shape[-1]      # scale preserves magnitude


def make_head(in_features: int, horizon: int, inc: Increments, nodes: int = 0) -> nn.Module:
    """Output head: shared, per-horizon, or Gaussian.

    Args:
        in_features: Width of the representation reaching the head.
        horizon: Forecast steps to emit.
        inc: Which increments are enabled.
        nodes: ``0`` (default) when the head is applied node-wise, so the node
            axis is already present in the input. A positive value makes the head
            emit that many nodes from a graph-level vector, which is what STGAT
            needs -- its LSTM collapses the node axis before the head.

    Returns:
        A module mapping ``(..., in_features)`` to ``(..., nodes, horizon)``, with
        a trailing size-2 axis carrying ``(mu, log_var)`` when ``probabilistic``.

    ``per_horizon_heads`` gives each step its own linear map (Eq. 11) at the same
    depth as the shared head -- it splits one layer rather than adding one, so the
    arms differ in structure and not in capacity-by-depth. ``probabilistic``
    doubles the output width to carry ``(mu, log_var)`` (Eq. 12).
    """
    width = 2 if inc.probabilistic else 1
    cls = _PerHorizonHead if inc.per_horizon_heads else _SharedHead
    return cls(in_features, horizon, width, nodes)


class _HeadBase(nn.Module):
    def __init__(self, horizon: int, width: int, nodes: int) -> None:
        super().__init__()
        self.horizon, self.width, self.nodes = horizon, width, nodes

    def _shape(self, out: torch.Tensor, lead: tuple[int, ...]) -> torch.Tensor:
        """Reshape a flat head output to ``(*lead, [nodes,] horizon[, 2])``."""
        out = out.reshape(*lead, max(self.nodes, 1), self.horizon, self.width)
        if self.width == 1:
            out = out.squeeze(-1)
        if self.nodes == 0:
            out = out.squeeze(-2 if self.width == 1 else -3)
        return out


class _SharedHead(_HeadBase):
    """One layer emitting every forecast step -- the reference wiring."""

    def __init__(self, in_features: int, horizon: int, width: int, nodes: int) -> None:
        super().__init__(horizon, width, nodes)
        self.linear = nn.Linear(in_features, max(nodes, 1) * horizon * width)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self._shape(self.linear(h), h.shape[:-1])


class _PerHorizonHead(_HeadBase):
    """Eq. (11): one output layer per forecast step."""

    def __init__(self, in_features: int, horizon: int, width: int, nodes: int) -> None:
        super().__init__(horizon, width, nodes)
        self.heads = nn.ModuleList(
            nn.Linear(in_features, max(nodes, 1) * width) for _ in range(horizon)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        lead = h.shape[:-1]
        stacked = torch.stack(
            [head(h).reshape(*lead, max(self.nodes, 1), self.width) for head in self.heads],
            dim=-2,
        )
        return self._shape(stacked, lead)


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood behind Eq. (12), constant term dropped.

    Optimising this is what makes the Eq. (14) intervals calibrated rather than
    decorative. ``log_var`` is clamped because an unconstrained variance lets the
    model buy loss by declaring everything uncertain.
    """
    log_var = log_var.clamp(-10.0, 10.0)
    return torch.mean(0.5 * (log_var + (target - mu) ** 2 / torch.exp(log_var)))


def artifact_windows(window_index: np.ndarray, window: int, horizon: int) -> np.ndarray:
    """Flag windows whose input or target overlaps the week-395 artifact.

    Args:
        window_index: Time index ``i`` of each window; its target spans
            ``[i, i + horizon)`` and its input ``[i - window, i)``.
        window: Input length.
        horizon: Output length.

    Returns:
        Boolean array, ``True`` where the window touches week 395.

    Use it to report a fold twice -- all windows, and artifact-free -- rather than
    to drop anything. Under this project's rolling-origin protocol these windows
    appear only in the origin-0.85 **test** split, so they cannot be trained on
    and cannot be masked away from training.
    """
    idx = np.asarray(window_index)
    return (idx - window <= ARTIFACT_WEEK) & (idx + horizon > ARTIFACT_WEEK)
