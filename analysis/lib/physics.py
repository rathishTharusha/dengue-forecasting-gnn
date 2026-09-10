"""Physics-informed arms over the five verified architectures.

Three arms, one backbone
------------------------
``base``
    Unchanged. Predicts a residual over persistence in normalised log1p space.
``penalty``
    ``base`` plus ``lambda * renewal_penalty``. The architecture is untouched; the
    objective gains a term charging forecasts that no admissible ``R_t``
    trajectory could produce. This is the ``+physics`` ablation row
    ``docs/ROADMAP.md`` asks for.
``decoder``
    The backbone's output is reinterpreted as ``log R_t`` and cases are
    *reconstructed* through the renewal equation. Structurally unable to produce
    a flat forecast.

Every arm shares the backbone, the folds, the normalisation, the loss scale and
the protocol, so a difference between them is attributable to the physics.

Why the decoder is expected to do the work
------------------------------------------
``analysis/results/error_diagnosis.json`` measured the failure the baselines
have: predicted ``|weekly log growth|`` maxes out at 0.336 against an observed
3.638, and outbreak windows are 12.6% of the data but 61.7% of the squared error,
forecast with a -12.9 case bias. A penalty nudges a model that can already
express the right answer; these models cannot. The decoder changes what is
expressible, which is why it is the arm aimed at the 30% of RMSE sitting in
outbreak weeks.

Everything is scored in raw counts by the same ``adaptive.pooled_scores`` as every
other row, with and without the week-395 artifact windows.
"""

from __future__ import annotations

import numpy as np
import renewal
import torch
from torch import nn

__all__ = ["ARMS", "PhysicsNet", "window_history"]

#: Arm name -> keyword arguments for :class:`PhysicsNet`.
ARMS: dict[str, dict] = {
    "base": {"mode": "base"},
    "penalty": {"mode": "penalty"},
    "decoder": {"mode": "decoder"},
}


def window_history(cases: np.ndarray, index: np.ndarray,
                   weeks: int = renewal.HISTORY_WEEKS) -> torch.Tensor:
    """Observed raw counts immediately before each forecast window.

    Args:
        cases: ``(weeks, districts)`` raw counts for the whole series.
        index: Window start times ``i``; the forecast target is ``[i, i+horizon)``
            so the history is ``[i-weeks, i)``.
        weeks: How many weeks of history to take.

    Returns:
        ``(windows, districts, weeks)`` float32, most recent last.

    Windows near the start of the record have less than ``weeks`` of history --
    the protocol's first window begins at t=3 and ``HISTORY_WEEKS`` is 9, so six
    windows in the whole dataset are short, all in the earliest training fold.
    They are **edge-padded** with the first available week rather than zero-padded:
    zeros would understate infectious pressure and make the physics term push the
    forecast down exactly where the record is thinnest, whereas edge padding
    assumes the series was flat before the record began. Dropping them instead
    would give the physics arms a different training set from ``base`` and break
    the controlled comparison.
    """
    idx = np.asarray(index)
    rows = []
    for i in idx:
        past = cases[max(0, i - weeks) : i]
        if len(past) < weeks:
            past = np.concatenate([np.repeat(past[:1], weeks - len(past), axis=0), past])
        rows.append(past.T)
    return torch.tensor(np.stack(rows), dtype=torch.float32)


class PhysicsNet(nn.Module):
    """One architecture under one physics arm.

    Args:
        backbone: Any module mapping ``(batch, nodes, window)`` and an edge index
            to ``(batch, nodes, horizon)`` -- i.e. anything from
            :mod:`reproduced`.
        edge_index: Graph, passed through to the backbone on every call.
        w: Generation-interval weights from :func:`renewal.generation_interval`.
        mode: One of :data:`ARMS`.
        mean: Training-fold mean of ``log1p(cases)``.
        std: Training-fold standard deviation of ``log1p(cases)``.

    The fold's normalisation constants are held here because the decoder produces
    raw counts while the loss is taken in normalised log1p space, so something has
    to convert between them -- and doing it inside the module keeps the training
    loop identical across arms.
    """

    def __init__(self, backbone: nn.Module, edge_index: torch.Tensor, w: torch.Tensor,
                 mode: str, mean: float, std: float) -> None:
        super().__init__()
        if mode not in ARMS:
            raise ValueError(f"mode must be one of {tuple(ARMS)}, got {mode!r}")
        self.backbone = backbone
        self.mode = mode
        self.mean, self.std = mean, std
        self.register_buffer("edge_index", edge_index)
        self.decoder = renewal.RenewalDecoder(w) if mode == "decoder" else None
        if mode == "penalty":
            self.register_buffer("w", w.to(torch.float32))

    def to_counts(self, z: torch.Tensor) -> torch.Tensor:
        """Normalised log1p space -> raw counts, differentiably.

        The clamp mirrors ``adaptive.Fold.inverse``; without it ``expm1`` of a
        large activation overflows to inf early in training and takes the whole
        run with it.
        """
        return torch.expm1((z * self.std + self.mean).clamp(0.0, 12.0))

    def to_z(self, counts: torch.Tensor) -> torch.Tensor:
        """Raw counts -> normalised log1p space, the scale the loss lives on."""
        return (torch.log1p(counts.clamp_min(0.0)) - self.mean) / self.std

    def forward(self, x: torch.Tensor, persistence: torch.Tensor,
                history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(z_prediction, counts_prediction)``.

        Both are returned because the loss is taken in ``z`` and the physics term
        and the metrics are taken in counts, and recomputing either from the other
        would round-trip through a clamp twice.
        """
        raw = self.backbone(x, self.edge_index)
        if self.mode == "decoder":
            counts = self.decoder(raw, history[..., -renewal.KERNEL_WEEKS:])
            return self.to_z(counts), counts
        z = raw + persistence          # residual over persistence, as every other row
        return z, self.to_counts(z)

    def physics_loss(self, counts: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """The ``penalty`` arm's extra term; exactly zero for the other two."""
        if self.mode != "penalty":
            return counts.sum() * 0.0    # keeps the graph connected, contributes nothing
        return renewal.renewal_penalty(counts, history, self.w)
