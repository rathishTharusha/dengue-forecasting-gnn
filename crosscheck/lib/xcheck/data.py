"""Loading and windowing the Sri Lanka dengue array, two ways on purpose.

The Weng et al. reference implementation
(``reference_repo/Models/evaluation.py``) makes three data-handling choices that
this workspace reproduces *and* corrects, so the difference between them can be
measured rather than argued about:

===========================  ====================================  ==========================
Choice                        Reference (``normalize="global"``)    Corrected (``"train"``)
===========================  ====================================  ==========================
z-score statistics            whole array, test weeks included      training weeks only
Target channel                ``x[..., -6]``                        same (index 5 of 11)
Feature set                   case channel only by default          configurable
===========================  ====================================  ==========================

The global-statistics path leaks test-period mean and variance into training.
It is kept because it is what produced the published numbers.

Everything here is re-derived from the reference implementation and the paper;
nothing is imported from ``src/dengue_gnn``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "REFERENCE_CASES_IDX",
    "Windows",
    "build_edge_index",
    "load_adjacency",
    "load_array",
    "make_windows",
    "segment_split",
]

#: Target channel. The reference indexes it as ``x[..., -6]``; with 11 features
#: that is index 5, which is what this project calls ``cases_idx``.
REFERENCE_CASES_IDX = 5


@dataclass(frozen=True)
class Windows:
    """A windowed dataset in normalized space, plus the inverse transform.

    Attributes:
        x: ``(n_windows, n_nodes, n_features, window)`` inputs.
        y: ``(n_windows, n_nodes, horizon)`` targets, normalized.
        mean: Location used by the z-score.
        std: Scale used by the z-score.
        index: Time index ``i`` of each window, where ``y`` covers ``[i, i+horizon)``.
    """

    x: np.ndarray
    y: np.ndarray
    mean: float
    std: float
    index: np.ndarray

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Undo the z-score, returning raw case counts."""
        return np.asarray(values, dtype=np.float64) * self.std + self.mean

    def __len__(self) -> int:
        return int(self.x.shape[0])


def load_array(npy_path: str | Path) -> np.ndarray:
    """Load the processed array as ``(T, N, F)`` float64, NaNs zeroed.

    ``np.nan_to_num`` matches the reference implementation's ``load_data``.
    """
    raw = np.load(Path(npy_path), allow_pickle=True)
    return np.nan_to_num(np.asarray(raw, dtype=np.float64))


def load_adjacency(adj_path: str | Path, self_loops: bool = True) -> tuple[np.ndarray, list[str]]:
    """Build the dense adjacency matrix and the district ordering.

    Districts are ordered by ``sorted()`` of the JSON keys -- the reference
    implementation's ``idx_map``, and the ordering every downstream index
    assumes.

    Returns:
        ``(A, names)`` with ``A`` of shape ``(n, n)`` in ``{0.0, 1.0}``.
    """
    adj = json.loads(Path(adj_path).read_text(encoding="utf-8"))
    names = sorted(adj)
    idx = {name: i for i, name in enumerate(names)}
    n = len(names)

    a = np.zeros((n, n), dtype=np.float64)
    if self_loops:
        np.fill_diagonal(a, 1.0)
    for district, neighbours in adj.items():
        for neighbour in neighbours:
            a[idx[district], idx[neighbour]] = 1.0
    return a, names


def build_edge_index(adjacency: np.ndarray) -> np.ndarray:
    """Convert a dense adjacency matrix to a ``(2, n_edges)`` COO edge index."""
    src, dst = np.nonzero(adjacency)
    return np.stack([src, dst]).astype(np.int64)


def make_windows(
    raw: np.ndarray,
    window: int = 3,
    horizon: int = 3,
    cases_idx: int = REFERENCE_CASES_IDX,
    use_all_features: bool = False,
    normalize: str = "global",
    train_end: int | None = None,
) -> Windows:
    """Slice ``raw`` into ``(window -> horizon)`` samples with a z-scored target.

    Args:
        raw: ``(T, N, F)`` array from :func:`load_array`.
        window: Number of past weeks per input.
        horizon: Number of future weeks predicted.
        cases_idx: Index of the case-count channel.
        use_all_features: Keep all ``F`` channels (``True``) or the case channel
            alone (``False``). The reference runs its GNNs with
            ``use_disease_only=True``, i.e. ``False`` here.
        normalize: ``"global"`` reproduces the reference -- statistics over the
            entire array, test weeks included. ``"train"`` computes them from
            ``raw[:train_end]`` only, which is what a forecasting protocol
            requires.
        train_end: Required when ``normalize="train"``; the first time index
            **excluded** from the statistics.

    Returns:
        A :class:`Windows` bundle.

    Raises:
        ValueError: On an unknown ``normalize`` mode, a missing ``train_end``,
            or a ``raw`` too short to yield a single window.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 3:
        raise ValueError(f"expected (T, N, F), got shape {raw.shape}")
    t_len = raw.shape[0]
    if t_len - horizon <= window:
        raise ValueError(
            f"series of length {t_len} too short for window={window}, horizon={horizon}"
        )

    if normalize == "global":
        stat_source = raw
    elif normalize == "train":
        if train_end is None:
            raise ValueError('normalize="train" requires train_end')
        if train_end <= window:
            raise ValueError(f"train_end={train_end} leaves no training weeks")
        stat_source = raw[:train_end]
    else:
        raise ValueError(f"unknown normalize mode {normalize!r}")

    # The reference z-scores the whole tensor by the case channel's statistics
    # when running disease-only, and by the whole tensor's otherwise.
    if use_all_features:
        mean = float(stat_source.mean())
        std = float(stat_source.std())
    else:
        mean = float(stat_source[..., cases_idx].mean())
        std = float(stat_source[..., cases_idx].std())
    if std == 0:
        raise ValueError("zero standard deviation; cannot z-score")

    z = (raw - mean) / std
    cases = z[..., cases_idx]

    xs, ys, idx = [], [], []
    for i in range(window, t_len - horizon):
        if use_all_features:
            # (window, N, F) -> (N, F, window)
            xs.append(np.transpose(z[i - window : i], (1, 2, 0)))
        else:
            # (window, N) -> (N, 1, window)
            xs.append(cases[i - window : i].T[:, None, :])
        ys.append(cases[i : i + horizon].T)
        idx.append(i)

    return Windows(
        x=np.stack(xs),
        y=np.stack(ys),
        mean=mean,
        std=std,
        index=np.asarray(idx, dtype=np.int64),
    )


def segment_split(
    n: int,
    train_frac: float = 0.7,
    val_frac: float = 0.0,
) -> tuple[slice, slice, slice]:
    """Chronological train / validation / test slices over ``n`` samples.

    Reproduces the reference's index arithmetic: ``t_idx = int(train * n)`` and
    ``v_idx = int((train + val) * n)``, with the remainder as test. With
    ``val_frac=0`` this is the paper's plain 70/30 split.
    """
    if not 0 < train_frac <= 1:
        raise ValueError("train_frac must lie in (0, 1]")
    if not 0 <= val_frac < 1:
        raise ValueError("val_frac must lie in [0, 1)")
    t_idx = int(train_frac * n)
    v_idx = int((train_frac + val_frac) * n)
    return slice(0, t_idx), slice(t_idx, v_idx), slice(v_idx, n)
