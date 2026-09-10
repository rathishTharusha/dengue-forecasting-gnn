"""Dynamic graph construction for the DengueGNN reproduction (R2).

Equations (2)-(4) of GulMohamed et al.: a hybrid adjacency blending static
geographic structure with a gravity-model mobility prior, symmetrically
normalized.

Kept in its own module, free of ``torch``, so the graph construction can be
tested and reasoned about without a deep-learning stack -- the same discipline
the main project applies to ``src/dengue_gnn`` (see ``CLAUDE.md``: CI installs
no torch).
"""

from __future__ import annotations

import numpy as np

__all__ = ["gravity_mobility", "hop_distance", "hybrid_adjacency", "symmetric_normalize"]


def gravity_mobility(population: np.ndarray, distance: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Gravity-model mobility matrix, the paper's Eq. (3).

    ``G_ij = p_i * p_j / d_ij^2``, zero on the diagonal, then scaled so the
    largest entry is 1 -- which makes it commensurate with the binary geographic
    adjacency it is blended with in :func:`hybrid_adjacency`.

    Args:
        population: Length-``n`` populations. Uniform values reduce this to a
            pure distance-decay kernel.
        distance: ``(n, n)`` pairwise distances. The diagonal is ignored.
        eps: Guard against division by zero for coincident points.

    Returns:
        ``(n, n)`` non-negative matrix with a zero diagonal.

    Raises:
        ValueError: If ``distance`` is not square of side ``len(population)``.
    """
    p = np.asarray(population, dtype=np.float64).ravel()
    d = np.asarray(distance, dtype=np.float64)
    n = p.size
    if d.shape != (n, n):
        raise ValueError(f"distance must be ({n}, {n}), got {d.shape}")

    g = np.outer(p, p) / (d**2 + eps)
    np.fill_diagonal(g, 0.0)
    peak = g.max()
    return g / peak if peak > 0 else g


def hop_distance(adjacency: np.ndarray) -> np.ndarray:
    """Shortest-path hop counts, used as a topological stand-in for distance.

    Disconnected pairs (Sri Lanka's district graph has none, but islands are a
    general hazard) are placed one hop beyond the observed maximum rather than
    left at infinity, so the gravity kernel stays finite.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    hops = shortest_path(csr_matrix(np.asarray(adjacency)), directed=False, unweighted=True)
    finite = np.isfinite(hops)
    if not finite.all():
        hops[~finite] = hops[finite].max() + 1
    return hops


def hybrid_adjacency(a_geo: np.ndarray, a_mob: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend geographic and mobility adjacency, the paper's Eq. (2).

    ``alpha * A_geo + (1 - alpha) * A_mob``. ``alpha = 1`` recovers the
    static-graph ablation of the paper's Table 6.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    a_geo = np.asarray(a_geo, dtype=np.float64)
    a_mob = np.asarray(a_mob, dtype=np.float64)
    if a_geo.shape != a_mob.shape:
        raise ValueError("a_geo and a_mob must share a shape")
    return alpha * a_geo + (1.0 - alpha) * a_mob


def symmetric_normalize(a: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """``D^-1/2 A D^-1/2``, the paper's Eq. (4). Isolated nodes are left at zero."""
    a = np.asarray(a, dtype=np.float64).copy()
    if add_self_loops:
        a = a + np.eye(a.shape[0])
    deg = a.sum(axis=1)
    inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
    d = np.diag(inv_sqrt)
    return d @ a @ d
