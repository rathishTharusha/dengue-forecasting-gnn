"""Shared library code for the dengue forecasting GNN project.

Code starts life in ``notebooks/`` and moves here once it stabilizes and is used
in more than one place, so that every phase of the project scores its results
with exactly the same implementation.

See CONTRIBUTING.md for the notebook -> src promotion rule.

Everything here is torch-free, so it stays importable in environments without
torch -- CI installs numpy, pytest and ruff only. The modelling code lives in
``analysis/lib`` and does pull in torch.

``metrics`` is the shared scoring implementation; ``seir`` is the SEIR-SEI model
validated against Phaijoo & Gurung (2018); ``data`` is a minimal case-series
loader. The Phase-2/3 modelling modules were removed once the work moved to
verified architectures -- see tag ``phase23-archive``.
"""

__version__ = "0.2.0"

from dengue_gnn.metrics import (
    metrics,
    peak_week_error,
    peak_week_error_by_horizon,
    score,
    smape,
)

__all__ = [
    "__version__",
    "metrics",
    "peak_week_error",
    "peak_week_error_by_horizon",
    "score",
    "smape",
]
