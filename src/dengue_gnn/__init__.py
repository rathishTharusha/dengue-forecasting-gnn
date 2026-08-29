"""Shared library code for the dengue forecasting GNN project.

Code starts life in ``notebooks/`` and moves here once it stabilizes and is used
in more than one place, so that every phase of the project scores its results
with exactly the same implementation.

See CONTRIBUTING.md for the notebook -> src promotion rule.

Torch-dependent modules (``models``, ``losses``) are not imported here, so that
``dengue_gnn.metrics`` stays importable in environments without torch -- CI
installs numpy only.
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
