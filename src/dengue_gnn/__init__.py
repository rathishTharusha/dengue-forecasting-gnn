"""Shared library code for the dengue forecasting GNN project.

Code starts life in ``notebooks/`` and moves here once it stabilizes and is used
in more than one place, so that every phase of the project scores its results
with exactly the same implementation.

See CONTRIBUTING.md for the notebook -> src promotion rule.
"""

__version__ = "0.1.0"

from dengue_gnn.metrics import metrics, score

__all__ = ["__version__", "metrics", "score"]
