"""Shared library code for the dengue forecasting GNN project."""

__version__ = "0.4.0"

from dengue_gnn.data import (
    FoldData,
    build_fold_tensors,
    get_persistence_forecast,
    get_rolling_origin_splits,
    load_adjacency,
    load_dataset,
)
from dengue_gnn.gan import (
    ConditionalCritic,
    ConditionalGenerator,
    compute_gradient_penalty,
    heuristic_augmentation,
    synthesize_gan_augmentation,
    train_wgan_gp,
)
from dengue_gnn.losses import (
    EpidemicDynamicsLoss,
    PhysicsInformedLoss,
    SpatialSmoothnessLoss,
    compute_normalized_laplacian,
)
from dengue_gnn.metrics import metrics, score
from dengue_gnn.models import (
    AdaptiveGCN,
    DenseGraphConv,
    GNNBaseline,
    create_adaptive_model,
    create_baseline_model,
    edge_index_to_dense_adj,
)
from dengue_gnn.runner import (
    TrainConfig,
    evaluate_fold,
    run_rolling_origin_cv,
    set_seed,
    train_fold,
)

__all__ = [
    "__version__",
    "metrics",
    "score",
    "FoldData",
    "load_adjacency",
    "load_dataset",
    "get_rolling_origin_splits",
    "build_fold_tensors",
    "get_persistence_forecast",
    "GNNBaseline",
    "DenseGraphConv",
    "AdaptiveGCN",
    "create_baseline_model",
    "create_adaptive_model",
    "edge_index_to_dense_adj",
    "compute_normalized_laplacian",
    "SpatialSmoothnessLoss",
    "EpidemicDynamicsLoss",
    "PhysicsInformedLoss",
    "ConditionalGenerator",
    "ConditionalCritic",
    "compute_gradient_penalty",
    "train_wgan_gp",
    "heuristic_augmentation",
    "synthesize_gan_augmentation",
    "TrainConfig",
    "set_seed",
    "train_fold",
    "evaluate_fold",
    "run_rolling_origin_cv",
]
