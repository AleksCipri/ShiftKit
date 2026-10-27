from .datasets import (
    DataManager, SineWaveDataset, CaliforniaHousingDataset,
    california_normalisation_stats,
)
# Dependency-free container: available with or without PyTorch Geometric.
from .node_batch import NodeGraphBatch

try:
    from .pyg_utils import ensure_masks, build_pyg_domain_loaders
except ImportError:
    ensure_masks = None
    build_pyg_domain_loaders = None

__all__ = ["DataManager", "SineWaveDataset", "CaliforniaHousingDataset",
           "california_normalisation_stats", "NodeGraphBatch"]
if ensure_masks is not None:
    __all__ += ["ensure_masks", "build_pyg_domain_loaders"]
