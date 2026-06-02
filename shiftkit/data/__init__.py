from .datasets import DataManager, SineWaveDataset, CaliforniaHousingDataset

try:
    from .pyg_utils import NodeGraphBatch, ensure_masks, build_pyg_domain_loaders
except ImportError:
    NodeGraphBatch = None
    ensure_masks = None
    build_pyg_domain_loaders = None

__all__ = ["DataManager", "SineWaveDataset", "CaliforniaHousingDataset"]
if NodeGraphBatch is not None:
    __all__ += ["NodeGraphBatch", "ensure_masks", "build_pyg_domain_loaders"]
