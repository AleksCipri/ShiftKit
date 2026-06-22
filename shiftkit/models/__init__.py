from .networks import MLP, CNN, MLPRegressor
from .gnn      import SimpleGCN

try:
    from .gnn_pyg import GNN
except ImportError:
    GNN = None  # torch-geometric not installed

__all__ = ["MLP", "CNN", "MLPRegressor", "SimpleGCN"]
if GNN is not None:
    __all__.append("GNN")
