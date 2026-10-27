from .networks import MLP, CNN, GalaxyCNN, MLPRegressor, mean_var_from_head, VAR_EPS
from .gnn      import SimpleGCN

try:
    from .gnn_pyg import GNN
except ImportError:
    GNN = None  # torch-geometric not installed

__all__ = ["MLP", "CNN", "GalaxyCNN", "MLPRegressor", "SimpleGCN", "mean_var_from_head", "VAR_EPS"]
if GNN is not None:
    __all__.append("GNN")
