"""
ShiftKit — a lightweight domain adaptation framework.

Quick start
-----------
from shiftkit.data        import DataManager
from shiftkit.models      import CNN, MLP
from shiftkit.methods     import MMDTrainer
from shiftkit.diagnostics import plot_latent_space, plot_training_history
"""

from .data.datasets      import DataManager
from .models.networks    import MLP, CNN, MLPRegressor
from .models.gnn         import SimpleGCN
try:
    from .models.gnn_pyg import GNN
except ImportError:
    GNN = None  # torch-geometric not installed
from .methods.base       import BaseTrainer, TrainerRegistry
from .methods.mmd        import MMDLoss, MMDTrainer, SourceOnlyTrainer
from .methods.lmmd       import LMMDLoss, LMMDTrainer
from .methods.coral      import CORALLoss, CORALTrainer
from .methods.dann       import DANNTrainer, GradientReversalLayer, DomainDiscriminator
from .methods.sidda      import SIDDATrainer, SIDDARegressionTrainer
from .methods.regression import SourceOnlyRegressionTrainer, MMDRegressionTrainer
from .methods.kliep      import KLIEPWeightEstimator, KLIEPTrainer
from .methods.kmm        import KMMWeightEstimator, KMMTrainer
from .data.datasets      import SineWaveDataset, CaliforniaHousingDataset
from .diagnostics.plots  import (
    plot_latent_space, plot_latent_space_domains, plot_training_history,
    compare_latent_spaces, plot_confusion_matrix, plot_roc_curve,
)

__version__ = "0.1.0"
__all__ = [
    "DataManager",
    "MLP", "CNN", "MLPRegressor", "SimpleGCN",
    "SineWaveDataset", "CaliforniaHousingDataset",
    "BaseTrainer", "TrainerRegistry",
    "MMDLoss", "MMDTrainer", "SourceOnlyTrainer",
    "LMMDLoss", "LMMDTrainer",
    "CORALLoss", "CORALTrainer",
    "DANNTrainer", "GradientReversalLayer", "DomainDiscriminator",
    "SIDDATrainer", "SIDDARegressionTrainer",
    "SourceOnlyRegressionTrainer", "MMDRegressionTrainer",
    "KLIEPWeightEstimator", "KLIEPTrainer",
    "KMMWeightEstimator", "KMMTrainer",
    "plot_latent_space", "plot_latent_space_domains", "plot_training_history",
    "compare_latent_spaces",
    "plot_confusion_matrix", "plot_roc_curve",
]
if GNN is not None:
    __all__.append("GNN")
