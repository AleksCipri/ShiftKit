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
from .models.networks    import MLP, CNN
from .methods.mmd        import MMDLoss, MMDTrainer
from .diagnostics.plots  import plot_latent_space, plot_training_history

__version__ = "0.1.0"
__all__ = [
    "DataManager",
    "MLP", "CNN",
    "MMDLoss", "MMDTrainer",
    "plot_latent_space", "plot_training_history",
]
