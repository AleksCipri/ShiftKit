from .base       import BaseTrainer, TrainerRegistry
from .mmd        import MMDLoss, MMDTrainer, SourceOnlyTrainer
from .dann       import DANNTrainer, GradientReversalLayer, DomainDiscriminator
from .sidda      import SIDDATrainer
from .lmmd       import LMMDLoss, LMMDTrainer
from .coral      import CORALLoss, CORALTrainer
from .regression import SourceOnlyRegressionTrainer, MMDRegressionTrainer
from .kliep      import KLIEPWeightEstimator, KLIEPTrainer

# ─── register built-in trainers ───────────────────────────────────────────────
TrainerRegistry.register("source_only",  SourceOnlyTrainer)
TrainerRegistry.register("mmd",          MMDTrainer)
TrainerRegistry.register("lmmd",         LMMDTrainer)
TrainerRegistry.register("coral",        CORALTrainer)
TrainerRegistry.register("dann",         DANNTrainer)
TrainerRegistry.register("sidda",        SIDDATrainer)
TrainerRegistry.register("kliep",        KLIEPTrainer)

__all__ = [
    "BaseTrainer", "TrainerRegistry",
    "MMDLoss", "MMDTrainer", "SourceOnlyTrainer",
    "LMMDLoss", "LMMDTrainer",
    "CORALLoss", "CORALTrainer",
    "DANNTrainer", "GradientReversalLayer", "DomainDiscriminator",
    "SIDDATrainer",
    "SourceOnlyRegressionTrainer", "MMDRegressionTrainer",
    "KLIEPWeightEstimator", "KLIEPTrainer",
]
