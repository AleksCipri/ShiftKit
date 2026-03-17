from .base   import BaseTrainer, TrainerRegistry
from .mmd    import MMDLoss, MMDTrainer, SourceOnlyTrainer
from .dann   import DANNTrainer, GradientReversalLayer, DomainDiscriminator
from .sidda  import SIDDATrainer
from .lmmd   import LMMDLoss, LMMDTrainer

# ─── register built-in trainers ───────────────────────────────────────────────
TrainerRegistry.register("source_only",  SourceOnlyTrainer)
TrainerRegistry.register("mmd",          MMDTrainer)
TrainerRegistry.register("lmmd",         LMMDTrainer)
TrainerRegistry.register("dann",         DANNTrainer)
TrainerRegistry.register("sidda",        SIDDATrainer)

__all__ = [
    "BaseTrainer", "TrainerRegistry",
    "MMDLoss", "MMDTrainer", "SourceOnlyTrainer",
    "LMMDLoss", "LMMDTrainer",
    "DANNTrainer", "GradientReversalLayer", "DomainDiscriminator",
    "SIDDATrainer",
]
