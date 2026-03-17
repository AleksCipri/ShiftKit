from .base   import BaseTrainer, TrainerRegistry
from .mmd    import MMDLoss, MMDTrainer, SourceOnlyTrainer
from .dann   import DANNTrainer, GradientReversalLayer, DomainDiscriminator
from .sidda  import SIDDATrainer

# ─── register built-in trainers ───────────────────────────────────────────────
TrainerRegistry.register("source_only",  SourceOnlyTrainer)
TrainerRegistry.register("mmd",          MMDTrainer)
TrainerRegistry.register("dann",         DANNTrainer)
TrainerRegistry.register("sidda",        SIDDATrainer)

__all__ = [
    "BaseTrainer", "TrainerRegistry",
    "MMDLoss", "MMDTrainer", "SourceOnlyTrainer",
    "DANNTrainer", "GradientReversalLayer", "DomainDiscriminator",
    "SIDDATrainer",
]
