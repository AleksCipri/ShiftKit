from .base       import BaseTrainer, TrainerRegistry
from .mmd        import MMDLoss, MMDTrainer, SourceOnlyTrainer
from .dann       import DANNTrainer, GradientReversalLayer, DomainDiscriminator
from .sidda      import SIDDATrainer, SIDDARegressionTrainer
from .lmmd       import LMMDLoss, LMMDTrainer
from .coral      import CORALLoss, CORALTrainer
from .regression import SourceOnlyRegressionTrainer, MMDRegressionTrainer
from .kliep      import KLIEPWeightEstimator, KLIEPTrainer
from .kmm        import KMMWeightEstimator, KMMTrainer
from .node_batch import unpack_batch, latents_and_targets, node_latent_vectors

# ─── register built-in trainers ───────────────────────────────────────────────
TrainerRegistry.register("source_only",  SourceOnlyTrainer)
TrainerRegistry.register("mmd",          MMDTrainer)
TrainerRegistry.register("lmmd",         LMMDTrainer)
TrainerRegistry.register("coral",        CORALTrainer)
TrainerRegistry.register("dann",         DANNTrainer)
TrainerRegistry.register("sidda",        SIDDATrainer)
TrainerRegistry.register("kliep",        KLIEPTrainer)
TrainerRegistry.register("kmm",          KMMTrainer)

# SourceOnly/MMD/LMMD/CORAL/DANN subclass BaseTrainer directly.  SIDDA, KLIEP
# and KMM implement the same fit/evaluate interface; registering them as
# virtual subclasses makes isinstance/issubclass(…, BaseTrainer) hold for all
# eight registered trainers (a no-op for classes that already subclass it).
for _cls in (SIDDATrainer, KLIEPTrainer, KMMTrainer):
    BaseTrainer.register(_cls)
del _cls

__all__ = [
    "BaseTrainer", "TrainerRegistry",
    "MMDLoss", "MMDTrainer", "SourceOnlyTrainer",
    "LMMDLoss", "LMMDTrainer",
    "CORALLoss", "CORALTrainer",
    "DANNTrainer", "GradientReversalLayer", "DomainDiscriminator",
    "SIDDATrainer", "SIDDARegressionTrainer",
    "SourceOnlyRegressionTrainer", "MMDRegressionTrainer",
    "KLIEPWeightEstimator", "KLIEPTrainer",
    "KMMWeightEstimator", "KMMTrainer",
    "unpack_batch", "latents_and_targets", "node_latent_vectors",
]
