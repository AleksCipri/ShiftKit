"""
Base class and registry for domain adaptation trainers.

BaseTrainer
-----------
Abstract base class that defines the interface all trainers must implement.
Subclass it to build a custom DA method that is drop-in compatible with
ShiftKit's diagnostics and history utilities.

TrainerRegistry
---------------
Lightweight registry that maps string keys to trainer classes.  Built-in
trainers are registered automatically at import time.  Register your own:

    from shiftkit.methods import TrainerRegistry

    @TrainerRegistry.register("my_method")
    class MyTrainer(BaseTrainer):
        ...
"""

from abc import ABC, abstractmethod
from typing import List


# ─── Abstract base class ──────────────────────────────────────────────────────

class BaseTrainer(ABC):
    """
    Abstract base class for ShiftKit domain adaptation trainers.

    All built-in trainers (MMDTrainer, DANNTrainer, SIDDATrainer, …) follow
    this interface.  Subclass BaseTrainer to create a custom DA method that
    works seamlessly with ShiftKit's history utilities and diagnostics.

    Required
    --------
    Subclasses must implement:
      - ``fit(epochs)``    — run training, return per-epoch history list
      - ``evaluate(loader, domain)`` — compute accuracy on a DataLoader

    History format
    --------------
    ``fit()`` should return a ``list[dict]``, one dict per epoch.  At minimum
    include the keys used by ``plot_training_history``:

        {
            "epoch":      int,
            "ce_loss":    float,
            "total_loss": float,
            "src_acc":    float,
            "tgt_acc":    float,
        }

    Add method-specific keys freely (e.g. ``"mmd_loss"``, ``"da_loss"``).
    """

    @abstractmethod
    def fit(self, epochs: int = 10) -> List[dict]:
        """Train the model and return a per-epoch history list."""

    @abstractmethod
    def evaluate(self, loader, domain: str = "source") -> dict:
        """
        Evaluate on a labelled DataLoader.

        Returns
        -------
        dict with keys ``domain`` (str), ``accuracy`` (float), ``n_samples`` (int).
        """


# ─── Registry ─────────────────────────────────────────────────────────────────

class TrainerRegistry:
    """
    Registry that maps string keys to trainer classes.

    Built-in trainers are registered automatically.  Register a custom
    trainer with the ``register`` decorator or method:

    Examples
    --------
    Decorator style (recommended)::

        from shiftkit.methods import TrainerRegistry, BaseTrainer

        @TrainerRegistry.register("my_method")
        class MyTrainer(BaseTrainer):
            def __init__(self, model, source_loader, target_loader, **kwargs):
                ...
            def fit(self, epochs=10):
                ...
            def evaluate(self, loader, domain="source"):
                ...

    Explicit registration::

        TrainerRegistry.register("my_method", MyTrainer)

    Instantiating from the registry::

        trainer = TrainerRegistry.create(
            "my_method",
            model=model,
            source_loader=train_src,
            target_loader=train_tgt,
        )
        history = trainer.fit(epochs=20)
    """

    _REGISTRY: dict = {}

    @staticmethod
    def register(name: str, cls=None):
        """
        Register a trainer class under *name*.

        Can be used as a plain call or as a class decorator::

            # decorator
            @TrainerRegistry.register("my_method")
            class MyTrainer(BaseTrainer): ...

            # explicit
            TrainerRegistry.register("my_method", MyTrainer)
        """
        if cls is None:
            # called as a decorator factory: @TrainerRegistry.register("name")
            def decorator(klass):
                TrainerRegistry._REGISTRY[name] = klass
                return klass
            return decorator
        # called explicitly: TrainerRegistry.register("name", MyClass)
        TrainerRegistry._REGISTRY[name] = cls
        return cls

    @staticmethod
    def available() -> List[str]:
        """Return a sorted list of all registered trainer names."""
        return sorted(TrainerRegistry._REGISTRY.keys())

    @staticmethod
    def get(name: str):
        """
        Return the trainer *class* registered under *name*.

        Raises
        ------
        KeyError if the name is not registered.
        """
        if name not in TrainerRegistry._REGISTRY:
            available = TrainerRegistry.available()
            raise KeyError(
                f"Unknown trainer '{name}'. "
                f"Available: {available}. "
                f"Register a custom trainer with TrainerRegistry.register()."
            )
        return TrainerRegistry._REGISTRY[name]

    @staticmethod
    def create(name: str, **kwargs):
        """
        Instantiate the trainer registered under *name*, forwarding **kwargs**.

        Example::

            trainer = TrainerRegistry.create(
                "mmd",
                model=model,
                source_loader=train_src,
                target_loader=train_tgt,
                mmd_weight=0.5,
            )
        """
        return TrainerRegistry.get(name)(**kwargs)
