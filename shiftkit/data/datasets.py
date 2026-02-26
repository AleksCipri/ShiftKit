"""
Data loading utilities for domain adaptation experiments.

Provides DataManager, which returns paired (source, target) DataLoaders.
New datasets can be registered via DataManager.register().
"""

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import numpy as np

# ─── built-in dataset pairs ───────────────────────────────────────────────────

class NoisyMNIST(Dataset):
    """MNIST with additive Gaussian noise — used as a synthetic target domain."""

    def __init__(self, root: str, train: bool = True, noise_std: float = 0.3,
                 transform=None, download: bool = True):
        self.base = torchvision.datasets.MNIST(
            root=root, train=train, download=download,
            transform=T.ToTensor()
        )
        self.noise_std = noise_std
        self.extra_transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        noise = torch.randn_like(img) * self.noise_std
        img = (img + noise).clamp(0.0, 1.0)
        if self.extra_transform is not None:
            img = self.extra_transform(img)
        return img, label


# ─── DataManager ─────────────────────────────────────────────────────────────

_REGISTRY: dict = {}


def _register_defaults():
    """Register built-in dataset pairs at import time."""

    def _mnist_noisy_mnist(root, batch_size, train, num_workers, **kw):
        noise_std = kw.get("noise_std", 0.3)
        base_tf = T.Normalize((0.1307,), (0.3081,))

        source_ds = torchvision.datasets.MNIST(
            root=root, train=train, download=True,
            transform=T.Compose([T.ToTensor(), base_tf])
        )
        target_ds = NoisyMNIST(
            root=root, train=train, noise_std=noise_std,
            transform=base_tf, download=True
        )
        source_loader = DataLoader(
            source_ds, batch_size=batch_size, shuffle=train,
            num_workers=num_workers, pin_memory=True
        )
        target_loader = DataLoader(
            target_ds, batch_size=batch_size, shuffle=train,
            num_workers=num_workers, pin_memory=True
        )
        return source_loader, target_loader

    _REGISTRY["mnist_noisy_mnist"] = _mnist_noisy_mnist


_register_defaults()


class DataManager:
    """
    Central hub for loading source/target domain data.

    Usage
    -----
    dm = DataManager(root="./data", batch_size=64)
    train_src, train_tgt = dm.load("mnist_noisy_mnist")
    test_src,  test_tgt  = dm.load("mnist_noisy_mnist", train=False)

    Custom datasets
    ---------------
    DataManager.register("my_pair", my_factory_fn)
    # factory signature: (root, batch_size, train, num_workers, **kwargs)
    #                    -> (source_loader, target_loader)
    """

    def __init__(
        self,
        root: str = "./data",
        batch_size: int = 64,
        num_workers: int = 2,
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    # ------------------------------------------------------------------
    def load(
        self,
        name: str,
        train: bool = True,
        **kwargs,
    ):
        """
        Return (source_loader, target_loader) for the registered dataset pair.

        Parameters
        ----------
        name    : registered key, e.g. "mnist_noisy_mnist"
        train   : whether to load the training split
        **kwargs: forwarded to the factory (e.g. noise_std=0.5)
        """
        if name not in _REGISTRY:
            raise ValueError(
                f"Unknown dataset '{name}'. "
                f"Available: {list(_REGISTRY.keys())}"
            )
        factory = _REGISTRY[name]
        return factory(
            root=self.root,
            batch_size=self.batch_size,
            train=train,
            num_workers=self.num_workers,
            **kwargs,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def register(name: str, factory):
        """Register a custom dataset-pair factory under *name*."""
        _REGISTRY[name] = factory

    @staticmethod
    def available() -> list:
        """List all registered dataset-pair names."""
        return list(_REGISTRY.keys())
