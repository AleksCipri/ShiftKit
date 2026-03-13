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


# ─── Synthetic Graph Dataset ─────────────────────────────────────────────────

class SyntheticGraphDataset(Dataset):
    """
    Synthetic benchmark for graph-level domain adaptation (binary classification).

    Two structurally distinct graph classes:
      - Class 0: 2-community Stochastic Block Model (SBM) — tightly clustered
      - Class 1: Erdős–Rényi (ER) random graph — no community structure

    Each graph has fixed size (n_nodes) and a small node-feature matrix.  Node
    features carry a class-correlated signal that is intentionally noisier in
    the target domain, creating the domain shift.

    Format
    ------
    Each sample ``(x, y)`` where:
      - ``x`` : float32 tensor of shape ``(n_nodes, n_nodes + feat_dim)``
                The first ``n_nodes`` columns are the adjacency matrix row,
                the remaining ``feat_dim`` columns are node features.
                Models can split with: ``adj = x[:, :N]``, ``feats = x[:, N:]``
      - ``y`` : int64 scalar label (0 or 1)

    The packed format enables standard DataLoader batching without PyG:
    a batch has shape ``(B, n_nodes, n_nodes + feat_dim)``.

    Parameters
    ----------
    n_graphs       : number of graphs per domain
    n_nodes        : nodes per graph (fixed; enables tensor batching)
    feat_dim       : node feature dimensionality
    feature_noise  : std of Gaussian noise added to node features
    edge_flip_prob : probability of flipping each edge (structural perturbation)
    p_in, p_out    : SBM intra/inter-community edge probability (class 0)
    p_er           : Erdős–Rényi edge probability (class 1)
    train          : True → training split, False → test split (different seed)
    seed           : base random seed
    """

    def __init__(
        self,
        n_graphs: int = 1000,
        n_nodes: int = 10,
        feat_dim: int = 4,
        feature_noise: float = 0.1,
        edge_flip_prob: float = 0.0,
        p_in: float = 0.7,
        p_out: float = 0.05,
        p_er: float = 0.25,
        train: bool = True,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed if train else seed + 100)

        adjs, feats, labels = [], [], []
        for _ in range(n_graphs):
            label = rng.randint(0, 2)
            adj = (self._sbm(n_nodes, p_in, p_out, rng) if label == 0
                   else self._er(n_nodes, p_er, rng))

            if edge_flip_prob > 0.0:
                adj = self._flip_edges(adj, edge_flip_prob, rng)

            node_feat = self._node_features(adj, n_nodes, feat_dim,
                                             label, feature_noise, rng)
            adjs.append(adj)
            feats.append(node_feat)
            labels.append(label)

        adjs   = np.stack(adjs).astype(np.float32)   # (G, N, N)
        feats  = np.stack(feats).astype(np.float32)  # (G, N, feat_dim)
        labels = np.array(labels, dtype=np.int64)

        # Pack: x[:, :N] = adjacency, x[:, N:] = node features
        x = np.concatenate([adjs, feats], axis=-1)   # (G, N, N+feat_dim)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(labels)
        self.n_nodes  = n_nodes
        self.feat_dim = feat_dim

    # ── graph generators ──────────────────────────────────────────────────

    @staticmethod
    def _sbm(n, p_in, p_out, rng):
        half = n // 2
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                p = p_in if (i < half) == (j < half) else p_out
                if rng.random() < p:
                    adj[i, j] = adj[j, i] = 1.0
        return adj

    @staticmethod
    def _er(n, p, rng):
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i, j] = adj[j, i] = 1.0
        return adj

    @staticmethod
    def _flip_edges(adj, p, rng):
        n = adj.shape[0]
        flip = np.triu(rng.random((n, n)) < p, k=1)
        flip = flip + flip.T
        return np.clip(adj + flip, 0, 1)   # XOR-like but bounded to [0,1]

    @staticmethod
    def _node_features(adj, n, feat_dim, label, noise_std, rng):
        # Feature 0: normalised degree (structural, class-informative)
        degree = adj.sum(axis=1) / max(n - 1, 1)          # (N,)
        # Features 1..: class-offset Gaussian (class 0 → +0.5, class 1 → -0.5)
        class_offset = 0.5 if label == 0 else -0.5
        extra = rng.randn(n, feat_dim - 1) * noise_std + class_offset
        return np.column_stack([degree, extra])            # (N, feat_dim)

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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
        pin = torch.cuda.is_available()
        source_loader = DataLoader(
            source_ds, batch_size=batch_size, shuffle=train,
            num_workers=num_workers, pin_memory=pin
        )
        target_loader = DataLoader(
            target_ds, batch_size=batch_size, shuffle=train,
            num_workers=num_workers, pin_memory=pin
        )
        return source_loader, target_loader

    _REGISTRY["mnist_noisy_mnist"] = _mnist_noisy_mnist

    def _synthetic_graphs(root, batch_size, train, num_workers, **kw):
        """
        Source: clean SyntheticGraphDataset (feature_noise=0.1, no edge flips).
        Target: noisy SyntheticGraphDataset (feature_noise=0.5, edge_flip_prob=0.05).

        Keyword overrides (all optional):
          n_graphs, n_nodes, feat_dim,
          feature_noise_src, feature_noise_tgt, edge_flip_prob
        """
        n_graphs   = kw.get("n_graphs",           1000)
        n_nodes    = kw.get("n_nodes",              10)
        feat_dim   = kw.get("feat_dim",              4)
        noise_src  = kw.get("feature_noise_src",   0.1)
        noise_tgt  = kw.get("feature_noise_tgt",   0.5)
        edge_flip  = kw.get("edge_flip_prob",      0.05)

        src_ds = SyntheticGraphDataset(
            n_graphs=n_graphs, n_nodes=n_nodes, feat_dim=feat_dim,
            feature_noise=noise_src, edge_flip_prob=0.0,
            train=train, seed=42,
        )
        tgt_ds = SyntheticGraphDataset(
            n_graphs=n_graphs, n_nodes=n_nodes, feat_dim=feat_dim,
            feature_noise=noise_tgt, edge_flip_prob=edge_flip,
            train=train, seed=99,   # independent split
        )
        pin = torch.cuda.is_available()
        return (
            DataLoader(src_ds, batch_size=batch_size, shuffle=train,
                       num_workers=num_workers, pin_memory=pin),
            DataLoader(tgt_ds, batch_size=batch_size, shuffle=train,
                       num_workers=num_workers, pin_memory=pin),
        )

    _REGISTRY["synthetic_graphs"] = _synthetic_graphs


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
        num_workers: int = 0,
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
