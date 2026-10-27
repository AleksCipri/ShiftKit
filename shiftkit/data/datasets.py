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
    """
    MNIST with additive Gaussian noise — used as a synthetic target domain.

    By default the noise for sample ``idx`` is drawn from a generator seeded by
    ``(noise_seed, idx)``, so every access to the same index returns the same
    corrupted image. Evaluating a model twice on the test split therefore uses
    the same fixed corruption. Pass ``fixed_noise=False`` to restore the old
    behaviour of drawing fresh noise from the global RNG on every access (an
    augmentation that makes repeated evaluations differ).
    """

    def __init__(self, root: str, train: bool = True, noise_std: float = 0.3,
                 transform=None, download: bool = True,
                 noise_seed: int = 0, fixed_noise: bool = True):
        self.base = torchvision.datasets.MNIST(
            root=root, train=train, download=download,
            transform=T.ToTensor()
        )
        self.noise_std = noise_std
        self.extra_transform = transform
        self.noise_seed = int(noise_seed)
        self.fixed_noise = fixed_noise

    def __len__(self):
        return len(self.base)

    def _noise_like(self, img: torch.Tensor, idx: int) -> torch.Tensor:
        """Return N(0, noise_std²) noise for *idx* (deterministic if fixed_noise)."""
        if not self.fixed_noise:
            return torch.randn_like(img) * self.noise_std
        state = np.random.SeedSequence([self.noise_seed, int(idx)]).generate_state(2)
        gen = torch.Generator().manual_seed(int(state[0]) << 32 | int(state[1]))
        noise = torch.randn(img.shape, generator=gen, dtype=img.dtype)
        return noise.to(img.device) * self.noise_std

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = (img + self._noise_like(img, idx)).clamp(0.0, 1.0)
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
    n_graphs       : number of graphs generated for *this split* (the train and
                     test splits each contain ``n_graphs`` graphs, drawn with
                     different seeds; this is not an 80/20 split of one pool)
    n_nodes        : nodes per graph (fixed; enables tensor batching)
    feat_dim       : node feature dimensionality
    feature_noise  : std of Gaussian noise added to node features
    edge_flip_prob : probability of toggling each node pair (XOR: an existing
                     edge is removed, a missing edge is added)
    p_in, p_out    : SBM intra/inter-community edge probability (class 0)
    p_er           : Erdős–Rényi edge probability (class 1)
    train          : True → training split (``seed``), False → test split
                     (``seed + 100``). Graphs are generated once at construction.
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
        """Toggle each off-diagonal undirected edge independently with prob. *p*.

        True XOR: present edges are removed and absent edges are added. The
        result stays symmetric with an empty diagonal (no self-loops).
        """
        n = adj.shape[0]
        flip = np.triu(rng.random((n, n)) < p, k=1)
        flip = flip | flip.T
        out = np.logical_xor(adj > 0, flip).astype(adj.dtype)
        np.fill_diagonal(out, 0)
        return out

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


# ─── Sine Wave Regression Dataset ────────────────────────────────────────────

class SineWaveDataset(Dataset):
    """
    Synthetic regression dataset for domain adaptation demonstrations.

    Source domain: y = sin(x) + noise
    Target domain: y = sin(x + phase_shift) + noise  (default: π/3 shift)

    Both domains draw x from the same range, so the phase shift changes only
    p(y|x) (concept shift).  Unlabelled marginal alignment cannot identify the
    new phase: use this as a limitation example, not as a demonstration that
    feature alignment helps.  For a covariate-shift example, use different
    ``x_range`` values with ``phase=0`` for both domains.

    Parameters
    ----------
    n_samples    : number of (x, y) pairs
    phase        : phase offset applied to the sine function
    amplitude    : vertical scaling factor
    noise_std    : standard deviation of additive Gaussian noise
    x_range      : input range, default (−π, π)
    train        : True → training split, False → held-out test split
    seed         : base random seed
    """

    def __init__(
        self,
        n_samples: int = 2000,
        phase: float = 0.0,
        amplitude: float = 1.0,
        noise_std: float = 0.05,
        x_range: tuple = (-np.pi, np.pi),
        train: bool = True,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed if train else seed + 100)
        x = rng.uniform(*x_range, n_samples).astype(np.float32)
        y = (amplitude * np.sin(x + phase)
             + rng.randn(n_samples).astype(np.float32) * noise_std)
        self.x = torch.from_numpy(x).unsqueeze(1)   # (N, 1)
        self.y = torch.from_numpy(y).unsqueeze(1)   # (N, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ─── California Housing Regression Dataset ────────────────────────────────────

def _fetch_california(data_home=None, download_if_missing=True):
    try:
        from sklearn.datasets import fetch_california_housing
    except ImportError as e:
        raise ImportError(
            "CaliforniaHousingDataset requires scikit-learn. "
            "Install it with: pip install scikit-learn"
        ) from e
    # scikit-learn downloads the data on first use unless it is already cached.
    return fetch_california_housing(
        data_home=data_home, download_if_missing=download_if_missing
    )


def _california_split_indices(latitude: np.ndarray, domain: str, seed: int):
    """Return (train_idx, test_idx) into the full dataset for one region."""
    if domain not in ("north", "south"):
        raise ValueError(f"domain must be 'north' or 'south', got {domain!r}")
    mask = (latitude >= 36.0) if domain == "north" else (latitude < 36.0)
    region = np.flatnonzero(mask)
    # Reproducible 80/20 train-test split within the region
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(region))
    split = int(0.8 * len(region))
    return region[perm[:split]], region[perm[split:]]


def california_normalisation_stats(
    seed: int = 42,
    fit_on: str = "source_train",
    data=None,
    data_home=None,
    download_if_missing: bool = True,
):
    """
    Per-feature ``(x_mean, x_std)`` for the California Housing domain split.

    Statistics are fitted on *training* inputs only, never on test rows:

    - ``fit_on="source_train"`` (default): northern (source) training rows.
    - ``fit_on="source_target_train"``: northern training rows plus the
      *unlabelled inputs* of the southern (target) training rows. Using
      unlabelled target training inputs is a legitimate choice in
      unsupervised DA, but it makes preprocessing depend on target data.

    Labels are never used. The same statistics should be applied to every
    split of both domains.
    """
    if data is None:
        data = _fetch_california(data_home, download_if_missing)
    X_all = data.data.astype(np.float32)
    lat = data.data[:, 6]
    rows = [_california_split_indices(lat, "north", seed)[0]]
    if fit_on == "source_target_train":
        rows.append(_california_split_indices(lat, "south", seed)[0])
    elif fit_on != "source_train":
        raise ValueError(
            "fit_on must be 'source_train' or 'source_target_train', "
            f"got {fit_on!r}"
        )
    X_fit = X_all[np.concatenate(rows)]
    return X_fit.mean(0), X_fit.std(0) + 1e-8


class CaliforniaHousingDataset(Dataset):
    """
    Geographic domain split of the California Housing dataset (sklearn builtin).

    Source domain: Northern California — latitude >= 36° (8,826 districts)
    Target domain: Southern California — latitude <  36° (11,814 districts)

    The two regions differ in price distribution, housing density and feature
    correlations. A geographic split alone does not guarantee that the
    conditional distribution of prices given features is the same in both.

    Features are standardised with statistics fitted on **training rows only**
    (see :func:`california_normalisation_stats`) and the same statistics are
    applied to the train and test splits of both domains. If ``x_mean`` /
    ``x_std`` are not given, the statistics are fitted on the northern (source)
    training rows for this ``seed``, so standalone datasets share one scale.

    scikit-learn downloads the data on first use if it is not cached in
    ``data_home``.

    Parameters
    ----------
    domain   : ``"north"`` (source) or ``"south"`` (target)
    train    : True → 80 % training split, False → 20 % test split
    seed     : random seed for the train/test shuffle
    x_mean   : per-feature mean for normalisation (numpy array, shape (8,))
    x_std    : per-feature std  for normalisation (numpy array, shape (8,))
    data_home, download_if_missing : forwarded to
               ``sklearn.datasets.fetch_california_housing``
    """

    FEATURE_NAMES = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude",
    ]

    def __init__(
        self,
        domain: str = "north",
        train: bool = True,
        seed: int = 42,
        x_mean=None,
        x_std=None,
        data_home=None,
        download_if_missing: bool = True,
    ):
        data = _fetch_california(data_home, download_if_missing)
        X_all = data.data.astype(np.float32)
        y_all = data.target.astype(np.float32)

        # Normalise with training-only statistics (never test rows)
        if (x_mean is None) != (x_std is None):
            raise ValueError("Pass both x_mean and x_std, or neither.")
        if x_mean is None:
            x_mean, x_std = california_normalisation_stats(seed=seed, data=data)

        # Geographic split on raw latitude (feature index 6 in original data),
        # then a reproducible 80/20 train-test split within the region
        tr_idx, te_idx = _california_split_indices(data.data[:, 6], domain, seed)
        idx = tr_idx if train else te_idx

        X = (X_all[idx] - x_mean) / x_std
        self.x = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y_all[idx]).unsqueeze(1)   # (N, 1)
        self.x_mean = x_mean
        self.x_std  = x_std

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
        # Test split: fixed per-index noise, so repeated evaluations see the
        # same corrupted images. Train split: fresh noise on every access by
        # default (augmentation, as before); pass fixed_noise=True to freeze it.
        # Train and test use different base seeds.
        noise_seed = int(kw.get("noise_seed", 0)) + (0 if train else 1_000_003)
        fixed_noise = bool(kw.get("fixed_noise", not train))
        base_tf = T.Normalize((0.1307,), (0.3081,))

        source_ds = torchvision.datasets.MNIST(
            root=root, train=train, download=True,
            transform=T.Compose([T.ToTensor(), base_tf])
        )
        target_ds = NoisyMNIST(
            root=root, train=train, noise_std=noise_std,
            transform=base_tf, download=True,
            noise_seed=noise_seed, fixed_noise=fixed_noise,
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

    def _sine_wave(root, batch_size, train, num_workers, **kw):
        """
        Source: sin(x),          phase = 0.
        Target: sin(x + π/3),   phase = π/3 (60° shift).

        Keyword overrides (all optional):
          n_samples, phase_src, phase_tgt, noise_std
        """
        n_samples = kw.get("n_samples", 2000)
        phase_src = kw.get("phase_src", 0.0)
        phase_tgt = kw.get("phase_tgt", float(np.pi / 3))
        noise_std = kw.get("noise_std", 0.05)

        src_ds = SineWaveDataset(
            n_samples=n_samples, phase=phase_src, noise_std=noise_std, train=train, seed=42
        )
        tgt_ds = SineWaveDataset(
            n_samples=n_samples, phase=phase_tgt, noise_std=noise_std, train=train, seed=99
        )
        pin = torch.cuda.is_available()
        return (
            DataLoader(src_ds, batch_size=batch_size, shuffle=train,
                       num_workers=num_workers, pin_memory=pin),
            DataLoader(tgt_ds, batch_size=batch_size, shuffle=train,
                       num_workers=num_workers, pin_memory=pin),
        )

    _REGISTRY["sine_wave"] = _sine_wave

    def _california_housing(root, batch_size, train, num_workers, **kw):
        """
        Source: Northern California (latitude >= 36).
        Target: Southern California (latitude <  36).

        Normalisation statistics are fitted once on training rows only and
        shared by both domains and both splits.

        Keyword overrides (all optional):
          seed (42), norm_fit_on ("source_train" | "source_target_train"),
          data_home, download_if_missing
        """
        seed = int(kw.get("seed", 42))
        data_home = kw.get("data_home")
        download_if_missing = kw.get("download_if_missing", True)
        x_mean, x_std = california_normalisation_stats(
            seed=seed,
            fit_on=kw.get("norm_fit_on", "source_train"),
            data_home=data_home,
            download_if_missing=download_if_missing,
        )
        common = dict(train=train, seed=seed, x_mean=x_mean, x_std=x_std,
                      data_home=data_home,
                      download_if_missing=download_if_missing)
        src_ds = CaliforniaHousingDataset(domain="north", **common)
        tgt_ds = CaliforniaHousingDataset(domain="south", **common)
        pin = torch.cuda.is_available()
        return (
            DataLoader(src_ds, batch_size=batch_size, shuffle=train,
                       num_workers=num_workers, pin_memory=pin),
            DataLoader(tgt_ds, batch_size=batch_size, shuffle=train,
                       num_workers=num_workers, pin_memory=pin),
        )

    _REGISTRY["california_housing"] = _california_housing

    def _pyg_domains(root, batch_size, train, num_workers, **kw):
        """
        PyG source/target domain pair (graph-level or node-level).

        Required kwargs
        ---------------
        source, target : PyG ``Data`` (node-level) or list/dataset of ``Data`` (graph-level)
        task_level     : ``"node"`` or ``"graph"`` (default ``"node"``)

        Optional kwargs
        -----------------
        train_ratio, val_ratio, split_seed, split_mode (``"stratified"`` | ``"random"``),
        split (node-level only: ``"train"`` | ``"val"`` | ``"test"``; overrides ``train``)
        """
        from .pyg_utils import build_pyg_domain_loaders

        source = kw.get("source")
        target = kw.get("target")
        if source is None or target is None:
            raise ValueError(
                "pyg_domains requires 'source' and 'target' PyG Data object(s). "
                "Example: dm.load('pyg_domains', source=src_data, target=tgt_data, ...)"
            )
        task_level = kw.get("task_level", "node")
        train_ratio = float(kw.get("train_ratio", 0.6))
        val_ratio = float(kw.get("val_ratio", 0.2))
        split_seed = int(kw.get("split_seed", 42))
        split_mode = kw.get("split_mode", "stratified")

        return build_pyg_domain_loaders(
            task_level=task_level,
            source=source,
            target=target,
            train=train,
            batch_size=batch_size,
            num_workers=num_workers,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            split_seed=split_seed,
            split_mode=split_mode,
            split=kw.get("split"),
        )

    _REGISTRY["pyg_domains"] = _pyg_domains


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
