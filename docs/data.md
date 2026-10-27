# Data

`shiftkit.data` provides dataset loading and paired source/target DataLoader creation.

---

## DataManager

Central hub for loading source/target domain data. Maintains a registry of dataset-pair factories and returns paired `DataLoader` objects.

```python
from shiftkit.data import DataManager

dm = DataManager(root="./data", batch_size=64)
train_src, train_tgt = dm.load("mnist_noisy_mnist", train=True)
test_src,  test_tgt  = dm.load("mnist_noisy_mnist", train=False)
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root` | `str` | `"./data"` | Root directory where datasets are downloaded |
| `batch_size` | `int` | `64` | Batch size for both DataLoaders |
| `num_workers` | `int` | `0` | Number of DataLoader worker processes |

### Methods

#### `load(name, train=True, **kwargs)`

Return `(source_loader, target_loader)` for the named dataset pair.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Registered dataset key, e.g. `"mnist_noisy_mnist"` |
| `train` | `bool` | `True` | Load training split if `True`, test split if `False` |
| `**kwargs` | | | Forwarded to the factory (e.g. `noise_std=0.5`) |

**Returns:** `(DataLoader, DataLoader)` — source loader, target loader

**Raises:** `ValueError` if `name` is not registered.

#### `register(name, factory)` *(static)*

Register a custom dataset-pair factory.

```python
def my_factory(root, batch_size, train, num_workers, **kwargs):
    source_ds = ...  # your source dataset
    target_ds = ...  # your target dataset
    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=train)
    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=train)
    return source_loader, target_loader

DataManager.register("my_pair", my_factory)

# then use it:
dm = DataManager()
src, tgt = dm.load("my_pair")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Key to register under |
| `factory` | `callable` | Function with signature `(root, batch_size, train, num_workers, **kwargs)` → `(DataLoader, DataLoader)` |

#### `available()` *(static)*

Return a list of all registered dataset-pair names.

```python
print(DataManager.available())
# ['mnist_noisy_mnist', 'my_pair', ...]
```

---

## NoisyMNIST

A `torch.utils.data.Dataset` that wraps `torchvision.datasets.MNIST` and adds per-sample Gaussian noise. Used as the built-in synthetic target domain.

```python
from shiftkit.data.datasets import NoisyMNIST

ds = NoisyMNIST(root="./data", train=True, noise_std=0.3)
img, label = ds[0]   # img is a clipped noisy tensor in [0, 1]
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root` | `str` | — | Path to dataset directory |
| `train` | `bool` | `True` | Training split if `True`, test split if `False` |
| `noise_std` | `float` | `0.3` | Standard deviation of additive Gaussian noise |
| `transform` | `callable` | `None` | Additional transforms applied after noise injection |
| `download` | `bool` | `True` | Download the dataset if not present |

| `noise_seed` | `int` | `0` | Base seed for the per-sample noise |
| `fixed_noise` | `bool` | `True` | If `True`, sample `idx` always gets the same noise (generator seeded from `(noise_seed, idx)`); if `False`, fresh noise from the global RNG on every access |

!!! note
    Noise is injected as `img = (img + N(0, noise_std²)).clamp(0, 1)`.
    The `"mnist_noisy_mnist"` factory uses **fixed** noise for the test split, so every
    evaluation sees the same corrupted test images, and **fresh** noise on every access
    for the training split (an augmentation; set `fixed_noise=True` in `dm.load` to
    freeze it). Train and test noise use different base seeds; override with
    `noise_seed=`.

---

## SyntheticGraphDataset

A purely synthetic graph classification dataset designed as a minimal benchmark for **graph-based domain adaptation**. No download required — graphs are generated once, when the dataset is constructed.

Each sample is a small graph with **N=10 nodes** and **2 classes**:

| Class | Generator | Structure |
|-------|-----------|-----------|
| 0 | Stochastic Block Model (SBM) | Two tightly-connected communities (p_in=0.7, p_out=0.05) |
| 1 | Erdős–Rényi (ER) | Uniform random edges (p=0.25) |

**Domain shift** is introduced by two mechanisms in the target domain:

- **Feature noise**: node features have higher Gaussian noise (σ=0.5 vs σ=0.1 in the source)
- **Edge flips**: each node pair is independently toggled with probability 5% (XOR: an existing edge is removed, a missing edge is added; the adjacency stays symmetric with no self-loops)

**Tensor format** — each sample `x` has shape `(N, N + feat_dim)`. The first `N` columns are the adjacency matrix; the remaining columns are node features `[degree / (N-1), class-offset Gaussian noise]`. This packed format is compatible with standard `DataLoader` batching without requiring PyTorch Geometric.

```python
from shiftkit.data.datasets import SyntheticGraphDataset
from torch.utils.data import DataLoader

src_ds = SyntheticGraphDataset(feature_noise=0.1, edge_flip_prob=0.0)
tgt_ds = SyntheticGraphDataset(feature_noise=0.5, edge_flip_prob=0.05, seed=99)

src_loader = DataLoader(src_ds, batch_size=32, shuffle=True)
x, y = next(iter(src_loader))
# x: (32, 10, 14)   — 10 nodes, 10-col adj + 4-col features
# y: (32,)          — class labels {0, 1}
```

Or via `DataManager`:

```python
dm = DataManager(batch_size=32)
train_src, train_tgt = dm.load("synthetic_graphs", train=True)
test_src,  test_tgt  = dm.load("synthetic_graphs", train=False)
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_graphs` | `int` | `1000` | Number of graphs **in this split**: `train=True` and `train=False` each generate `n_graphs` graphs (with seeds `seed` and `seed + 100`); not an 80/20 split |
| `n_nodes` | `int` | `10` | Number of nodes per graph |
| `feat_dim` | `int` | `4` | Number of node feature dimensions |
| `feature_noise` | `float` | `0.1` | Std dev of additive Gaussian feature noise |
| `edge_flip_prob` | `float` | `0.0` | Probability of toggling each node pair (XOR) |
| `p_in` | `float` | `0.7` | SBM within-community edge probability (class 0) |
| `p_out` | `float` | `0.05` | SBM between-community edge probability (class 0) |
| `p_er` | `float` | `0.25` | Erdős–Rényi edge probability (class 1) |
| `train` | `bool` | `True` | If `True` generate the training split (seed `seed`), else the test split (seed `seed + 100`) |
| `seed` | `int` | `42` | Random seed for reproducibility |

---

## CaliforniaHousingDataset

Geographic split of scikit-learn's California Housing data: source = north
(latitude ≥ 36°, 8,826 districts), target = south (latitude < 36°, 11,814 districts),
each split 80/20 into train/test with `seed`. scikit-learn **downloads** the data on
first use unless it is already cached (`data_home`, `download_if_missing`).

Features are standardised with statistics fitted on **training rows only** and the
same statistics are applied to all splits of both domains. Choose with `norm_fit_on`:

- `"source_train"` (default) — northern training rows only;
- `"source_target_train"` — northern training rows plus the *unlabelled inputs* of the
  southern training rows. This is a legitimate unsupervised-DA choice, but it makes the
  preprocessing depend on target data. Test rows are never used.

```python
dm = DataManager(batch_size=128)
train_src, train_tgt = dm.load("california_housing", train=True)
test_src,  test_tgt  = dm.load("california_housing", train=False)
```

A geographic split does not by itself guarantee that the price-given-features
relationship is the same in both regions.

---

## Built-in dataset pairs

| Key | Source | Target | Extra kwargs |
|-----|--------|--------|-------------|
| `"mnist_noisy_mnist"` | `torchvision.MNIST` | `NoisyMNIST` | `noise_std` (default `0.3`), `noise_seed`, `fixed_noise` |
| `"synthetic_graphs"` | `SyntheticGraphDataset` (noise=0.1) | `SyntheticGraphDataset` (noise=0.5, flip=0.05) | `n_graphs` (per split), `n_nodes`, `feat_dim`, `feature_noise_src`, `feature_noise_tgt`, `edge_flip_prob` |
| `"sine_wave"` | `SineWaveDataset` (phase 0) | `SineWaveDataset` (phase π/3) | `n_samples`, `phase_src`, `phase_tgt`, `noise_std` |
| `"california_housing"` | `CaliforniaHousingDataset` north (lat ≥ 36°) | `CaliforniaHousingDataset` south (lat < 36°) | `seed`, `norm_fit_on`, `data_home`, `download_if_missing` |
| `"pyg_domains"` | User-supplied PyG `Data` | User-supplied PyG `Data` | `source`, `target`, `task_level`, `train_ratio`, `val_ratio`, `split_seed`, `split_mode`, `split` |

---

## PyG domains (`pyg_domains`)

Requires the optional dependency `torch-geometric` (`pip install torch-geometric`); the rest of ShiftKit works without it. Loads external PyG graphs for domain adaptation with automatic **stratified** train/val/test splits when masks are not already on the `Data` objects.

### Node-level (one graph per domain)

Use when labels live on **nodes** (transductive node classification). Each domain is a single `torch_geometric.data.Data` object. Training runs message passing on the full graph; loss and MMD use **train** nodes only; evaluation uses **test** nodes.

```python
from shiftkit.data import DataManager
from shiftkit.models import GNN
from shiftkit.methods import MMDTrainer

dm = DataManager(batch_size=1, num_workers=0)
train_src, train_tgt = dm.load(
    "pyg_domains",
    train=True,
    task_level="node",
    source=source_graph,
    target=target_graph,
    train_ratio=0.6,
    val_ratio=0.2,
    split_seed=42,
    split_mode="stratified",
)
test_src, test_tgt = dm.load(
    "pyg_domains",
    train=False,
    task_level="node",
    source=source_graph,
    target=target_graph,
    train_ratio=0.6,
    val_ratio=0.2,
    split_seed=42,
    split_mode="stratified",
)

model = GNN(source_graph, "SAGE", hidden_channels=64, num_layers=2,
            num_classes=10, pool="none")
trainer = MMDTrainer(model, train_src, train_tgt, mmd_weight=1.0)
```

Pair with `shiftkit.models.GNN(..., pool="none")` so `encode()` returns per-node embeddings.

**Splits.** Source masks use `split_seed`, target masks `split_seed + 1`. In stratified
mode every class with at least one node gets at least one training node; classes with
fewer than three nodes get no validation node (and a single-node class no test node),
and a warning is issued. `train=True` yields train nodes and `train=False` test nodes.
Validation nodes are never used for training; load them with `split="val"`:

```python
val_src, val_tgt = dm.load(
    "pyg_domains",
    split="val",
    task_level="node",
    source=source_graph,
    target=target_graph,
    train_ratio=0.6,
    val_ratio=0.2,
    split_seed=42,
)
```

Use target validation **labels** only if your problem setting allows a small labelled
target set; see [Tuning λ](tutorials/tuning.md).

### Graph-level (many graphs per domain)

Pass a **list** of `Data` objects per domain. Splits are by graph index (stratified on graph labels when discrete). Use default `pool="mean"` on `GNN`.

!!! note "Graph-level training includes validation graphs"
    Unlike the node-level path, `train=True` returns the **train and validation** graphs
    together (`train_mask | val_mask`); `train=False` returns the test graphs. There is
    no separate graph-level validation loader (`split="val"` is node-level only). If you
    need a validation set, split your graph list yourself before calling `load`.

```python
train_src, train_tgt = dm.load(
    "pyg_domains",
    train=True,
    task_level="graph",
    source=list_of_src_graphs,
    target=list_of_tgt_graphs,
    train_ratio=0.6,
    val_ratio=0.2,
)
```

### Existing masks

If you set `data.train_mask` yourself, automatic splitting is skipped and your masks are
used: `train_mask` for `train=True`, `test_mask` for `train=False` and `val_mask` for
`split="val"`. Masks created by ShiftKit record the split parameters; loading the same
`Data` object again with a different `train_ratio`, `val_ratio`, `split_seed` or
`split_mode` recomputes them rather than silently reusing the old split. Masks are
written onto the `Data` object in place, so pass separate objects (e.g. `data.clone()`)
if the same graph is used as both source and target.

See `examples/pyg_node_mmd.py` for a full node-level example.
