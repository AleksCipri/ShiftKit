# Models

`shiftkit.models` provides neural network architectures with a shared **encoder / classifier** interface. Splitting these two components is central to how domain adaptation methods operate — the encoder produces a latent representation `z`, and DA losses (e.g. MMD) are computed on `z` directly.

All models implement:

| Method | Signature | Description |
|--------|-----------|-------------|
| `encode` | `(x: Tensor) → Tensor` | Map input to latent vector `z ∈ ℝᵈ` |
| `classify` | `(z: Tensor) → Tensor` | Map latent vector to class logits |
| `forward` | `(x: Tensor) → Tensor` | `classify(encode(x))` — standard `nn.Module` interface |

---

## CNN

A small convolutional network designed for **1×28×28 inputs** (MNIST-like). Two conv-pool blocks feed into a fully-connected bottleneck that produces the latent vector.

```
Input (1×28×28)
  → Conv2d(1→32, k=3) + BN + ReLU + MaxPool  →  32×14×14
  → Conv2d(32→64, k=3) + BN + ReLU + MaxPool  →  64×7×7
  → Flatten → Linear(3136→256) → ReLU → Dropout
  → Linear(256→latent_dim) → ReLU              →  z ∈ ℝᵈ
  → Linear(latent_dim→num_classes)              →  logits
```

```python
from shiftkit.models import CNN

model = CNN(latent_dim=128, num_classes=10, dropout=0.3)

z      = model.encode(x)    # (B, 128)
logits = model.classify(z)  # (B, 10)
logits = model(x)           # equivalent
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent_dim` | `int` | `128` | Dimensionality of the bottleneck embedding |
| `num_classes` | `int` | `10` | Number of output classes |
| `dropout` | `float` | `0.3` | Dropout probability before the final FC layer |

---

## MLP

A fully-connected network that flattens the input and passes it through configurable hidden layers before the bottleneck.

```
Input (1×28×28 → flattened 784)
  → Linear(784→h₁) + ReLU + Dropout
  → Linear(h₁→h₂)  + ReLU + Dropout
  → ...
  → Linear(hₙ→latent_dim) + ReLU    →  z ∈ ℝᵈ
  → Linear(latent_dim→num_classes)   →  logits
```

```python
from shiftkit.models import MLP

model = MLP(latent_dim=128, num_classes=10, hidden_dims=(512, 256), dropout=0.3)

z      = model.encode(x)    # (B, 128)
logits = model.classify(z)  # (B, 10)
logits = model(x)           # equivalent
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent_dim` | `int` | `128` | Dimensionality of the bottleneck embedding |
| `num_classes` | `int` | `10` | Number of output classes |
| `hidden_dims` | `Tuple[int, ...]` | `(512, 256)` | Sizes of hidden layers before the bottleneck |
| `dropout` | `float` | `0.3` | Dropout probability after each hidden layer |

---

## Choosing between CNN and MLP

| | CNN | MLP |
|--|-----|-----|
| Input type | 2-D images (preserves spatial structure) | Any flattened vector |
| Inductive bias | Translation equivariance | None |
| Parameters (default) | ~856 K | ~560 K |
| Speed | Slightly faster on GPU | Slightly faster on CPU |

For image inputs, **CNN is recommended**. MLP is useful when inputs are already feature vectors.

---

## SimpleGCN

A two-layer **Graph Convolutional Network** for graph classification. Designed to work with [`SyntheticGraphDataset`](data.md#syntheticgraphdataset) and the packed `(B, N, N+feat_dim)` tensor format — **no PyTorch Geometric required**.

```
Input x: (B, N, N+feat_dim)
  split → adj (B,N,N) + h₀ (B,N,feat_dim)
  â = D̂⁻¹/²(A+I)D̂⁻¹/²              (symmetric normalisation with self-loops)
  h₁ = ReLU(â · GCN₁(h₀))           (B, N, hidden_dim)
  h₂ = ReLU(â · GCN₂(h₁))           (B, N, latent_dim)
  z  = h₂.mean(dim=1)                (B, latent_dim)  — graph-level embedding
  logits = Linear(latent_dim → num_classes)
```

```python
from shiftkit.models import SimpleGCN

model = SimpleGCN(n_nodes=10, feat_dim=4, latent_dim=64, num_classes=2)

# x shape: (B, N, N+feat_dim)  — first N cols = adjacency, rest = features
z      = model.encode(x)    # (B, 64)
logits = model.classify(z)  # (B, 2)
logits = model(x)           # equivalent
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_nodes` | `int` | `10` | Number of nodes per graph (must match dataset) |
| `feat_dim` | `int` | `4` | Number of node feature dimensions |
| `latent_dim` | `int` | `64` | Graph-level embedding dimensionality |
| `num_classes` | `int` | `2` | Number of output classes |
| `hidden_dim` | `int` | `64` | Hidden dimensionality of the first GCN layer |
| `dropout` | `float` | `0.0` | Dropout probability between GCN layers |

### End-to-end example

```python
from shiftkit.data import DataManager
from shiftkit.models import SimpleGCN
from shiftkit.methods import MMDTrainer

dm = DataManager(batch_size=32)
train_src, train_tgt = dm.load("synthetic_graphs", train=True)
test_src,  test_tgt  = dm.load("synthetic_graphs", train=False)

model = SimpleGCN(n_nodes=10, feat_dim=4, latent_dim=64, num_classes=2)

trainer = MMDTrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    mmd_weight=1.0,
    warmup_epochs=5,
    lr=1e-3,
)
history = trainer.fit(epochs=30)
result  = trainer.evaluate(test_tgt, domain="target-test")
print(f"Target accuracy: {result['accuracy']*100:.1f}%")
```

---

## GNN

A configurable **PyTorch Geometric** GNN that supports both graph-level and node-level domain adaptation.
Uses the same `encode` / `classify` / `regress` / `forward` interface as `CNN`, `MLP`, and `SimpleGCN`.

!!! note "Optional dependency"
    `GNN` requires `torch-geometric`, which is not installed by default:
    ```bash
    pip install torch-geometric
    ```

Supported conv layers:

| `model_name` | Architecture | Notes |
|---|---|---|
| `"SAGE"` | GraphSAGE | Mean/max/sum neighbourhood aggregation; good default |
| `"GCN"` | Graph Convolutional Network | Spectral-style; assumes undirected graphs |
| `"GAT"` | Graph Attention Network | Attention-weighted aggregation |
| `"GIN"` | Graph Isomorphism Network | Maximally expressive (Weisfeiler-Leman); uses inner MLP |
| `"GraphConv"` | General Graph Conv | Learnable self + neighbour weights |

Supported pooling (`pool` parameter):

| `pool` | Output shape | Use case |
|---|---|---|
| `"mean"` / `"max"` / `"sum"` | `(num_graphs, hidden_channels)` | **Graph-level** DA |
| `"none"` | `(num_nodes, hidden_channels)` | **Node-level** DA |

### Graph-level example

```python
from shiftkit.models import GNN
from shiftkit.data import DataManager
from shiftkit.methods import MMDTrainer

# source_graph / target_graph are lists of PyG Data objects
dm = DataManager(batch_size=32)
train_src, train_tgt = dm.load(
    "pyg_domains", train=True, task_level="graph",
    source=list_of_src_graphs, target=list_of_tgt_graphs,
)

model = GNN(list_of_src_graphs[0], "SAGE",
            hidden_channels=64, num_layers=3, num_classes=2)

trainer = MMDTrainer(model, train_src, train_tgt, mmd_weight=1.0, lr=1e-3)
trainer.fit(epochs=30)
```

### Node-level example

```python
from shiftkit.models import GNN
from shiftkit.data import DataManager
from shiftkit.methods import MMDTrainer

# source_graph / target_graph are single PyG Data objects (one graph per domain)
dm = DataManager(batch_size=1, num_workers=0)
train_src, train_tgt = dm.load(
    "pyg_domains", train=True, task_level="node",
    source=source_graph, target=target_graph,
    train_ratio=0.6, val_ratio=0.2, split_seed=42, split_mode="stratified",
)

model = GNN(source_graph, "SAGE",
            hidden_channels=32, num_layers=2, num_classes=3,
            pool="none")   # required: returns per-node embeddings

trainer = MMDTrainer(model, train_src, train_tgt, mmd_weight=0.5, lr=1e-3)
trainer.fit(epochs=30)
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `Data` | — | Template PyG `Data` object; `num_node_features` is read from it |
| `model_name` | `str` | — | Conv layer type (see table above) |
| `hidden_channels` | `int` | — | Width of each conv layer; also the latent dimension |
| `num_layers` | `int` | — | Number of message-passing layers (≥ 1) |
| `num_classes` | `int` | `2` | Output classes for `classify()` |
| `regress` | `bool` | `False` | Build a scalar regression head instead of classifier |
| `pool` | `str` | `"mean"` | Readout aggregation; use `"none"` for node-level |
| `use_layernorm` | `bool` | `True` | Apply `LayerNorm` after each conv |
| `dropout` | `float` | `0.0` | Dropout probability between conv layers |
| `aggr` | `str` | `"mean"` | Neighbour aggregation for SAGE and GraphConv |

---

## Choosing between SimpleGCN and GNN

| | `SimpleGCN` | `GNN` |
|--|-------------|-------|
| PyTorch Geometric required | No | Yes (`pip install torch-geometric`) |
| Install complexity | None — works with base ShiftKit install | Medium — must match PyTorch and CUDA versions |
| Input format | Packed `(B, N, N+feat_dim)` tensor | PyG `Data` / `Batch` |
| Graph size | Fixed `n_nodes` per batch | Variable |
| Conv options | GCN only | SAGE, GCN, GAT, GIN, GraphConv |
| Task level | Graph-level only | Graph-level **and** node-level |
| Best for | Zero-dependency first experiments | Real data, expressive convs, node-level DA |

Use `SimpleGCN` when you want no extra dependencies and your graphs are small and fixed-size.
Use `GNN` for graph-level problems when you need a more expressive architecture — e.g. GAT attention, GIN's injective aggregation, or variable-size graphs — or when you need node-level predictions.

---

## Using a custom model

Any model that exposes `.encode(x)` and `.classify(z)` can be used with `MMDTrainer` and `SourceOnlyTrainer`:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU())
        self.classifier = nn.Linear(64, 10)

    def encode(self, x):
        return self.encoder(x)

    def classify(self, z):
        return self.classifier(z)

    def forward(self, x):
        return self.classify(self.encode(x))
```
