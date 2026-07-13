# Tutorial 7 — Node-level DA with PyTorch Geometric

Most ShiftKit tutorials work with **image or tabular data** where each sample is an independent vector.
Graph-structured data is different: relationships between entities are part of the input.
This tutorial introduces **node-level domain adaptation** using PyTorch Geometric (PyG), where the goal is to
transfer a node classifier from one graph (source domain) to another (target domain).

The complete runnable script lives at `examples/pyg_node_mmd.py`.

---

## Graph-level vs node-level DA

ShiftKit supports two flavours of graph domain adaptation:

| | Graph-level | Node-level |
|--|-------------|------------|
| Domain = | many small graphs | one large graph |
| Prediction target | one label per graph | one label per node |
| Training | mini-batches of graphs | mini-batches of **node subsets** from the full graph |
| `GNN pool` setting | `"mean"`, `"max"`, `"sum"` | **`"none"`** |
| DataManager `task_level` | `"graph"` | `"node"` |

This tutorial focuses on **node-level** DA — a transductive setting common in social networks,
citation graphs, and particle physics interaction graphs.

---

## The domain gap

We create two synthetic chain graphs, each with 300 nodes, 8 node features, and 3 classes:

```
source: x ~ N(0, 1)      shift = 0.0
target: x ~ N(1.5, 1)    shift = 1.5
```

The topology is identical (a bidirectional chain 0↔1↔2↔…↔299), but the feature distributions
are offset by 1.5 standard deviations. A model trained only on the source graph will see a
covariate shift on the target and classify poorly.

MMD alignment encourages the GNN encoder to map both graphs' node embeddings into a shared
latent space, reducing the feature-distribution gap.

---

## Run the script

```bash
pip install torch-geometric   # one-time; PyG is an optional ShiftKit dependency
python examples/pyg_node_mmd.py
```

---

## Code walkthrough

### 1 — Build the domain graphs

```python
import torch
from torch_geometric.data import Data

def make_domain_graph(n_nodes, feat_dim, n_classes, seed, shift=0.0):
    torch.manual_seed(seed)
    x = torch.randn(n_nodes, feat_dim) + shift          # node features
    row = torch.arange(n_nodes - 1)
    edge_index = torch.stack([row, row + 1], dim=0)     # chain topology
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # bidirectional
    y = torch.randint(0, n_classes, (n_nodes,))
    return Data(x=x, edge_index=edge_index, y=y)

source_graph = make_domain_graph(300, feat_dim=8, n_classes=3, seed=0, shift=0.0)
target_graph = make_domain_graph(300, feat_dim=8, n_classes=3, seed=1, shift=1.5)
```

Each `Data` object holds the **full graph** — message passing runs over all nodes, but
training loss and MMD are computed only on the train-split node indices.

---

### 2 — Create stratified node-split loaders

```python
from shiftkit.data import DataManager

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
    split_mode="stratified",   # stratify splits by class label
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
```

!!! note "`batch_size=1`"
    For node-level DA each domain is a **single graph**, so the DataLoader always yields
    one item per step — the full graph plus a set of train/test node indices.
    `batch_size=1` is required.

!!! note "No download needed"
    Graphs are passed directly as `Data` objects. `DataManager` does not write to disk.

---

### 3 — Build the GNN with `pool="none"`

```python
from shiftkit.models import GNN

model_so  = GNN(source_graph, "SAGE", hidden_channels=32, num_layers=2,
                num_classes=3, pool="none")
model_mmd = GNN(source_graph, "SAGE", hidden_channels=32, num_layers=2,
                num_classes=3, pool="none")
```

!!! warning "`pool=\"none\"` is required for node-level DA"
    The default `pool="mean"` collapses all node embeddings into a single graph-level vector.
    For node classification you need **per-node** embeddings, which requires `pool="none"`.
    `encode()` then returns shape `(num_nodes, hidden_channels)` instead of `(1, hidden_channels)`.

---

### 4 — Train source-only baseline

```python
from shiftkit.methods import SourceOnlyTrainer

so = SourceOnlyTrainer(model_so, train_src, train_tgt, lr=1e-3, device="cpu")
so.fit(epochs=30)
```

The source-only trainer minimises cross-entropy on **source train nodes** only.
It ignores the target graph during training, so any feature shift carries straight through to predictions.

---

### 5 — Train with MMD alignment

```python
from shiftkit.methods import MMDTrainer

mmd = MMDTrainer(model_mmd, train_src, train_tgt,
                 mmd_weight=0.5, lr=1e-3, device="cpu")
mmd.fit(epochs=30)
```

`MMDTrainer` adds a kernel MMD² penalty between source and target **node embeddings**
(train-split nodes from each domain), pulling the latent distributions closer together
while still optimising the classification loss on source nodes.

---

### 6 — Evaluate

```python
for name, trainer in [("Source-Only", so), ("MMD", mmd)]:
    r_src = trainer.evaluate(test_src, domain="source-test")
    r_tgt = trainer.evaluate(test_tgt, domain="target-test")
    print(
        f"{name:12s}  src acc={r_src['accuracy']*100:.1f}%  "
        f"tgt acc={r_tgt['accuracy']*100:.1f}%"
    )
```

---

## Typical results

| Method | Source accuracy | Target accuracy |
|--------|-----------------|-----------------|
| Source-only | ~98% | ~50–65% |
| MMD (λ=0.5) | ~95–98% | ~65–80% |

*(Results vary by random seed. The shift of 1.5 is large enough that source-only models
generalise poorly to the target; MMD alignment recovers a significant fraction of that gap.)*

---

## Tuning tips

| Symptom | Likely cause | Try |
|---------|-------------|-----|
| Target accuracy worse with MMD | λ too large | Reduce `mmd_weight` (try 0.1–0.3) |
| Source accuracy drops sharply | Alignment dominates cross-entropy | Add `warmup_epochs` to MMDTrainer |
| Both accuracies plateau early | Encoder capacity too low | Increase `hidden_channels` or `num_layers` |
| Unstable training | Graph too small for MMD estimate | Increase `n_nodes` or reduce `mmd_weight` |

---

## When to use node-level vs graph-level DA

Use **node-level** when:

- Your dataset is a single large graph per domain (citation network, social network, detector readout).
- Labels are assigned to **nodes**, not to whole graphs.
- Message passing needs access to the **full neighbourhood** during inference.

Use **graph-level** when:

- You have many independent small graphs per domain (molecule datasets, event graphs).
- Labels are assigned per graph.
- You use `DataManager` with `task_level="graph"` and `GNN(..., pool="mean")`.

---

## API reference

### `GNN` (node-level configuration)

```python
from shiftkit.models import GNN

model = GNN(
    data,                  # template PyG Data object (for num_node_features)
    model_name="SAGE",     # conv type: SAGE | GCN | GAT | GIN | GraphConv
    hidden_channels=32,    # width of each conv layer = latent dim
    num_layers=2,          # number of message-passing layers
    num_classes=3,         # output classes
    pool="none",           # REQUIRED for node-level: returns per-node embeddings
    dropout=0.0,
)
```

### `DataManager.load("pyg_domains", task_level="node", ...)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `Data` | Source domain graph |
| `target` | `Data` | Target domain graph |
| `task_level` | `str` | `"node"` for node-level DA |
| `train_ratio` | `float` | Fraction of nodes for training (default `0.6`) |
| `val_ratio` | `float` | Fraction of nodes for validation (default `0.2`) |
| `split_seed` | `int` | Random seed for reproducible splits |
| `split_mode` | `str` | `"stratified"` (by class) or `"random"` |
