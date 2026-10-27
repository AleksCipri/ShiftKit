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

We create two synthetic graphs, each with 300 nodes, 8 node features and 3 classes,
using **one shared labelling rule**:

- each node gets a class label (balanced across the 3 classes);
- its features are drawn around a **class-specific mean** (`class_sep=1.5` from the
  origin) plus unit Gaussian noise, so features carry class information;
- edges are **homophilous**: same-class pairs connect with probability `p_in=0.05`,
  different-class pairs with `p_out=0.005` (about 80% of edges join same-class nodes).

```
source: x = mean[y] + N(0, I)           shift = 0.0
target: x = mean[y] + N(0, I) + 1.5     shift = 1.5   (same class means, same edge rule)
```

The labelling rule is identical in both domains; only the target features are
translated by 1.5 in every dimension (a covariate shift). A model trained only on
the source graph sees shifted inputs on the target.

MMD alignment encourages the GNN encoder to map both graphs' node embeddings into a
shared latent space. Matching marginal embedding distributions does not by itself
guarantee that classes are matched correctly; it can help here because the shift is a
simple translation shared by all classes.

---

## Run the script

```bash
pip install torch-geometric   # one-time; already included if you used requirements.txt
python examples/pyg_node_mmd.py
```

The script runs on CPU in a few seconds, seeds Python, NumPy and PyTorch, and
prints its full configuration as JSON on the first line.

---

## Code walkthrough

### 1 — Build the domain graphs

```python
import torch
from torch_geometric.data import Data

def make_class_means(n_classes, feat_dim, class_sep, seed):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n_classes, feat_dim, generator=g)
    return class_sep * means / means.norm(dim=1, keepdim=True)

def make_domain_graph(n_nodes, class_means, seed, shift=0.0, p_in=0.05, p_out=0.005):
    g = torch.Generator().manual_seed(seed)
    n_classes, feat_dim = class_means.shape
    y = torch.arange(n_nodes) % n_classes                      # balanced labels
    y = y[torch.randperm(n_nodes, generator=g)]
    x = class_means[y] + torch.randn(n_nodes, feat_dim, generator=g) + shift

    same = y.view(-1, 1) == y.view(1, -1)                      # homophilous edges
    prob = torch.where(same, torch.tensor(p_in), torch.tensor(p_out))
    upper = torch.triu(torch.rand(n_nodes, n_nodes, generator=g) < prob, diagonal=1)
    src, dst = upper.nonzero(as_tuple=True)
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    return Data(x=x, edge_index=edge_index, y=y)

means = make_class_means(3, feat_dim=8, class_sep=1.5, seed=0)  # shared by both domains
source_graph = make_domain_graph(300, means, seed=1, shift=0.0)
target_graph = make_domain_graph(300, means, seed=2, shift=1.5)
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

!!! note "Masks are stored on the `Data` objects"
    The first `load` call writes `train_mask`, `val_mask` and `test_mask` onto each
    graph (source uses `split_seed`, target `split_seed + 1`). Later calls with the
    same split parameters reuse them; calls with different parameters recompute them.
    Masks you set yourself are never overwritten.

!!! note "Validation nodes"
    `train=True` yields train nodes, `train=False` yields test nodes. The `val_mask`
    nodes are not used for training; load them with `split="val"` for model
    selection, e.g. `val_src, val_tgt = dm.load("pyg_domains", split="val", ...)`.
    Using target **labels** on validation nodes is only legitimate if your setting
    allows a small labelled target set (see [Tuning λ](tuning.md)).

---

### 3 — Build the GNN with `pool="none"`

```python
from shiftkit.models import GNN

model_so  = GNN(source_graph, "SAGE", hidden_channels=32, num_layers=2,
                num_classes=3, pool="none", device="cpu")
model_mmd = GNN(source_graph, "SAGE", hidden_channels=32, num_layers=2,
                num_classes=3, pool="none", device="cpu")
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

## Example results (one seed)

Output of `python examples/pyg_node_mmd.py` (seed 0, configuration as printed by the
script, CPU, torch 2.12 / torch-geometric 2.8.1), test nodes:

| Method | Source test accuracy | Target test accuracy |
|--------|-----------------|-----------------|
| Source-only | 78.3% | 33.3% |
| MMD (λ=0.5) | 86.7% | 51.7% |

These are numbers from **one seed** on 60 test nodes per domain (chance ≈ 33%); they
are not a benchmark and will change with the seed, the shift, `class_sep` and the
trainer implementation. The source-only target accuracy (33.3%) is at chance level. A linear classifier on raw source features reaches ≈70% on held-out source
nodes, confirming the labels are learnable. Repeat over several seeds before drawing
conclusions about MMD.

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
| `val_ratio` | `float` | Fraction of nodes for validation (default `0.2`); load with `split="val"` |
| `split_seed` | `int` | Random seed for reproducible splits |
| `split_mode` | `str` | `"stratified"` (by class) or `"random"` |
| `split` | `str \| None` | `"train"`, `"val"` or `"test"`; overrides `train` (node-level only) |
