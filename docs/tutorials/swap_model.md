# Tutorial 3 — Swapping the model

All ShiftKit trainers work with any model that exposes `.encode(x)` and
`.classify(z)`. This tutorial covers three scenarios: switching from CNN to MLP,
using the built-in graph model `SimpleGCN`, and writing a completely custom
architecture.

---

## CNN → MLP

Set `MODEL_TYPE = "mlp"` in the `examples/mnist_mmd.py` config, or directly:

```python
from shiftkit.models  import MLP
from shiftkit.methods import MMDTrainer

model = MLP(latent_dim=128, num_classes=10, hidden_dims=(512, 256), dropout=0.3)
trainer = MMDTrainer(model, train_src, train_tgt, mmd_weight=1.0)
trainer.fit(epochs=10)
```

`MLP` flattens the input automatically, so it works on the same MNIST loaders
without any data changes.

---

## Graph data with SimpleGCN

For the built-in graph benchmark (`"synthetic_graphs"`):

```python
from shiftkit.data    import DataManager
from shiftkit.models  import SimpleGCN
from shiftkit.methods import LMMDTrainer

dm = DataManager(batch_size=32)
train_src, train_tgt = dm.load("synthetic_graphs", train=True)
test_src,  test_tgt  = dm.load("synthetic_graphs", train=False)

model = SimpleGCN(n_nodes=10, feat_dim=4, latent_dim=64, num_classes=2)

trainer = LMMDTrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    num_classes=2,
    lmmd_weight=1.0,
    warmup_epochs=5,
)
history = trainer.fit(epochs=30)
```

Each graph sample is a packed tensor `(N, N+feat_dim)` — no PyTorch Geometric
required.  See [SimpleGCN](../models.md#simplegcn) for architecture details.

---

## Custom architecture

Any `nn.Module` with `.encode()`, `.classify()`, and `.forward()` works:

```python
import torch.nn as nn
from shiftkit.methods import MMDTrainer

class ResidualEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.bottleneck = nn.Linear(256, latent_dim)
        self.head = nn.Linear(latent_dim, num_classes)
        self.act  = nn.ReLU()

    def encode(self, x):
        h = self.act(self.fc1(x.view(x.size(0), -1)))
        h = self.act(self.fc2(h) + h)   # residual connection
        return self.act(self.bottleneck(h))

    def classify(self, z):
        return self.head(z)

    def forward(self, x):
        return self.classify(self.encode(x))

model = ResidualEncoder(input_dim=784, latent_dim=64, num_classes=10)
trainer = MMDTrainer(model, train_src, train_tgt, mmd_weight=0.5)
trainer.fit(epochs=10)
```

!!! note
    For `SIDDATrainer`, the model must also expose a `latent_dim` attribute
    (used to set up the learnable η parameters).
