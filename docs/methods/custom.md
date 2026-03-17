# Custom Methods

ShiftKit makes it easy to plug in your own domain adaptation method. Two tools are provided:

- **`BaseTrainer`** — abstract base class that defines the required interface
- **`TrainerRegistry`** — name-to-class registry so custom trainers can be stored, discovered, and instantiated by name

---

## BaseTrainer

Subclass `BaseTrainer` and implement two methods:

| Method | Signature | Description |
|--------|-----------|-------------|
| `fit` | `(epochs: int) → list[dict]` | Run training and return per-epoch history |
| `evaluate` | `(loader, domain: str) → dict` | Compute accuracy on a DataLoader |

### Minimal example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from shiftkit.methods import BaseTrainer

class FeatureAlignTrainer(BaseTrainer):
    """Custom trainer: CE + L2 distance between domain means."""

    def __init__(self, model, source_loader, target_loader,
                 align_weight=1.0, lr=1e-3, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.align_weight = align_weight
        self.ce = nn.CrossEntropyLoss()
        self.opt = optim.Adam(model.parameters(), lr=lr)
        self.history = []

    def fit(self, epochs=10):
        for epoch in range(1, epochs + 1):
            self.model.train()
            ce_sum = da_sum = 0.0
            n = 0
            for (x_src, y_src), (x_tgt, _) in zip(self.source_loader,
                                                    self.target_loader):
                x_src, y_src = x_src.to(self.device), y_src.to(self.device)
                x_tgt = x_tgt.to(self.device)

                z_src = self.model.encode(x_src)
                z_tgt = self.model.encode(x_tgt)

                ce   = self.ce(self.model.classify(z_src), y_src)
                da   = (z_src.mean(0) - z_tgt.mean(0)).pow(2).sum()
                loss = ce + self.align_weight * da

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                ce_sum += ce.item()
                da_sum += da.item()
                n += 1

            record = {
                "epoch": epoch,
                "ce_loss": ce_sum / n,
                "da_loss": da_sum / n,
                "total_loss": (ce_sum + self.align_weight * da_sum) / n,
                "src_acc": 0.0,   # omitted for brevity
                "tgt_acc": 0.0,
            }
            self.history.append(record)
            print(f"[{epoch:>3}/{epochs}] CE={record['ce_loss']:.4f}  "
                  f"DA={record['da_loss']:.4f}")
        return self.history

    def evaluate(self, loader, domain="source"):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                correct += (self.model(x).argmax(1) == y).sum().item()
                total   += y.size(0)
        return {"domain": domain, "accuracy": correct / total, "n_samples": total}
```

### History format

`fit()` should return `list[dict]` with at least these keys so it works with `plot_training_history`:

| Key | Type | Description |
|-----|------|-------------|
| `epoch` | `int` | Epoch index (1-based) |
| `ce_loss` | `float` | Cross-entropy loss |
| `total_loss` | `float` | Combined loss |
| `src_acc` | `float` | Source accuracy |
| `tgt_acc` | `float` | Target accuracy |

Add any extra keys you need (e.g. `da_loss`, `eta`, …).

---

## TrainerRegistry

Register a custom trainer under a string key so it can be discovered and instantiated by name.

### Decorator style

```python
from shiftkit.methods import TrainerRegistry, BaseTrainer

@TrainerRegistry.register("feature_align")
class FeatureAlignTrainer(BaseTrainer):
    ...
```

### Explicit registration

```python
TrainerRegistry.register("feature_align", FeatureAlignTrainer)
```

Both forms are equivalent. Use the decorator when defining the class; use the explicit call when registering a class defined elsewhere.

### Using the registry

```python
# List all available trainers
print(TrainerRegistry.available())
# ['dann', 'feature_align', 'mmd', 'sidda', 'source_only']

# Get the class
cls = TrainerRegistry.get("feature_align")

# Instantiate directly
trainer = TrainerRegistry.create(
    "feature_align",
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    align_weight=0.5,
)
history = trainer.fit(epochs=20)
```

### Sweeping over methods

The registry is useful for running experiments across multiple methods without hard-coding class names:

```python
from shiftkit.methods import TrainerRegistry

methods = ["source_only", "mmd", "dann", "feature_align"]
results = {}

for name in methods:
    trainer = TrainerRegistry.create(
        name,
        model=build_model(),
        source_loader=train_src,
        target_loader=train_tgt,
    )
    history = trainer.fit(epochs=20)
    results[name] = trainer.evaluate(test_tgt, domain="target-test")

for name, res in results.items():
    print(f"{name:>15}  target acc: {res['accuracy']*100:.1f}%")
```

---

## API reference

### `BaseTrainer`

```python
from shiftkit.methods import BaseTrainer
```

Abstract base class. Subclass and implement `fit()` and `evaluate()`.

### `TrainerRegistry`

```python
from shiftkit.methods import TrainerRegistry
```

| Method | Description |
|--------|-------------|
| `TrainerRegistry.register(name, cls=None)` | Register a trainer class (also works as a decorator) |
| `TrainerRegistry.available()` | Return sorted list of registered names |
| `TrainerRegistry.get(name)` | Return the class registered under *name* |
| `TrainerRegistry.create(name, **kwargs)` | Instantiate the trainer, forwarding kwargs |
