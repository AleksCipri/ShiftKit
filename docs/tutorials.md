# Tutorials & Examples

---

## Tutorial 1 — MNIST → Noisy MNIST with MMD

This tutorial walks through the built-in end-to-end experiment: adapting a CNN from clean MNIST digits (source) to the same digits corrupted with Gaussian noise (target).

### Run the script

```bash
python examples/mnist_mmd.py
```

The script is configured via a `CONFIG` block at the top:

```python
MODEL_TYPE  = "cnn"    # "cnn" or "mlp"
LATENT_DIM  = 128
EPOCHS      = 10
BATCH_SIZE  = 128
LR          = 1e-3
MMD_WEIGHT  = 1.0      # λ
NOISE_STD   = 0.3      # Gaussian noise std on target
```

### What the script does

**Step 1 — Load data**

```python
dm = DataManager(root="./data", batch_size=BATCH_SIZE, num_workers=0)
train_src, train_tgt = dm.load("mnist_noisy_mnist", train=True,  noise_std=NOISE_STD)
test_src,  test_tgt  = dm.load("mnist_noisy_mnist", train=False, noise_std=NOISE_STD)
```

**Step 2 — Build two independent models**

Both use the same architecture so the comparison is fair — different random initialisations ensure they train independently.

```python
model_baseline = CNN(latent_dim=128, num_classes=10)
model_mmd      = CNN(latent_dim=128, num_classes=10)
```

**Step 3 — Train**

```python
baseline = SourceOnlyTrainer(model_baseline, train_src, train_tgt, lr=LR)
mmd      = MMDTrainer(model_mmd, train_src, train_tgt,
                       mmd_weight=MMD_WEIGHT, lr=LR)

history_baseline = baseline.fit(epochs=EPOCHS)
history_mmd      = mmd.fit(epochs=EPOCHS)
```

**Step 4 — Evaluate**

```
  Domain               Source-Only         MMD
  --------------------------------------------
  source-train              99.44%      99.58%
  source-test               98.89%      99.14%
  target-test               95.27%      94.50%
```

**Step 5 — Plot**

```python
plot_training_history({"Source Only": history_baseline, "MMD": history_mmd},
                      save_path="outputs/training_history.png")

compare_latent_spaces({"Source Only": model_baseline, "MMD": model_mmd},
                      test_src, test_tgt,
                      save_path="outputs/latent_space_comparison.png")
```

Outputs are saved to `outputs/`.

---

## Tutorial 2 — Swapping the model

To use an MLP instead of a CNN, change one line in the `CONFIG` block:

```python
MODEL_TYPE = "mlp"
```

Or directly in Python:

```python
from shiftkit.models import MLP

model = MLP(latent_dim=128, num_classes=10, hidden_dims=(512, 256), dropout=0.3)
trainer = MMDTrainer(model, train_src, train_tgt, mmd_weight=1.0)
trainer.fit(epochs=10)
```

---

## Tutorial 3 — Registering a custom dataset pair

You can register any pair of PyTorch datasets without modifying the library.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from shiftkit.data import DataManager

def my_factory(root, batch_size, train, num_workers, **kwargs):
    # Replace with your actual dataset loading logic
    X_src = torch.randn(1000, 1, 28, 28)
    y_src = torch.randint(0, 10, (1000,))
    X_tgt = torch.randn(1000, 1, 28, 28) + 0.5   # shifted distribution

    source_ds = TensorDataset(X_src, y_src)
    target_ds = TensorDataset(X_tgt, torch.zeros(1000, dtype=torch.long))

    return (
        DataLoader(source_ds, batch_size=batch_size, shuffle=train),
        DataLoader(target_ds, batch_size=batch_size, shuffle=train),
    )

DataManager.register("my_custom_pair", my_factory)

dm = DataManager(batch_size=64)
train_src, train_tgt = dm.load("my_custom_pair", train=True)
```

---

## Tutorial 4 — Using a custom model

Any model that exposes `.encode(x)` and `.classify(z)` works with all trainers.

```python
import torch.nn as nn
from shiftkit.methods import MMDTrainer

class MyEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super().__init__()
        self.encoder    = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim), nn.ReLU(),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        return self.encoder(x.view(x.size(0), -1))

    def classify(self, z):
        return self.classifier(z)

    def forward(self, x):
        return self.classify(self.encode(x))

model = MyEncoder(input_dim=784, latent_dim=64, num_classes=10)
trainer = MMDTrainer(model, train_src, train_tgt, mmd_weight=0.5)
trainer.fit(epochs=5)
```

---

## Tutorial 5 — Tuning MMD weight λ

The `mmd_weight` (λ) controls the trade-off between source classification accuracy and domain alignment. A quick sweep:

```python
from shiftkit.models import CNN
from shiftkit.methods import MMDTrainer

results = {}
for lam in [0.1, 0.5, 1.0, 2.0, 5.0]:
    model   = CNN(latent_dim=128, num_classes=10)
    trainer = MMDTrainer(model, train_src, train_tgt,
                         mmd_weight=lam, lr=1e-3)
    trainer.fit(epochs=10)
    stats = trainer.evaluate(test_tgt, domain="target-test")
    results[lam] = stats["accuracy"]
    print(f"λ={lam:<5}  target acc={stats['accuracy']*100:.2f}%")
```

!!! tip
    Start at `λ=1.0`. If source accuracy degrades sharply, reduce λ. If the source/target latent spaces remain separated after training, increase λ.
