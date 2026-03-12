# ShiftKit

A lightweight, modular **domain adaptation** framework for PyTorch. Transfer knowledge from a labelled *source* domain to an unlabelled *target* domain using deep latent-space alignment.

![Framework Overview](assets/framework_overview.png)

---

## Installation

```bash
git clone https://github.com/AleksCipri/ShiftKit.git
cd ShiftKit
pip install -r requirements.txt
```

**Dependencies:** `torch`, `torchvision`, `numpy`, `matplotlib`, `scikit-learn`, `tqdm`

---

## Quick start

```python
from shiftkit.data        import DataManager
from shiftkit.models      import CNN
from shiftkit.methods     import MMDTrainer, SourceOnlyTrainer
from shiftkit.diagnostics import compare_latent_spaces, plot_training_history

# 1. Load data
dm = DataManager(batch_size=128)
train_src, train_tgt = dm.load("mnist_noisy_mnist", train=True)
test_src,  test_tgt  = dm.load("mnist_noisy_mnist", train=False)

# 2. Build models
model_baseline = CNN(latent_dim=128, num_classes=10)
model_mmd      = CNN(latent_dim=128, num_classes=10)

# 3. Train — baseline vs MMD domain adaptation
baseline = SourceOnlyTrainer(model_baseline, train_src, train_tgt)
mmd      = MMDTrainer(model_mmd, train_src, train_tgt, mmd_weight=1.0)

history_baseline = baseline.fit(epochs=10)
history_mmd      = mmd.fit(epochs=10)

# 4. Visualise
plot_training_history({"Source Only": history_baseline, "MMD": history_mmd})
compare_latent_spaces({"Source Only": model_baseline, "MMD": model_mmd},
                      test_src, test_tgt)
```

---

## Data

`shiftkit.data` handles loading source/target dataset pairs and exposes a registry for custom datasets.

### DataManager

The central hub for all data loading. Call `load()` with a dataset key and get back paired DataLoaders for the source and target domains.

```python
from shiftkit.data import DataManager

dm = DataManager(root="./data", batch_size=64, num_workers=0)

# Load built-in MNIST → Noisy MNIST pair
train_src, train_tgt = dm.load("mnist_noisy_mnist", train=True)
test_src,  test_tgt  = dm.load("mnist_noisy_mnist", train=False)

# Pass keyword args to the factory (e.g. noise level)
train_src, train_tgt = dm.load("mnist_noisy_mnist", train=True, noise_std=0.5)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `root` | `"./data"` | Root directory for downloaded datasets |
| `batch_size` | `64` | Batch size for both loaders |
| `num_workers` | `0` | DataLoader worker processes |

**Registering a custom dataset pair:**

```python
def my_factory(root, batch_size, train, num_workers, **kwargs):
    source_ds = ...   # your source torch Dataset
    target_ds = ...   # your target torch Dataset
    return (
        DataLoader(source_ds, batch_size=batch_size, shuffle=train),
        DataLoader(target_ds, batch_size=batch_size, shuffle=train),
    )

DataManager.register("my_source_target", my_factory)
src, tgt = DataManager().load("my_source_target")
```

### NoisyMNIST

Wraps `torchvision.MNIST` and injects per-sample Gaussian noise on each access — a simple synthetic target domain.

```python
from shiftkit.data.datasets import NoisyMNIST

ds = NoisyMNIST(root="./data", train=True, noise_std=0.3)
img, label = ds[0]   # tensor in [0, 1] with noise applied
```

Noise is applied as `img = (img + N(0, σ²)).clamp(0, 1)` at read time, so every epoch sees different noise realisations.

### Built-in dataset pairs

| Key | Source | Target |
|-----|--------|--------|
| `"mnist_noisy_mnist"` | MNIST | NoisyMNIST (σ=0.3 by default) |

---

## Models

`shiftkit.models` provides neural network architectures with a shared **encoder / classifier** interface. This split is what allows domain adaptation methods to operate directly in the latent space.

| Method | Description |
|--------|-------------|
| `model.encode(x)` | Map input → latent vector `z ∈ ℝᵈ` |
| `model.classify(z)` | Map latent vector → class logits |
| `model(x)` | `classify(encode(x))` — standard `nn.Module` interface |

### CNN

Two convolutional blocks followed by a fully-connected bottleneck. Designed for **1×28×28 inputs**.

```
Input (1×28×28)
  → Conv(1→32) + BN + ReLU + MaxPool   →  32×14×14
  → Conv(32→64) + BN + ReLU + MaxPool  →  64×7×7
  → Flatten → Linear(3136→256) → Dropout
  → Linear(256→latent_dim)              →  z ∈ ℝᵈ
  → Linear(latent_dim→num_classes)      →  logits
```

```python
from shiftkit.models import CNN

model = CNN(latent_dim=128, num_classes=10, dropout=0.3)
z      = model.encode(x)    # (B, 128)
logits = model.classify(z)  # (B, 10)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | `128` | Bottleneck embedding size |
| `num_classes` | `10` | Number of output classes |
| `dropout` | `0.3` | Dropout probability |

### MLP

Fully-connected network, suitable for any flattened input.

```
Input (flattened)
  → Linear(in→h₁) + ReLU + Dropout
  → Linear(h₁→h₂) + ReLU + Dropout
  → Linear(hₙ→latent_dim)   →  z ∈ ℝᵈ
  → Linear(latent_dim→num_classes)  →  logits
```

```python
from shiftkit.models import MLP

model = MLP(latent_dim=128, num_classes=10, hidden_dims=(512, 256), dropout=0.3)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | `128` | Bottleneck embedding size |
| `num_classes` | `10` | Number of output classes |
| `hidden_dims` | `(512, 256)` | Hidden layer sizes before bottleneck |
| `dropout` | `0.3` | Dropout probability per layer |

### Custom models

Any model with `.encode(x)` and `.classify(z)` methods works with all trainers:

```python
class MyModel(nn.Module):
    def encode(self, x):   return self.encoder(x)
    def classify(self, z): return self.head(z)
    def forward(self, x):  return self.classify(self.encode(x))
```

---

## Domain Adaptation Methods

`shiftkit.methods` provides training loops. Both trainers record identical per-epoch history dicts so their results can be compared directly.

### MMDTrainer

Minimises a combined supervised + domain alignment loss:

$$\mathcal{L} = \text{CrossEntropy}(\hat{y}_\text{src}, y_\text{src}) + \lambda \cdot \widehat{\text{MMD}}^2(z_\text{src}, z_\text{tgt})$$

The MMD term uses a **mixture of RBF kernels** to capture domain discrepancy at multiple scales.

```python
from shiftkit.methods import MMDTrainer

trainer = MMDTrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    mmd_weight=1.0,    # λ
    lr=1e-3,
)
history = trainer.fit(epochs=10)

result = trainer.evaluate(test_tgt, domain="target-test")
print(f"Target acc: {result['accuracy']*100:.1f}%")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mmd_weight` | `1.0` | λ — balance between classification and alignment |
| `lr` | `1e-3` | Adam learning rate |
| `device` | auto | `'cuda'`, `'mps'`, or `'cpu'` |
| `mmd_sigmas` | `[0.1,1,5,10,50]` | RBF kernel bandwidths |

### SourceOnlyTrainer

A **no-DA baseline** — cross-entropy on source data only. Use this to quantify how much benefit domain adaptation provides. Produces the same history format as `MMDTrainer` (with `mmd_loss = 0`).

```python
from shiftkit.methods import SourceOnlyTrainer

baseline = SourceOnlyTrainer(model, train_src, train_tgt, lr=1e-3)
history  = baseline.fit(epochs=10)
```

### Training history format

Both trainers return a `list[dict]` with one entry per epoch:

| Key | Description |
|-----|-------------|
| `epoch` | Epoch index (1-based) |
| `ce_loss` | Cross-entropy loss |
| `mmd_loss` | MMD² loss (`0.0` for `SourceOnlyTrainer`) |
| `total_loss` | Total loss |
| `src_acc` | Source accuracy |
| `tgt_acc` | Target accuracy (tracked but not directly optimised) |

### MMDLoss

The raw MMD² module for use in custom training loops:

```python
from shiftkit.methods import MMDLoss

mmd = MMDLoss(sigmas=[0.1, 1.0, 5.0, 10.0, 50.0])
loss = mmd(z_source, z_target)   # scalar tensor
```

---

## Diagnostics

`shiftkit.diagnostics` provides visualisation tools for latent spaces and training dynamics.

### plot_training_history

Plots CE loss (left) and source/target accuracy (right). Accepts multiple histories for overlay comparison.

```python
from shiftkit.diagnostics import plot_training_history

plot_training_history(
    histories={"Source Only": history_baseline, "MMD": history_mmd},
    save_path="outputs/training_history.png",
)
```

![Training History](assets/training_history.png)

### plot_latent_space

Encodes samples from both domains, projects to 2-D with t-SNE, and plots by domain (left) and by class (right).

```python
from shiftkit.diagnostics import plot_latent_space

plot_latent_space(model, test_src, test_tgt, max_samples=2000,
                  save_path="outputs/latent_space.png")
```

### compare_latent_spaces

Side-by-side grid comparing multiple models — one row per model, two panels per row.

```python
from shiftkit.diagnostics import compare_latent_spaces

compare_latent_spaces(
    models={"Source Only": model_baseline, "MMD": model_mmd},
    source_loader=test_src,
    target_loader=test_tgt,
    save_path="outputs/latent_space_comparison.png",
)
```

![Latent Space Comparison](assets/latent_space_comparison.png)

!!! tip "Interpreting the domain panel"
    A well-adapted model shows source and target points **interleaved** rather than in separate clusters — the encoder has learned to ignore domain-specific variation while preserving class structure.
