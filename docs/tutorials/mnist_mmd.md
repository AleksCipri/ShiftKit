# Tutorial 1 — MNIST → Noisy MNIST with MMD

End-to-end walkthrough of the built-in benchmark: adapting a CNN from clean
MNIST digits (source) to the same digits corrupted with Gaussian noise (target).

The complete runnable script lives at `examples/mnist_mmd.py`.

---

## Run the script

```bash
python examples/mnist_mmd.py
```

The script is configured via a `CONFIG` block at the top:

```python
MODEL_TYPE    = "cnn"   # "cnn" or "mlp"
LATENT_DIM    = 128
EPOCHS        = 10
BATCH_SIZE    = 128
LR            = 1e-3
MMD_WEIGHT    = 1.0     # λ
DOMAIN_WEIGHT = 1.0     # DANN domain loss weight
NOISE_STD     = 0.3     # Gaussian noise std on target
```

---

## Step-by-step breakdown

### Step 1 — Load data

```python
from shiftkit.data import DataManager

dm = DataManager(root="./data", batch_size=128, num_workers=0)
train_src, train_tgt = dm.load("mnist_noisy_mnist", train=True,  noise_std=0.3)
test_src,  test_tgt  = dm.load("mnist_noisy_mnist", train=False, noise_std=0.3)
```

`DataManager` returns paired `(source_loader, target_loader)` DataLoaders. The
target domain is MNIST with additive Gaussian noise; class labels are identical.

### Step 2 — Build independent models

Both use the same architecture so the comparison is fair — different random
initialisations ensure they train independently.

```python
from shiftkit.models import CNN

model_baseline = CNN(latent_dim=128, num_classes=10, dropout=0.3)
model_mmd      = CNN(latent_dim=128, num_classes=10, dropout=0.3)
model_dann     = CNN(latent_dim=128, num_classes=10, dropout=0.3)
```

### Step 3 — Train

```python
from shiftkit.methods import SourceOnlyTrainer, MMDTrainer, DANNTrainer

baseline = SourceOnlyTrainer(model_baseline, train_src, train_tgt, lr=1e-3)
mmd      = MMDTrainer(model_mmd, train_src, train_tgt, mmd_weight=1.0, lr=1e-3)
dann     = DANNTrainer(model_dann, train_src, train_tgt, domain_weight=1.0, lr=1e-3)

history_baseline = baseline.fit(epochs=10)
history_mmd      = mmd.fit(epochs=10)
history_dann     = dann.fit(epochs=10)
```

### Step 4 — Evaluate

```
  Domain               Source-Only         MMD        DANN
  ---------------------------------------------------------
  source-train              99.44%      99.58%      99.51%
  source-test               98.89%      99.14%      98.97%
  target-test               95.27%      96.80%      97.10%
```

### Step 5 — Plot training history

```python
from shiftkit.diagnostics import plot_training_history

plot_training_history(
    histories={
        "Source Only": history_baseline,
        "MMD":         history_mmd,
        "DANN":        history_dann,
    },
    save_path="outputs/training_history.png",
)
```

The left panel shows total loss per epoch; the right panel overlays source
(solid) and target (dashed) accuracy — the gap between them quantifies remaining
domain shift.

### Step 6 — Plot latent spaces

```python
from shiftkit.diagnostics import compare_latent_spaces

compare_latent_spaces(
    models={
        "Source Only": model_baseline,
        "MMD":         model_mmd,
        "DANN":        model_dann,
    },
    source_loader=test_src,
    target_loader=test_tgt,
    max_samples=2000,
    save_path="outputs/latent_space_comparison.png",
)
```

Each panel shows a t-SNE projection coloured by class (filled = source,
open = target). Better alignment means source and target clusters overlap.

---

## What to look for

- **Source-Only**: source clusters (filled) are tight; target clusters (open) are shifted.
- **MMD**: global distribution matching reduces the gap.
- **DANN**: adversarial training typically produces even tighter alignment.
- Target accuracy improvement over Source-Only quantifies the DA benefit.
