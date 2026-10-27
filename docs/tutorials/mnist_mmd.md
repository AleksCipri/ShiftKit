# Tutorial 1 — MNIST → Noisy MNIST: Source-Only vs MMD vs DANN

End-to-end walkthrough of the built-in benchmark: adapting a CNN from clean
MNIST digits (source) to the same digits corrupted with Gaussian noise (target).

The complete runnable script lives at `examples/mnist_mmd.py`.

---

## Run the script

```bash
python examples/mnist_mmd.py
```

The script is configured via a `CONFIG` block at the top. It seeds Python,
NumPy and PyTorch with `SEED` and prints the full configuration (and library
versions) at the start of the run, so the printed output records its settings:

```python
SEED          = 0
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
Test-split noise is fixed per image, so every evaluation sees the same corrupted
test set; training noise is redrawn on each access. The trainers use only the
target *images* — target labels are used only for monitoring and evaluation.

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

!!! warning "Illustrative output"
    The table below is illustrative output from a single unrecorded run of an
    earlier version of the script (no seed, configuration or environment was
    saved, and target test noise was then redrawn on every evaluation). It is
    not a benchmark and has not been reproduced. Your numbers will differ;
    repeat over several seeds before comparing methods.

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

The left panel shows the **cross-entropy (classification) loss** per epoch — not
the total loss including the MMD/DANN term; the right panel overlays source
(solid) and target (dashed) accuracy. These accuracies are measured on the
*training* batches during training (target labels are used only for this
monitoring), so the gap between them indicates, but does not precisely measure,
the remaining domain shift; use the held-out evaluation above for that.

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

Each model gets one row with two t-SNE panels of the same 2-D projection: the
**left panel is coloured by domain** (source vs target) and the **right panel by
class label**. Overlap of source and target points in the left panel suggests
more similar latent distributions; the projection is exploratory and does not by
itself show that target classes are correctly aligned.

---

## What to look for

- **Source-Only**: in the domain panel, source and target points often occupy
  partly different regions.
- **MMD / DANN**: alignment terms usually increase domain overlap in the projection.
  Whether this helps the target task must be checked with held-out target accuracy.
- The target-test accuracy difference relative to Source-Only, averaged over several
  seeds, is the measure of DA benefit; a single run is not.
