# Tutorial 2 — MMD vs LMMD comparison

Compares **global MMD** (aligns marginal distributions) against **local LMMD**
(aligns per-class subdomains) on the MNIST → Noisy MNIST benchmark.

The complete runnable script lives at `examples/mnist_mmd_lmmd.py`.

---

## Run the script

```bash
python examples/mnist_mmd_lmmd.py
```

Config block:

```python
LATENT_DIM   = 128
EPOCHS       = 15
BATCH_SIZE   = 128
LR           = 1e-3
MMD_WEIGHT   = 1.0
LMMD_WEIGHT  = 1.0
NUM_CLASSES  = 10
NOISE_STD    = 0.3
WARMUP       = 3      # source-only epochs before DA loss activates
```

---

## Key difference: global vs local alignment

**Global MMD** computes one kernel statistic over all source and target samples:

$$\widehat{\text{MMD}}^2(p, q) = \mathbb{E}[k(z^s, z^s)] - 2\,\mathbb{E}[k(z^s, z^t)] + \mathbb{E}[k(z^t, z^t)]$$

**Local LMMD** computes one kernel statistic *per class* and averages:

$$\hat{d}(p, q) = \frac{1}{C}\sum_{c=1}^{C}\left[\ldots k(z_i^s, z_j^s)\ldots - 2\ldots k(z_i^s, z_j^t)\ldots\right]$$

where source weights come from ground-truth labels and target weights from the
model's current softmax predictions.

The benefit is that global MMD can accidentally push *class 3 source* features
toward *class 8 target* features (negative transfer), whereas LMMD only aligns
class 3 to class 3.

---

## Code walkthrough

```python
from shiftkit.data    import DataManager
from shiftkit.models  import CNN
from shiftkit.methods import SourceOnlyTrainer, MMDTrainer, LMMDTrainer

dm = DataManager(root="./data", batch_size=128)
train_src, train_tgt = dm.load("mnist_noisy_mnist", train=True,  noise_std=0.3)
test_src,  test_tgt  = dm.load("mnist_noisy_mnist", train=False, noise_std=0.3)

model_baseline = CNN(latent_dim=128, num_classes=10, dropout=0.3)
model_mmd      = CNN(latent_dim=128, num_classes=10, dropout=0.3)
model_lmmd     = CNN(latent_dim=128, num_classes=10, dropout=0.3)

baseline = SourceOnlyTrainer(model_baseline, train_src, train_tgt, lr=1e-3)
mmd      = MMDTrainer(model_mmd,  train_src, train_tgt,
                      mmd_weight=1.0, lr=1e-3, warmup_epochs=3)
lmmd     = LMMDTrainer(model_lmmd, train_src, train_tgt,
                       num_classes=10, lmmd_weight=1.0, lr=1e-3, warmup_epochs=3)

history_baseline = baseline.fit(epochs=15)
history_mmd      = mmd.fit(epochs=15)
history_lmmd     = lmmd.fit(epochs=15)
```

!!! note "num_classes is required"
    `LMMDTrainer` requires `num_classes` so it can build the per-class weight
    matrices. It must match the number of output logits in your model.

---

## Outputs

```
  Domain               Source-Only         MMD        LMMD
  ---------------------------------------------------------
  source-train              99.x%       99.x%       99.x%
  source-test               98.x%       99.x%       99.x%
  target-test               95.x%       96.x%       97.x%
```

*(Exact numbers vary by run; LMMD typically matches or beats MMD on target.)*

The script saves two figures to `outputs/`:

- `mmd_lmmd_history.png` — training curves for all three methods
- `mmd_lmmd_latent.png` — t-SNE latent space side-by-side

---

## When to prefer LMMD over MMD

- When your task has **many classes** and class-level features differ substantially.
- When global MMD training shows good source accuracy but poor target accuracy
  (sign of negative transfer between classes).
- When you have reliable pseudo-labels — LMMD degrades if early predictions are
  very noisy; use `warmup_epochs` to train the encoder on source first.

!!! tip "Warmup recommendation"
    Set `warmup_epochs` to 5–10% of total epochs so the model can produce
    reasonable pseudo-labels before LMMD alignment begins.
