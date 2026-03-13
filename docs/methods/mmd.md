# MMDTrainer

Trains a model by minimising a combined supervised + domain alignment loss:

$$\mathcal{L} = \underbrace{\text{CrossEntropy}(\hat{y}_\text{src}, y_\text{src})}_{\text{supervised}} + \lambda \cdot \underbrace{\widehat{\text{MMD}}^2(z_\text{src}, z_\text{tgt})}_{\text{domain alignment}}$$

The classifier head is supervised on source labels only. The encoder is pulled
toward domain-invariant representations by minimising the **Maximum Mean
Discrepancy (MMD)** between source and target latent vectors.

> **Reference:** Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
> A Kernel Two-Sample Test. *Journal of Machine Learning Research*, 13, 723–773.
> [[PDF]](https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf)

---

## Usage

```python
from shiftkit.methods import MMDTrainer

trainer = MMDTrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    mmd_weight=1.0,
    lr=1e-3,
)
history = trainer.fit(epochs=10)

result = trainer.evaluate(test_tgt, domain="target-test")
print(f"Target accuracy: {result['accuracy']*100:.1f}%")
```

---

## Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | Network with `.encode()` and `.classify()` methods |
| `source_loader` | `DataLoader` | — | Labelled source DataLoader |
| `target_loader` | `DataLoader` | — | Target DataLoader (labels used for `tgt_acc` tracking only) |
| `mmd_weight` | `float` | `1.0` | λ — weight on the MMD regularisation term |
| `lr` | `float` | `1e-3` | Adam learning rate |
| `warmup_epochs` | `int` | `0` | Epochs of source-only CE pre-training before MMD DA begins |
| `device` | `str \| None` | `None` | `'cuda'`, `'mps'`, or `'cpu'`; auto-detected if `None` |
| `mmd_sigmas` | `list[float] \| None` | `None` | RBF kernel bandwidths; defaults to `[0.1, 1, 5, 10, 50]` |

---

## `fit(epochs=10)`

Train for `epochs` epochs and return the history.

**Returns:** `list[dict]` — one dict per epoch:

| Key | Description |
|-----|-------------|
| `epoch` | Epoch number (1-indexed) |
| `ce_loss` | Mean cross-entropy loss |
| `mmd_loss` | Mean MMD² loss (`0.0` during warmup) |
| `total_loss` | Mean total loss (CE + λ·MMD²; CE only during warmup) |
| `src_acc` | Source domain training accuracy |
| `tgt_acc` | Target domain accuracy (tracked, not optimised directly) |

---

## `evaluate(loader, domain="source")`

Compute classification accuracy on any labelled DataLoader.

**Returns:** `dict` with keys `domain` (str), `accuracy` (float), `n_samples` (int).

---

## Warmup phase

Set `warmup_epochs > 0` to train the encoder on source classification only before
MMD alignment begins. This ensures the latent space carries meaningful class
structure before the MMD term tries to align the distributions.

```python
trainer = MMDTrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    mmd_weight=1.0,
    warmup_epochs=5,    # first 5 epochs: CE only
    lr=1e-3,
)
history = trainer.fit(epochs=20)
# Epochs 1–5:  [warmup]  CE only
# Epochs 6–20: [MMD  ]   CE + λ·MMD²
```

---

## MMDLoss

The raw MMD² loss module, exposed for use in custom training loops.

$$\widehat{\text{MMD}}^2(P, Q) = \sum_{\sigma} \left[ \mathbb{E}[k_\sigma(x,x')] - 2\,\mathbb{E}[k_\sigma(x,y)] + \mathbb{E}[k_\sigma(y,y')] \right]$$

where $k_\sigma(x, y) = \exp\!\left(-\|x-y\|^2 / 2\sigma^2\right)$ is the RBF kernel.
Summing over multiple bandwidths σ captures domain discrepancy at different scales.

```python
from shiftkit.methods import MMDLoss

mmd = MMDLoss(sigmas=[0.1, 1.0, 5.0, 10.0, 50.0])
loss = mmd(z_source, z_target)   # scalar tensor
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sigmas` | `list[float] \| None` | `None` | Kernel bandwidths; defaults to `[0.1, 1.0, 5.0, 10.0, 50.0]` |

### `forward(source, target)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `Tensor (n, d)` | Latent vectors from source domain |
| `target` | `Tensor (m, d)` | Latent vectors from target domain |

**Returns:** Scalar MMD² estimate.
