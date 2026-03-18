# CORALTrainer

**CORAL** (CORrelation ALignment) aligns the **second-order statistics** (covariance matrices) of source and target feature distributions. Rather than matching distribution means or using a kernel, it minimises the squared Frobenius norm between source and target covariances in the latent space — making it computationally lightweight and kernel-free.

$$\mathcal{L} = \underbrace{\text{CrossEntropy}(\hat{y}_\text{src}, y_\text{src})}_{\text{supervised}} + \lambda \cdot \underbrace{\frac{1}{4d^2}\|C_S - C_T\|^2_F}_{\text{CORAL}}$$

where the covariance matrices are:

$$C_S = \frac{1}{n_S - 1}\left(Z_S^\top Z_S - \frac{1}{n_S}(\mathbf{1}^\top Z_S)^\top(\mathbf{1}^\top Z_S)\right)$$

and $C_T$ is defined analogously, $d$ is the latent dimensionality, and $\|\cdot\|_F^2$ is the squared Frobenius norm.

The $4d^2$ normalisation ensures the loss is scale-invariant with respect to the number of features.

> **Reference:** Sun, B., & Saenko, K. (2016).
> Deep CORAL: Correlation Alignment for Deep Domain Adaptation.
> *ECCV Workshops 2016*, LNCS 9915, 443–450.
> [[Paper]](https://arxiv.org/abs/1607.01719)

---

## Usage

```python
from shiftkit.methods import CORALTrainer

trainer = CORALTrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    coral_weight=1.0,
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
| `coral_weight` | `float` | `1.0` | λ — weight on the CORAL regularisation term |
| `lr` | `float` | `1e-3` | Adam learning rate |
| `warmup_epochs` | `int` | `0` | Epochs of source-only CE pre-training before CORAL begins |
| `device` | `str \| None` | `None` | `'cuda'`, `'mps'`, or `'cpu'`; auto-detected if `None` |

---

## `fit(epochs=10)`

Train for `epochs` epochs and return the history.

**Returns:** `list[dict]` — one dict per epoch:

| Key | Description |
|-----|-------------|
| `epoch` | Epoch number (1-indexed) |
| `ce_loss` | Mean cross-entropy loss |
| `coral_loss` | Mean CORAL loss (`0.0` during warmup) |
| `mmd_loss` | Always `0.0` (history-format compatibility) |
| `total_loss` | Mean total loss (CE + λ·CORAL; CE only during warmup) |
| `src_acc` | Source domain training accuracy |
| `tgt_acc` | Target domain accuracy (tracked, not optimised directly) |

---

## `evaluate(loader, domain="source")`

Compute classification accuracy on any labelled DataLoader.

**Returns:** `dict` with keys `domain` (str), `accuracy` (float), `n_samples` (int).

---

## CORAL vs MMD

| | MMD | CORAL |
|--|-----|-------|
| What is matched | Full distribution (via kernel) | Covariance (second-order statistics) |
| Kernel required | Yes — RBF, bandwidth σ to choose | No — purely algebraic |
| Computation per batch | O(n²) kernel matrices | O(n·d²) covariance matrices |
| Captures | Arbitrary distributional differences | Mean (implicitly via centring) + variance/correlation |
| Best suited for | General distribution shift | Shift in feature scale/correlation structure |

CORAL is a strong, fast baseline when the domain shift is primarily in the spread or correlation of features rather than in higher-order structure.

---

## CORALLoss

The raw CORAL loss module, exposed for use in custom training loops.

```python
from shiftkit.methods import CORALLoss

coral = CORALLoss()
loss = coral(z_source, z_target)   # scalar tensor
```

The loss is parameter-free — no constructor arguments needed.

### `forward(source, target)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `Tensor (n_s, d)` | Source latent vectors |
| `target` | `Tensor (n_t, d)` | Target latent vectors |

**Returns:** Scalar CORAL loss `(1/4d²) ‖C_S − C_T‖²_F`.
