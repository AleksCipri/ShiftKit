# Methods

`shiftkit.methods` provides domain adaptation training loops. All trainers record identical per-epoch history dicts so their results can be directly compared.

| Trainer | DA mechanism | Key parameter |
|---------|-------------|---------------|
| `SourceOnlyTrainer` | None (baseline) | — |
| `MMDTrainer` | Latent distribution matching | `mmd_weight` λ |
| `DANNTrainer` | Adversarial domain discriminator | `domain_weight` λ |

---

## MMDTrainer

Trains a model by minimising a combined loss:

$$\mathcal{L} = \underbrace{\text{CrossEntropy}(\hat{y}_\text{src}, y_\text{src})}_{\text{supervised}} + \lambda \cdot \underbrace{\widehat{\text{MMD}}^2(z_\text{src}, z_\text{tgt})}_{\text{domain alignment}}$$

The classifier head is only supervised on source labels. The encoder is pulled toward domain-invariant representations by minimising the MMD between source and target latent vectors.

> **Reference:** Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A Kernel Two-Sample Test. *Journal of Machine Learning Research*, 13, 723–773. [[PDF]](https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf)

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

# evaluate on any labelled loader
stats = trainer.evaluate(test_tgt, domain="target-test")
print(f"Target accuracy: {stats['accuracy']*100:.1f}%")
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | Network with `.encode()` and `.classify()` methods |
| `source_loader` | `DataLoader` | — | Labelled source DataLoader |
| `target_loader` | `DataLoader` | — | Target DataLoader (labels used for accuracy tracking only) |
| `mmd_weight` | `float` | `1.0` | λ — weight on the MMD regularisation term |
| `lr` | `float` | `1e-3` | Adam learning rate |
| `device` | `str \| None` | `None` | `'cuda'`, `'mps'`, or `'cpu'`; auto-detected if `None` |
| `mmd_sigmas` | `list[float] \| None` | `None` | RBF kernel bandwidths; defaults to `[0.1, 1, 5, 10, 50]` |

### `fit(epochs=10)`

Train for `epochs` epochs and return the history.

**Returns:** `list[dict]` — one dict per epoch with keys:

| Key | Description |
|-----|-------------|
| `epoch` | Epoch number (1-indexed) |
| `ce_loss` | Mean cross-entropy loss |
| `mmd_loss` | Mean MMD² loss |
| `total_loss` | Mean total loss (CE + λ·MMD²) |
| `src_acc` | Source domain training accuracy |
| `tgt_acc` | Target domain accuracy (tracked, not optimised directly) |

### `evaluate(loader, domain="source")`

Compute accuracy on any labelled DataLoader.

**Returns:** `dict` with keys `domain`, `accuracy` (float), `n_samples` (int).

---

## SourceOnlyTrainer

A **no-adaptation baseline** that trains only on labelled source data with cross-entropy loss. Produces the same history format as `MMDTrainer` (`mmd_loss` is always `0.0`) for direct comparison.

```python
from shiftkit.methods import SourceOnlyTrainer

baseline = SourceOnlyTrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,   # used for tgt_acc tracking only
    lr=1e-3,
)
history = baseline.fit(epochs=10)
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | Network with standard `forward()` method |
| `source_loader` | `DataLoader` | — | Labelled source DataLoader |
| `target_loader` | `DataLoader` | — | Target DataLoader (labels used for accuracy tracking only) |
| `lr` | `float` | `1e-3` | Adam learning rate |
| `device` | `str \| None` | `None` | `'cuda'`, `'mps'`, or `'cpu'`; auto-detected if `None` |

`fit()` and `evaluate()` have the same signatures as `MMDTrainer`.

---

## MMDLoss

The raw MMD² loss module, exposed for use in custom training loops.

$$\widehat{\text{MMD}}^2(P, Q) = \sum_{\sigma} \left[ \mathbb{E}[k_\sigma(x,x')] - 2\,\mathbb{E}[k_\sigma(x,y)] + \mathbb{E}[k_\sigma(y,y')] \right]$$

where $k_\sigma(x, y) = \exp\!\left(-\|x-y\|^2 / 2\sigma^2\right)$ is the RBF kernel and the sum runs over a mixture of bandwidths.

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

---

---

## DANNTrainer

Trains a model using adversarial domain adaptation. A domain discriminator
is attached to the encoder output through a **Gradient Reversal Layer (GRL)**.
During backpropagation the GRL negates the discriminator's gradients, forcing
the encoder to produce representations that fool the discriminator — i.e.
domain-invariant features.

$$\mathcal{L} = \underbrace{\text{CrossEntropy}(\hat{y}_\text{src}, y_\text{src})}_{\text{task}} + \lambda \cdot \underbrace{\text{BCE}(\hat{d}, d_\text{label})}_{\text{domain (via GRL)}}$$

```
encoder(x) ──► z ──► classify(z) ──► CE loss
                └──► GRL ──► discriminator(z) ──► BCE loss
                        ↑ gradients negated here
```

> **Reference:** Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., & Lempitsky, V. (2016). Domain-Adversarial Training of Neural Networks. *Journal of Machine Learning Research*, 17(59), 1–35. [[PDF]](https://jmlr.org/papers/volume17/15-239/15-239.pdf)

```python
from shiftkit.methods import DANNTrainer

trainer = DANNTrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    domain_weight=1.0,
    lr=1e-3,
    alpha=1.0,
    schedule_alpha=True,   # ramp α from 0→1 over training (recommended)
)
history = trainer.fit(epochs=10)
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | Network with `.encode()`, `.classify()`, and `.latent_dim` |
| `source_loader` | `DataLoader` | — | Labelled source DataLoader |
| `target_loader` | `DataLoader` | — | Target DataLoader (labels used for tracking only) |
| `domain_weight` | `float` | `1.0` | λ — weight on the domain adversarial loss |
| `lr` | `float` | `1e-3` | Adam learning rate (shared by model + discriminator) |
| `alpha` | `float` | `1.0` | GRL reversal strength at the end of training |
| `schedule_alpha` | `bool` | `True` | Ramp α from 0→`alpha` using the original paper's schedule |
| `discriminator_hidden` | `int` | `128` | Hidden dim of the domain discriminator MLP |
| `device` | `str \| None` | `None` | `'cuda'`, `'mps'`, or `'cpu'`; auto-detected if `None` |

### `fit(epochs=10)`

**Returns:** `list[dict]` — one dict per epoch with keys:

| Key | Description |
|-----|-------------|
| `epoch` | Epoch number (1-indexed) |
| `ce_loss` | Mean cross-entropy loss |
| `domain_loss` | Mean domain discriminator loss |
| `total_loss` | Mean total loss (CE + λ·Domain) |
| `src_acc` | Source domain training accuracy |
| `tgt_acc` | Target domain accuracy (tracked, not directly optimised) |

`evaluate()` has the same signature as `MMDTrainer`.

!!! note "Alpha scheduling"
    The original paper ramps the GRL strength using
    $\alpha(p) = \alpha_\text{max} \cdot \left(\frac{2}{1 + e^{-10p}} - 1\right)$
    where $p = \text{epoch}/\text{epochs}$.
    This avoids large adversarial gradients early in training when the
    encoder representations are still noisy. Set `schedule_alpha=False` to
    use a fixed reversal strength instead.

---

## Comparing methods

```python
from shiftkit.diagnostics import plot_training_history

plot_training_history({
    "Source Only": history_baseline,
    "MMD":         history_mmd,
    "DANN":        history_dann,
})
```

The right panel shows source accuracy (solid lines) and target accuracy (dashed lines) for each method — the gap between them quantifies the domain shift.

!!! tip "Tuning λ"
    Both `MMDTrainer` and `DANNTrainer` accept a `domain_weight` / `mmd_weight` parameter λ.
    Start at `1.0`. Reduce to `0.1`–`0.5` if source accuracy degrades; increase to `2.0`–`5.0` if source and target embeddings remain separated after training.
