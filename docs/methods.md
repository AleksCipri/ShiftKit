# Methods

`shiftkit.methods` provides domain adaptation training loops. Both trainers record identical per-epoch history dicts so their results can be directly compared.

---

## MMDTrainer

Trains a model by minimising a combined loss:

$$\mathcal{L} = \underbrace{\text{CrossEntropy}(\hat{y}_\text{src}, y_\text{src})}_{\text{supervised}} + \lambda \cdot \underbrace{\widehat{\text{MMD}}^2(z_\text{src}, z_\text{tgt})}_{\text{domain alignment}}$$

The classifier head is only supervised on source labels. The encoder is pulled toward domain-invariant representations by minimising the MMD between source and target latent vectors.

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

## Comparing methods

```python
from shiftkit.diagnostics import plot_training_history

plot_training_history({
    "Source Only": history_baseline,
    "MMD":         history_mmd,
})
```

The right panel shows source accuracy (solid lines) and target accuracy (dashed lines) for each method — the gap between them quantifies the domain shift.

!!! tip "Tuning λ"
    Start with `mmd_weight=1.0`. If the model collapses (target acc drops sharply), reduce to `0.1`–`0.5`. If source and target distributions are very different, increasing to `2.0`–`5.0` may help alignment.
