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
| `target_loader` | `DataLoader` | — | Target DataLoader yielding `x` or `(x, y)`; labels, if present, are used only for `tgt_acc` |
| `mmd_weight` | `float` | `1.0` | λ — weight on the MMD regularisation term |
| `lr` | `float` | `1e-3` | Adam learning rate |
| `warmup_epochs` | `int` | `0` | Epochs of source-only CE pre-training before MMD DA begins |
| `device` | `str \| None` | `None` | `'cuda'`, `'mps'`, or `'cpu'`; auto-detected if `None` |
| `mmd_sigmas` | `list[float] \| "median" \| None` | `None` | RBF kernel bandwidths; defaults to `[0.1, 1, 5, 10, 50]` (see [MMDLoss](#mmdloss)) |

---

## `fit(epochs=10)`

Train for `epochs` epochs and return the history. One epoch covers the
**longer** of the two loaders; the shorter one is restarted as needed. Empty
loaders raise `ValueError`. During MMD epochs, tensor source and target
batches are encoded in **one joint forward pass** (concatenated, then split),
so BatchNorm layers normalise both domains with shared batch statistics.
Node-level graph batches are encoded per domain.

**Returns:** `list[dict]` — one dict per epoch:

| Key | Description |
|-----|-------------|
| `epoch` | Epoch number (1-indexed) |
| `ce_loss` | Mean cross-entropy loss |
| `mmd_loss` | Mean MMD² loss (`0.0` during warmup) |
| `total_loss` | Mean total loss (CE + λ·MMD²; CE only during warmup) |
| `src_acc` | Source training accuracy: training-mode predictions on each batch, taken *before* that batch's update |
| `tgt_acc` | Target accuracy in evaluation mode (no BatchNorm update, no dropout), after each update; `nan` if target batches have no labels. Diagnostic only |

Trainers do not seed random number generators; call `torch.manual_seed(...)`
before building loaders/models for repeatable runs.

---

## `evaluate(loader, domain="source")`

Compute classification accuracy on any labelled DataLoader, in evaluation
mode. Accepts tensor batches, PyG graph batches and node-level graph batches.
An empty or unlabelled loader raises `ValueError`.

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

For each bandwidth σ, with source latents $x_1,\dots,x_n$ and target latents
$y_1,\dots,y_m$, the **default (biased V-statistic)** estimate is

$$\widehat{\text{MMD}}^2_{b,\sigma} = \frac{1}{n^2}\sum_{i,j} k_\sigma(x_i,x_j) + \frac{1}{m^2}\sum_{i,j} k_\sigma(y_i,y_j) - \frac{2}{nm}\sum_{i,j} k_\sigma(x_i,y_j)$$

It includes the $i=j$ terms, is always $\ge 0$ (up to rounding), and is
biased upward for finite samples. With `unbiased=True` the within-domain
terms exclude the diagonal (U-statistic, Gretton et al., 2012):

$$\widehat{\text{MMD}}^2_{u,\sigma} = \frac{1}{n(n-1)}\sum_{i\ne j} k_\sigma(x_i,x_j) + \frac{1}{m(m-1)}\sum_{i\ne j} k_\sigma(y_i,y_j) - \frac{2}{nm}\sum_{i,j} k_\sigma(x_i,y_j)$$

which can be negative and needs $n, m \ge 2$. The returned loss is the
**sum** over bandwidths, $\sum_\sigma \widehat{\text{MMD}}^2_\sigma$ (not an
average), where $k_\sigma(x, y) = \exp\!\left(-\|x-y\|^2 / 2\sigma^2\right)$.
Squared distances are computed directly from differences, so they are never
negative even for large-norm latents.

Example: source `[0, 1]`, target `[2, 3]`, σ = 1 gives 1.162376 (biased) and
0.768906 (unbiased).

**Bandwidths.** Bandwidths are fixed: the defaults `[0.1, 1, 5, 10, 50]`, a
list you supply, or opt-in `sigmas="median"`, which uses a single bandwidth
equal to the median pairwise distance of the pooled source+target batch
(recomputed every call, not differentiated). No other automatic selection is
performed; the defaults are scale-dependent, so check them against the typical
distances in your latent space.

```python
from shiftkit.methods import MMDLoss

mmd = MMDLoss(sigmas=[0.1, 1.0, 5.0, 10.0, 50.0])
loss = mmd(z_source, z_target)   # scalar tensor

mmd_u = MMDLoss(sigmas="median", unbiased=True)
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sigmas` | `list[float] \| "median" \| None` | `None` | Kernel bandwidths; defaults to `[0.1, 1.0, 5.0, 10.0, 50.0]` |
| `unbiased` | `bool` | `False` | Use the U-statistic instead of the biased V-statistic |

### `forward(source, target)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `Tensor (n, d)` | Latent vectors from source domain |
| `target` | `Tensor (m, d)` | Latent vectors from target domain |

**Returns:** Scalar MMD² estimate (summed over bandwidths).
