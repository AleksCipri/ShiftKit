# LMMDTrainer

**LMMD** (Local Maximum Mean Discrepancy) extends global MMD by aligning
**per-class subdomains** instead of the overall marginal distributions.
Whereas standard MMD matches `p(z_src)` to `p(z_tgt)` globally, LMMD matches
`p(z_src | y=c)` to an estimate of `p(z_tgt | y=c)` for each class `c`. The
target side of each class is weighted by the model's own soft predictions, so
this *reduces the incentive* to pull different classes together only to the
extent those pseudo-labels are reliable; it does not prevent wrong-class
alignment.

$$\mathcal{L} = \underbrace{\text{CrossEntropy}(\hat{y}_\text{src}, y_\text{src})}_{\text{supervised}} + \lambda \cdot \underbrace{\hat{d}(p, q)}_{\text{LMMD}}$$

$$\hat{d}(p, q) = \frac{1}{|A|}\sum_{c\in A}\sum_{\sigma}\left[\sum_{i,j} w_i^{(s,c)} w_j^{(s,c)} k_\sigma(z_i^s, z_j^s) + \sum_{i,j} w_i^{(t,c)} w_j^{(t,c)} k_\sigma(z_i^t, z_j^t) - 2\sum_{i,j} w_i^{(s,c)} w_j^{(t,c)} k_\sigma(z_i^s, z_j^t)\right]$$

**Class-conditional weights:**

$$w_i^{(s,c)} = \frac{y_{ic}}{\sum_j y_{jc}}, \qquad w_i^{(t,c)} = \frac{\hat{p}_{ic}}{\sum_j \hat{p}_{jc}}$$

- **Source** weights are derived from one-hot ground-truth labels `y_src`.
- **Target** weights are derived from the model's current softmax predictions (treated as soft pseudo-labels; stop-gradient).

Kernels are **summed** over the bandwidths σ, and the result is averaged over
the set $A$ of *active* classes.

**Missing-class policy.** Within a batch, class $c$ is active only if it
appears in the source batch **and** its total target probability
$\sum_j \hat{p}_{jc}$ exceeds `min_target_mass` (default `1e-3`). Other
classes have no well-defined normalised source or target measure and are
skipped; if no class is active the loss is 0. Consequently identical source
and target latents give zero loss even when some classes are absent from the
source batch, and a class with negligible target mass is not inflated to full
weight by normalisation.

**Relationship to DSAN.** `LMMDTrainer` is a DSAN-style variant, not an exact
reproduction of the paper's training recipe: it uses a fixed RBF kernel
mixture, the missing-class policy above, a constant `lmmd_weight` (no warm-up
ramp of the trade-off), and a constant Adam learning rate.

> **Reference:** Zhu, Y., Zhuang, F., Wang, J., Ke, G., Chen, J., Bian, J., Xiong, H., & He, Q. (2021).
> Deep Subdomain Adaptation Network for Image Classification.
> *IEEE Transactions on Neural Networks and Learning Systems*, 32(4), 1713–1722.
> [[Paper]](https://ieeexplore.ieee.org/document/9085896) [[arXiv]](https://arxiv.org/abs/2106.09388)

---

## Usage

```python
from shiftkit.methods import LMMDTrainer

trainer = LMMDTrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    num_classes=10,
    lmmd_weight=1.0,
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
| `num_classes` | `int` | — | Number of output classes C |
| `lmmd_weight` | `float` | `1.0` | λ — weight on the LMMD regularisation term |
| `lr` | `float` | `1e-3` | Adam learning rate |
| `warmup_epochs` | `int` | `0` | Epochs of source-only CE pre-training before LMMD begins |
| `device` | `str \| None` | `None` | `'cuda'`, `'mps'`, or `'cpu'`; auto-detected if `None` |
| `lmmd_sigmas` | `list[float] \| None` | `None` | RBF kernel bandwidths; defaults to `[0.1, 1, 5, 10, 50]` |

---

## `fit(epochs=10)`

Train for `epochs` epochs and return the history. One epoch covers the
**longer** of the two loaders; the shorter one is restarted as needed. Empty
loaders raise `ValueError`. During LMMD epochs, tensor source and target
batches are encoded in one joint forward pass, so BatchNorm layers use shared
batch statistics for both domains. Trainers do not seed random number
generators.

**Returns:** `list[dict]` — one dict per epoch:

| Key | Description |
|-----|-------------|
| `epoch` | Epoch number (1-indexed) |
| `ce_loss` | Mean cross-entropy loss |
| `lmmd_loss` | Mean LMMD loss (`0.0` during warmup) |
| `mmd_loss` | Always `0.0` (history-format compatibility) |
| `total_loss` | Mean total loss (CE + λ·LMMD; CE only during warmup) |
| `src_acc` | Source training accuracy: training-mode predictions on each batch, taken *before* that batch's update |
| `tgt_acc` | Target accuracy in evaluation mode, after each update; `nan` without target labels (diagnostic only) |

---

## `evaluate(loader, domain="source")`

Compute classification accuracy on any labelled DataLoader, in evaluation
mode. Accepts tensor batches, PyG graph batches and node-level graph batches.
An empty or unlabelled loader raises `ValueError`.

**Returns:** `dict` with keys `domain` (str), `accuracy` (float), `n_samples` (int).

---

## MMD vs LMMD

| | MMD | LMMD |
|--|-----|------|
| What is aligned | Global marginals `p(z)` | Per-class conditionals `p(z \| y=c)` |
| Source supervision | Not used in DA loss | One-hot labels weight the kernel |
| Target labels | Not needed | Soft pseudo-labels from model predictions |
| Risk | Negative transfer if class structure differs | Requires reasonable pseudo-labels |
| `num_classes` parameter | Not required | Required |

LMMD can outperform global MMD when class structure is well separated and the
target pseudo-labels are reasonably accurate, because it discourages matching
source class A features to target features predicted as class B. With poor
pseudo-labels it can reinforce early mistakes; no general advantage is
guaranteed.

---

## LMMDLoss

The raw LMMD module, exposed for use in custom training loops.

```python
from shiftkit.methods import LMMDLoss

lmmd = LMMDLoss(num_classes=10, sigmas=[0.1, 1.0, 5.0, 10.0, 50.0])

# y_tgt_prob: softmax predictions for target batch, shape (n_t, C)
with torch.no_grad():
    y_tgt_prob = torch.softmax(model.classify(model.encode(x_tgt)), dim=1)

loss = lmmd(z_src, z_tgt, y_src, y_tgt_prob)   # scalar tensor
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | `int` | — | Number of classes C |
| `sigmas` | `list[float] \| None` | `None` | Kernel bandwidths; defaults to `[0.1, 1.0, 5.0, 10.0, 50.0]` |
| `min_target_mass` | `float` | `1e-3` | Classes whose summed target probability in the batch is ≤ this value are skipped |

### `forward(z_src, z_tgt, y_src, y_tgt_prob)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `z_src` | `Tensor (n_s, d)` | Source latent vectors |
| `z_tgt` | `Tensor (n_t, d)` | Target latent vectors |
| `y_src` | `Tensor (n_s,)` | Source ground-truth labels (long) |
| `y_tgt_prob` | `Tensor (n_t, C)` | Target softmax probabilities |

**Returns:** Scalar LMMD loss: kernel terms summed over bandwidths and
averaged over the active classes (0 if none). Source labels outside
`[0, num_classes)` raise `ValueError`.