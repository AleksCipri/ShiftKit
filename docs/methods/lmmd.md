# LMMDTrainer

**LMMD** (Local Maximum Mean Discrepancy) extends global MMD by aligning
**per-class subdomains** instead of the overall marginal distributions.
Whereas standard MMD matches `p(z_src)` to `p(z_tgt)` globally, LMMD matches
`p(z_src | y=c)` to `p(z_tgt | y=c)` for every class `c` — preventing
semantically different classes from being forced together during alignment.

$$\mathcal{L} = \underbrace{\text{CrossEntropy}(\hat{y}_\text{src}, y_\text{src})}_{\text{supervised}} + \lambda \cdot \underbrace{\hat{d}(p, q)}_{\text{LMMD}}$$

$$\hat{d}(p, q) = \frac{1}{C}\sum_{c=1}^{C}\left[\sum_{i,j} w_i^{(s,c)} w_j^{(s,c)} k(z_i^s, z_j^s) + \sum_{i,j} w_i^{(t,c)} w_j^{(t,c)} k(z_i^t, z_j^t) - 2\sum_{i,j} w_i^{(s,c)} w_j^{(t,c)} k(z_i^s, z_j^t)\right]$$

**Class-conditional weights:**

$$w_i^{(s,c)} = \frac{y_{ic}}{\sum_j y_{jc}}, \qquad w_i^{(t,c)} = \frac{\hat{p}_{ic}}{\sum_j \hat{p}_{jc}}$$

- **Source** weights are derived from one-hot ground-truth labels `y_src`.
- **Target** weights are derived from the model's current softmax predictions (treated as soft pseudo-labels; stop-gradient).

> **Reference:** Zhu, Y., Zhuang, F., Wang, J., Ke, G., Chen, J., Bian, J., Xiong, H., & He, Q. (2021).
> Deep Subdomain Adaptation Network for Image Classification.
> *IEEE Transactions on Neural Networks and Learning Systems*, 32(4), 1713–1722.
> [[Paper]](https://ieeexplore.ieee.org/document/9085896)

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
| `target_loader` | `DataLoader` | — | Target DataLoader (labels used for `tgt_acc` tracking only) |
| `num_classes` | `int` | — | Number of output classes C |
| `lmmd_weight` | `float` | `1.0` | λ — weight on the LMMD regularisation term |
| `lr` | `float` | `1e-3` | Adam learning rate |
| `warmup_epochs` | `int` | `0` | Epochs of source-only CE pre-training before LMMD begins |
| `device` | `str \| None` | `None` | `'cuda'`, `'mps'`, or `'cpu'`; auto-detected if `None` |
| `lmmd_sigmas` | `list[float] \| None` | `None` | RBF kernel bandwidths; defaults to `[0.1, 1, 5, 10, 50]` |

---

## `fit(epochs=10)`

Train for `epochs` epochs and return the history.

**Returns:** `list[dict]` — one dict per epoch:

| Key | Description |
|-----|-------------|
| `epoch` | Epoch number (1-indexed) |
| `ce_loss` | Mean cross-entropy loss |
| `lmmd_loss` | Mean LMMD loss (`0.0` during warmup) |
| `mmd_loss` | Always `0.0` (history-format compatibility) |
| `total_loss` | Mean total loss (CE + λ·LMMD; CE only during warmup) |
| `src_acc` | Source domain training accuracy |
| `tgt_acc` | Target domain accuracy (tracked, not optimised directly) |

---

## `evaluate(loader, domain="source")`

Compute classification accuracy on any labelled DataLoader.

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

LMMD typically outperforms global MMD when the source and target class structures are well-separated, because it avoids matching source class A features to target class B features.

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

### `forward(z_src, z_tgt, y_src, y_tgt_prob)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `z_src` | `Tensor (n_s, d)` | Source latent vectors |
| `z_tgt` | `Tensor (n_t, d)` | Target latent vectors |
| `y_src` | `Tensor (n_s,)` | Source ground-truth labels (long) |
| `y_tgt_prob` | `Tensor (n_t, C)` | Target softmax probabilities |

**Returns:** Scalar LMMD loss averaged over C classes.