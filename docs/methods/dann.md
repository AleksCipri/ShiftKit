# DANNTrainer

Trains a model using adversarial domain adaptation. A domain discriminator is
attached to the encoder output via a **Gradient Reversal Layer (GRL)**. During
backpropagation the GRL negates the discriminator's gradients, forcing the encoder
to learn domain-invariant representations that fool the discriminator.

$$\mathcal{L} = \underbrace{\text{CrossEntropy}(\hat{y}_\text{src}, y_\text{src})}_{\text{task}} + \lambda_d \cdot \underbrace{\text{BCE}(\hat{d},\, d_\text{label})}_{\text{domain (via GRL)}}$$

```
encoder(x) ──► z ──► classify(z) ──► CE loss
                └──► GRL ──► discriminator(z) ──► BCE loss
                        ↑ gradients negated here
```

> **Reference:** Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H.,
> Laviolette, F., Marchand, M., & Lempitsky, V. (2016).
> Domain-Adversarial Training of Neural Networks.
> *Journal of Machine Learning Research*, 17(59), 1–35.
> [[PDF]](https://jmlr.org/papers/volume17/15-239/15-239.pdf)

---

## Usage

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

result = trainer.evaluate(test_tgt, domain="target-test")
print(f"Target accuracy: {result['accuracy']*100:.1f}%")
```

---

## Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | Network with `.encode()`, `.classify()`, and `.latent_dim` |
| `source_loader` | `DataLoader` | — | Labelled source DataLoader |
| `target_loader` | `DataLoader` | — | Target DataLoader (labels used for `tgt_acc` tracking only) |
| `domain_weight` | `float` | `1.0` | λ_d — weight on the adversarial domain loss |
| `lr` | `float` | `1e-3` | Adam learning rate (shared by model + discriminator) |
| `alpha` | `float` | `1.0` | Final GRL reversal strength |
| `schedule_alpha` | `bool` | `True` | Ramp α from 0→`alpha` using the paper's sigmoid schedule (counts only over the DA phase) |
| `discriminator_hidden` | `int` | `128` | Hidden dim of the domain discriminator MLP |
| `warmup_epochs` | `int` | `0` | Epochs of source-only CE pre-training before adversarial DA begins; α is held at 0 during warmup |
| `semantic_weight` | `float` | `0.0` | λ_s — weight on the centroid alignment loss; `0.0` disables it |
| `centroid_momentum` | `float` | `0.1` | EMA momentum β for updating target centroids |
| `num_classes` | `int` | `10` | Number of classes (required when `semantic_weight > 0`) |
| `device` | `str \| None` | `None` | `'cuda'`, `'mps'`, or `'cpu'`; auto-detected if `None` |

---

## `fit(epochs=10)`

Train for `epochs` epochs and return the history.

**Returns:** `list[dict]` — one dict per epoch:

| Key | Description |
|-----|-------------|
| `epoch` | Epoch number (1-indexed) |
| `ce_loss` | Mean cross-entropy loss |
| `domain_loss` | Mean domain discriminator loss |
| `semantic_loss` | Mean centroid alignment loss (`0.0` when disabled) |
| `mmd_loss` | Always `0.0` (for history-format compatibility) |
| `total_loss` | Mean total loss |
| `src_acc` | Source domain training accuracy |
| `tgt_acc` | Target domain accuracy (tracked, not directly optimised) |

---

## `evaluate(loader, domain="source")`

Compute classification accuracy on any labelled DataLoader.

**Returns:** `dict` with keys `domain` (str), `accuracy` (float), `n_samples` (int).

---

## Warmup phase

Set `warmup_epochs > 0` to run source-only CE training before the adversarial
loop begins. During warmup the GRL strength α is held at 0 and the discriminator
receives no useful gradients, so neither is updated. The α schedule (if enabled)
only starts ramping **after** warmup ends, over the remaining DA epochs.

```python
trainer = DANNTrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    domain_weight=1.0,
    warmup_epochs=5,     # first 5 epochs: CE only, α=0
    schedule_alpha=True, # then ramp α from 0→1 over epochs 6–N
    lr=1e-3,
)
history = trainer.fit(epochs=20)
# Epochs 1–5:  [warmup]  CE only
# Epochs 6–20: [DANN ]   CE + λ·BCE_GRL  (α ramping 0→1)
```

---

## Alpha scheduling

!!! note
    The original paper ramps the GRL strength as:

    $$\alpha(p) = \alpha_\text{max} \cdot \left(\frac{2}{1 + e^{-10p}} - 1\right), \quad p = \frac{\text{epoch}}{\text{epochs}}$$

    This avoids large adversarial gradients early in training when encoder
    representations are still noisy. Set `schedule_alpha=False` to use a fixed
    reversal strength instead.

---

## Semantic Centroid Alignment

Enable by setting `semantic_weight > 0`. This adds a class-level alignment term
from **MSTN** (Xie et al., 2018) on top of the adversarial loss:

$$\mathcal{L} = \text{CE} + \lambda_d \cdot \text{BCE}_\text{GRL} + \lambda_s \cdot \frac{1}{K}\sum_{k=1}^{K} \left\| \mathbf{c}_k^\text{src} - \mathbf{c}_k^\text{tgt} \right\|^2$$

where source centroids are computed per-batch from ground-truth labels and target
centroids are maintained as an **exponential moving average** (EMA) of
pseudo-labeled target features:

$$\mathbf{c}_k^\text{tgt} \leftarrow (1 - \beta)\,\mathbf{c}_k^\text{tgt} + \beta \cdot \overline{z}_k^\text{tgt}$$

The EMA stabilises alignment when early pseudo-labels are noisy, since target
centroids evolve smoothly rather than jumping each batch.

```python
trainer = DANNTrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    domain_weight=1.0,
    semantic_weight=0.5,     # enable centroid alignment
    centroid_momentum=0.1,   # β — how fast centroids track the current batch
    num_classes=10,
    lr=1e-3,
)
history = trainer.fit(epochs=10)
```

!!! tip "Tuning tips"
    - Start with `semantic_weight` in `[0.1, 1.0]` and `centroid_momentum` in `[0.05, 0.2]`.
    - Lower `centroid_momentum` (e.g. `0.05`) gives smoother centroid updates, which
      helps early in training when pseudo-labels are unreliable.
    - Centroid alignment complements the adversarial loss: the GRL aligns global
      distributions while centroid alignment enforces class-level correspondence.

> **Reference:** Xie, S., Zheng, Z., Chen, L., & Chen, C. (2018).
> Learning Semantic Representations for Unsupervised Domain Adaptation.
> *ICML 2018*, PMLR 80:5423–5432.
> [[Paper]](https://proceedings.mlr.press/v80/xie18c.html)

---

## GradientReversalLayer

The GRL is also exposed as a standalone module for use in custom training loops:

```python
from shiftkit.methods import GradientReversalLayer

grl = GradientReversalLayer(alpha=1.0)
z_reversed = grl(z)   # identity forward, negated gradient backward
```

## DomainDiscriminator

The domain discriminator is also available standalone:

```python
from shiftkit.methods import DomainDiscriminator

discriminator = DomainDiscriminator(latent_dim=128, hidden_dim=128)
logits = discriminator(z)   # (B, 1) — source=0, target=1
```
