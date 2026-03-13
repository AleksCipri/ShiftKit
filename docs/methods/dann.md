# DANNTrainer

Trains a model using adversarial domain adaptation. A domain discriminator is
attached to the encoder output via a **Gradient Reversal Layer (GRL)**. During
backpropagation the GRL negates the discriminator's gradients, forcing the encoder
to learn domain-invariant representations that fool the discriminator.

$$\mathcal{L} = \underbrace{\text{CrossEntropy}(\hat{y}_\text{src}, y_\text{src})}_{\text{task}} + \lambda \cdot \underbrace{\text{BCE}(\hat{d},\, d_\text{label})}_{\text{domain (via GRL)}}$$

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
| `domain_weight` | `float` | `1.0` | λ — weight on the adversarial domain loss |
| `lr` | `float` | `1e-3` | Adam learning rate (shared by model + discriminator) |
| `alpha` | `float` | `1.0` | Final GRL reversal strength |
| `schedule_alpha` | `bool` | `True` | Ramp α from 0→`alpha` using the paper's sigmoid schedule |
| `discriminator_hidden` | `int` | `128` | Hidden dim of the domain discriminator MLP |
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
| `mmd_loss` | Always `0.0` (for history-format compatibility) |
| `total_loss` | Mean total loss (CE + λ·Domain) |
| `src_acc` | Source domain training accuracy |
| `tgt_acc` | Target domain accuracy (tracked, not directly optimised) |

---

## `evaluate(loader, domain="source")`

Compute classification accuracy on any labelled DataLoader.

**Returns:** `dict` with keys `domain` (str), `accuracy` (float), `n_samples` (int).

---

## Alpha scheduling

!!! note
    The original paper ramps the GRL strength as:

    $$\alpha(p) = \alpha_\text{max} \cdot \left(\frac{2}{1 + e^{-10p}} - 1\right), \quad p = \frac{\text{epoch}}{\text{epochs}}$$

    This avoids large adversarial gradients early in training when encoder
    representations are still noisy. Set `schedule_alpha=False` to use a fixed
    reversal strength instead.

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
