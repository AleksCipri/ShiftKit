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
    schedule_alpha=True,   # ramp α per epoch over the DA epochs
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
| `target_loader` | `DataLoader` | — | Target DataLoader yielding `x` or `(x, y)`; labels, if present, are used only for `tgt_acc` |
| `domain_weight` | `float` | `1.0` | λ_d — weight on the adversarial domain loss |
| `lr` | `float` | `1e-3` | Adam learning rate (shared by model + discriminator) |
| `alpha` | `float` | `1.0` | GRL reversal strength: final value when scheduled, constant value for every DA epoch otherwise |
| `schedule_alpha` | `bool` | `True` | Ramp α per epoch with a sigmoid over the DA epochs (variant of the paper's schedule, see below); `False` uses α = `alpha` after warmup |
| `discriminator_hidden` | `int` | `128` | Hidden dim of the domain discriminator MLP |
| `warmup_epochs` | `int` | `0` | Epochs of source-only CE pre-training before adversarial DA begins; α is held at 0 during warmup |
| `semantic_weight` | `float` | `0.0` | λ_s — weight on the centroid alignment loss; `0.0` disables it |
| `centroid_momentum` | `float` | `0.1` | EMA momentum β for updating target centroids |
| `num_classes` | `int` | `10` | Number of classes (required when `semantic_weight > 0`; source labels must be `< num_classes`, otherwise `ValueError`) |
| `device` | `str \| None` | `None` | `'cuda'`, `'mps'`, or `'cpu'`; auto-detected if `None` |

---

## `fit(epochs=10)`

Train for `epochs` epochs and return the history. One epoch covers the
**longer** of the two loaders; the shorter one is restarted as needed. Empty
loaders raise `ValueError`. During DA epochs, tensor source and target
batches are encoded in one joint forward pass, so BatchNorm layers use shared
batch statistics for both domains. Trainers do not seed random number
generators.

**Returns:** `list[dict]` — one dict per epoch:

| Key | Description |
|-----|-------------|
| `epoch` | Epoch number (1-indexed) |
| `ce_loss` | Mean cross-entropy loss |
| `domain_loss` | Mean domain discriminator loss |
| `semantic_loss` | Mean centroid alignment loss (`0.0` when disabled) |
| `mmd_loss` | Always `0.0` (for history-format compatibility) |
| `total_loss` | Mean total loss |
| `src_acc` | Source training accuracy: training-mode predictions on each batch, taken *before* that batch's update |
| `tgt_acc` | Target accuracy in evaluation mode, after each update; `nan` without target labels (diagnostic only) |

---

## `evaluate(loader, domain="source")`

Compute classification accuracy on any labelled DataLoader, in evaluation
mode. Accepts tensor batches, PyG graph batches and node-level graph batches.
An empty or unlabelled loader raises `ValueError`.

**Returns:** `dict` with keys `domain` (str), `accuracy` (float), `n_samples` (int).

---

## Warmup phase

Set `warmup_epochs > 0` to run source-only CE training before the adversarial
loop begins. During warmup the discriminator and GRL are not used (α is set
to 0). The α schedule (if enabled) only starts ramping **after** warmup ends,
over the remaining DA epochs; with `schedule_alpha=False`, α is set to `alpha`
for every epoch after warmup.

```python
trainer = DANNTrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    domain_weight=1.0,
    warmup_epochs=5,     # first 5 epochs: CE only, α=0
    schedule_alpha=True, # then ramp α toward 1 over epochs 6–20
    lr=1e-3,
)
history = trainer.fit(epochs=20)
# Epochs 1–5:  [warmup]  CE only
# Epochs 6–20: [DANN ]   CE + λ·BCE_GRL  (α ≈ 0.32 at epoch 6 → ≈ 1 at epoch 20)
```

---

## Alpha scheduling

!!! note
    ShiftKit sets the GRL strength once per epoch:

    $$\alpha(p) = \alpha_\text{max} \cdot \left(\frac{2}{1 + e^{-10p}} - 1\right), \quad p = \frac{\text{epoch} - \text{warmup}}{\text{epochs} - \text{warmup}}$$

    where warmup is `warmup_epochs`, so $p$ counts DA epochs only and reaches
    1 in the last epoch. Ganin et al. use the same sigmoid but with $p$ the
    overall training progress updated every iteration, together with an
    annealed learning rate. ShiftKit's schedule is therefore an epoch-level
    **variant**, and the Adam learning rate stays constant (no annealing). Set
    `schedule_alpha=False` to use the fixed strength `alpha` for all DA epochs.

---

## Semantic Centroid Alignment

Enable by setting `semantic_weight > 0`. This adds a class-level alignment term
inspired by **MSTN** (Xie et al., 2018) on top of the adversarial loss:

$$\mathcal{L} = \text{CE} + \lambda_d \cdot \text{BCE}_\text{GRL} + \lambda_s \cdot \frac{1}{|K|}\sum_{k\in K} \frac{1}{d}\left\| \mathbf{c}_k^\text{src} - \mathbf{c}_k^\text{tgt} \right\|^2$$

where $d$ is `latent_dim` (the code averages the squared difference over latent
dimensions) and $K$ is the set of classes that appear in the current source
batch **and** whose target centroid has already been initialised. Source
centroids are computed per batch from ground-truth labels; target centroids are
maintained as an **exponential moving average** (EMA) of pseudo-labeled target
features, initialised from the first batch in which a class is predicted:

$$\mathbf{c}_k^\text{tgt} \leftarrow (1 - \beta)\,\mathbf{c}_k^\text{tgt} + \beta \cdot \overline{z}_k^\text{tgt}$$

The EMA smooths the target centroids when early pseudo-labels are noisy.

This is an MSTN-style **variant**, not a faithful reproduction: MSTN keeps
moving averages for both source and target centroids, whereas ShiftKit uses
per-batch source centroids and *detached* EMA target centroids. Target
features therefore receive no direct gradient from this term; only the source
centroids are pulled toward the target ones. Classes never predicted on the
target are skipped rather than pulled toward an uninitialised (zero) centroid.

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
      distributions while centroid alignment encourages (it cannot enforce)
      class-level correspondence, to the extent the pseudo-labels are correct.

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
