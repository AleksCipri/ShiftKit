# SIDDATrainer

**SIDDA** (SInkhorn Dynamic Domain Adaptation) trains a model using an
**optimal-transport** domain alignment loss combined with **learnable, dynamically-weighted**
loss terms.  Two key ideas distinguish it from MMD and DANN:

1. **Sinkhorn divergence** as the DA loss — a debiased, entropy-regularised
   optimal transport distance that interpolates between MMD and the Wasserstein distance.

2. **Automatic loss balancing** — two scalar parameters η₁ (CE weight) and η₂ (DA weight)
   are jointly learned with the model, eliminating the need to hand-tune λ.

$$\mathcal{L} = \frac{1}{2\eta_1^2}\mathcal{L}_\text{CE} + \frac{1}{2\eta_2^2}\mathcal{S}_\sigma(z_\text{src}, z_\text{tgt}) + \log|\eta_1\eta_2|$$

where $\mathcal{S}_\sigma$ is the Sinkhorn divergence:

$$\mathcal{S}_\sigma(\mu,\nu) = \text{OT}_\sigma(\mu,\nu) - \tfrac{1}{2}\text{OT}_\sigma(\mu,\mu) - \tfrac{1}{2}\text{OT}_\sigma(\nu,\nu)$$

The log term regularises η, preventing collapse to zero or unbounded growth.

```
encoder(x) ──► z_src ──► classify(z_src) ──► CE loss ──────--┐
encoder(x) ──► z_tgt ──►                                     ├─► ℒ (weighted)
               Sinkhorn_σ(z_src, z_tgt) ──► DA loss ─────────┘
               ↑ σ recomputed each batch from feature distances
```

> **Reference:** Ciprijanovic, A., Lewis, A., Pedro, K., Downey, E., Nord, B.,
> & Stark, A. (2025). SIDDA: SInkhorn Dynamic Domain Adaptation for Image
> Classification with Equivariant Neural Networks. *Mach. Learn.: Sci. Technol.*, 6, 035032.
> [[Paper]](https://iopscience.iop.org/article/10.1088/2632-2153/adf701) 

!!! note "Dependency"
    SIDDA requires **geomloss** for the Sinkhorn divergence computation:
    ```bash
    pip install geomloss
    ```

---

## Usage

```python
from shiftkit.methods import SIDDATrainer

trainer = SIDDATrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    lr=1e-2,
    warmup_epochs=10,   # source-only pre-training before DA begins
)
history = trainer.fit(epochs=50)

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
| `lr` | `float` | `1e-2` | AdamW learning rate (model + η parameters) |
| `weight_decay` | `float` | `1e-3` | AdamW weight decay |
| `warmup_epochs` | `int` | `0` | Epochs of source-only pre-training before DA begins |
| `sigma_scale` | `float` | `0.05` | Scale factor for dynamic blur: σ = max(scale · max‖z‖, floor) |
| `sigma_floor` | `float` | `0.01` | Minimum blur value (prevents degenerate OT plans) |
| `grad_clip` | `float` | `10.0` | Gradient clipping max-norm |
| `device` | `str \| None` | `None` | `'cuda'`, `'mps'`, or `'cpu'`; auto-detected if `None` |

---

## `fit(epochs=50)`

Train for `epochs` epochs and return the history.

**Returns:** `list[dict]` — one dict per epoch:

| Key | Description |
|-----|-------------|
| `epoch` | Epoch number (1-indexed) |
| `ce_loss` | Mean cross-entropy loss |
| `da_loss` | Mean Sinkhorn divergence loss (`0.0` during warmup) |
| `mmd_loss` | Always `0.0` (for history-format compatibility) |
| `total_loss` | Mean total weighted loss |
| `src_acc` | Source domain training accuracy |
| `tgt_acc` | Target domain accuracy (tracked, not directly optimised) |
| `eta1` | Current η₁ (CE weight) at epoch end |
| `eta2` | Current η₂ (DA weight) at epoch end |
| `sigma` | Mean Sinkhorn blur σ used this epoch (`0.0` during warmup) |

---

## `evaluate(loader, domain="source")`

Compute classification accuracy on any labelled DataLoader.

**Returns:** `dict` with keys `domain` (str), `accuracy` (float), `n_samples` (int).

---

## Warmup phase

During the warmup phase the encoder trains on source classification only — no DA
loss, no η updates.  This ensures the latent space carries class-discriminative
structure before Sinkhorn alignment begins.

```python
trainer = SIDDATrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    warmup_epochs=10,   # first 10 epochs: source-only CE
    lr=1e-2,
)
history = trainer.fit(epochs=50)
# Epochs 1–10:  [warmup]  CE only
# Epochs 11–50: [SIDDA ]  CE + Sinkhorn DA + learnable η
```

!!! tip "Choosing warmup length"
    The paper found that equivariant networks require a shorter warmup than
    standard CNNs (e.g. 5–10 vs 10–30 epochs).  As a rule of thumb, warmup
    should be long enough that source accuracy is reasonable but short enough
    to avoid overfitting the source domain before adaptation starts.

---

## Dynamic Sinkhorn blur

The Sinkhorn regularisation strength σ is adapted **each batch**:

$$\sigma = \max\!\left(0.05 \cdot \max_{i,j}\|z_i - z_j^*\|_2,\; 0.01\right)$$

Layer normalisation is applied to the latent features before computing pairwise
distances, preventing a small number of outlier features from inflating σ.
As training progresses and source/target distributions align, σ shrinks
naturally, giving a progressively sharper (more exact) OT plan.

---

## Learnable loss weights η

η₁ and η₂ are scalar `nn.Parameter` values optimised jointly with the model.
Hard constraints are enforced after every gradient step:

- η₁ ≥ 1e-3 (prevents CE from being silenced)
- η₂ ≥ 0.25 · η₁ (prevents DA from completely dominating)

You can inspect how the balance evolves during training:

```python
import matplotlib.pyplot as plt

epochs = [h["epoch"] for h in history]
plt.plot(epochs, [h["eta1"] for h in history], label="η₁ (CE)")
plt.plot(epochs, [h["eta2"] for h in history], label="η₂ (DA)")
plt.xlabel("Epoch"); plt.ylabel("η"); plt.legend()
```
