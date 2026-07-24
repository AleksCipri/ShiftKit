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
encoder(x) ──► z_src ──► classify(z_src) ──► CE loss ────────┐
encoder(x) ──► z_tgt ──►                                     ├─► ℒ (weighted)
               Sinkhorn_σ(z_src, z_tgt) ──► DA loss ─────────┘
               ↑ σ recomputed each batch from feature distances
```

> **Reference:** Pandya, S., Patel, P., Nord, B. D., Walmsley, M., & Ćiprijanović, A. (2025).
> SIDDA: SInkhorn Dynamic Domain Adaptation for image classification with equivariant neural networks.
> *Machine Learning: Science and Technology*, 6(3), 035032.
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
| `use_potentials` | `bool` | `False` | Reweight per-sample CE loss using Kantorovich dual potentials (see [Latent-space instance reweighting](#latent-space-instance-reweighting)) |
| `potential_temperature` | `float` | `1.0` | Temperature τ for softmax reweighting: w = softmax(-f / τ). Lower τ = sharper focus on already-aligned samples |
| `weight_ot` | `bool` | `False` | Reweight the Sinkhorn transport plan itself via an EMA of previous-batch potentials, focusing OT alignment on the overlapping region (see [Latent-space instance reweighting](#latent-space-instance-reweighting)) |
| `ot_ema_momentum` | `float` | `0.9` | EMA momentum for OT weight updates: w_t = mom·w_{t-1} + (1-mom)·softmax(-f_t / τ) |
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
| `total_loss` | Mean total weighted loss |
| `src_acc` | Source domain training accuracy |
| `tgt_acc` | Target domain accuracy (tracked, not directly optimised) |
| `eta1` | Current η₁ (CE weight) at epoch end |
| `eta2` | Current η₂ (DA weight) at epoch end |
| `sigma` | Mean Sinkhorn blur σ used this epoch (`0.0` during warmup) |
| `mean_potential` | Mean source Kantorovich potential F across batches (`0.0` unless `use_potentials=True`) |

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

---

## Latent-space instance reweighting

SIDDA exposes two optional parameters — `use_potentials` and `weight_ot` — that use the **Kantorovich dual potentials** computed by geomloss to focus adaptation on the overlapping region of the two distributions. Both are off by default; they can be used independently or together.

### Dual potentials

During Sinkhorn iterations, geomloss computes dual potentials (f, g) as a byproduct:

- `f[i]` shape `(n_src,)` — transport cost for source point z_i in latent space
- `g[j]` shape `(n_tgt,)` — transport cost for target point z_j

A **low** `f[i]` means z_i is already close to the target; a **high** `f[i]` means it is far. The Sinkhorn divergence is recovered as their mean sum (primal = dual at the optimum):

$$\mathcal{S}_\sigma(\mu,\nu) = \mathbb{E}[f] + \mathbb{E}[g]$$

Both options derive instance weights from f via a temperature-scaled softmax:

$$w_i = \text{softmax}\!\left(\frac{-f_i}{\tau}\right)$$

### `use_potentials=True` — reweight the CE loss

Upweights source samples already close to the target when computing the classification loss, so the classifier learns primarily from the transferable subset:

$$\mathcal{L}_\text{CE} = \sum_i w_i \cdot \text{CE}(f(z_i), y_i)$$

The transport plan itself remains uniform — only the classifier supervision signal is reweighted.

### `weight_ot=True` — reweight the transport plan (EMA)

Modifies the Sinkhorn measure itself so that alignment effort concentrates on the overlapping region. geomloss accepts weighted discrete measures:

$$\alpha = \sum_i w_i^{(\text{src})} \delta_{z_i}, \quad \beta = \sum_j w_j^{(\text{tgt})} \delta_{z_j}$$

Because the weights for batch *t* can only come from the potentials of batch *t* (a circularity), an **exponential moving average** is used: weights from the previous batch are passed into the current Sinkhorn call, then the EMA is updated from the resulting potentials:

$$w_t = \text{mom} \cdot w_{t-1} + (1 - \text{mom}) \cdot \text{softmax}(-f_t / \tau)$$

The EMA is initialised with a uniform-measure solve on the first batch and reseeded automatically if the batch size changes (e.g. the final batch of an epoch).

### Temperature τ

`potential_temperature` controls the sharpness of both reweightings:

- **τ = 1.0** (default): moderate reweighting; all samples contribute
- **τ → 0**: hard selection — only the most-aligned samples receive weight
- **τ → ∞**: uniform weighting — equivalent to standard SIDDA

### Usage

```python
# CE reweighting only
trainer = SIDDATrainer(
    model=model, source_loader=train_src, target_loader=train_tgt,
    warmup_epochs=10, lr=1e-2,
    use_potentials=True,
    potential_temperature=1.0,
)

# OT plan reweighting only
trainer = SIDDATrainer(
    model=model, source_loader=train_src, target_loader=train_tgt,
    warmup_epochs=10, lr=1e-2,
    weight_ot=True,
    ot_ema_momentum=0.9,
    potential_temperature=1.0,
)

# Both together — CE and OT plan reweighted from the same potentials
# (single Sinkhorn call per batch, no extra cost)
trainer = SIDDATrainer(
    model=model, source_loader=train_src, target_loader=train_tgt,
    warmup_epochs=10, lr=1e-2,
    use_potentials=True,
    weight_ot=True,
    potential_temperature=1.0,
    ot_ema_momentum=0.9,
)
history = trainer.fit(epochs=50)

# Monitor transport cost — decreases as distributions align
import matplotlib.pyplot as plt
plt.plot([h["epoch"] for h in history], [h["mean_potential"] for h in history])
plt.xlabel("Epoch"); plt.ylabel("Mean source potential f")
```

!!! tip "When to use these options"
    Both options are most beneficial when the source and target distributions overlap
    only partially. `weight_ot=True` is the stronger intervention — it changes what
    the OT plan optimises, not just how the classifier uses source labels.
    Start with standard SIDDA as the baseline and add these if the domain gap is large.
