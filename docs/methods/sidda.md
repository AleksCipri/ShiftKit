# SIDDATrainer

**SIDDA** (SInkhorn Dynamic Domain Adaptation) trains a model using an
**optimal-transport** domain alignment loss combined with **learnable, dynamically-weighted**
loss terms.  Two key ideas distinguish it from MMD and DANN:

1. **Sinkhorn divergence** as the DA loss — a debiased, entropy-regularised
   optimal transport distance that interpolates between MMD and the Wasserstein distance.

2. **Learned loss balancing** — two scalar parameters η₁ (CE weight) and η₂ (DA weight)
   are jointly learned with the model instead of a hand-tuned λ.

$$\mathcal{L} = \frac{1}{2\eta_1^2}\mathcal{L}_\text{CE} + \frac{1}{2\eta_2^2}\mathcal{S}_\sigma(z_\text{src}, z_\text{tgt}) + \log|\eta_1\eta_2|$$

where $\mathcal{S}_\sigma$ is the Sinkhorn divergence:

$$\mathcal{S}_\sigma(\mu,\nu) = \text{OT}_\sigma(\mu,\nu) - \tfrac{1}{2}\text{OT}_\sigma(\mu,\mu) - \tfrac{1}{2}\text{OT}_\sigma(\nu,\nu)$$

For a fixed non-negative loss term ℒ, the η-optimum of ℒ/(2η²) + log η is
η² = ℒ, so the log term keeps η away from 0 and ∞ **only for non-negative
terms**. With a term that can be negative there is no optimum and η is driven
to its lower clamp; ShiftKit therefore never applies this weighting to such a
term (see [Regression variants](#regression-variants)). The learned η reflect
the current loss magnitudes; they are not shown to give a better balance than a
tuned λ.

```
encoder([x_src; x_tgt]) ──► [z_src; z_tgt]     one joint forward pass
z_src ──► classify(z_src) ──► CE loss ───────────┐
Sinkhorn_σ(z_src, z_tgt)  ──► DA loss ───────────┴─► ℒ (η-weighted)
               ↑ σ recomputed each batch from feature distances
```

After warmup, tensor source and target batches are encoded together (as in
the published code), so BatchNorm layers see both domains in one batch.
Node-level and PyG graph batches are encoded per domain. An epoch covers the
longer of the two loaders; the shorter one restarts.

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
| `target_loader` | `DataLoader` | — | Target DataLoader yielding `x`, `(x,)` or `(x, y)`; labels are used only for `tgt_acc` monitoring (`nan` if absent) |
| `lr` | `float` | `1e-2` | AdamW learning rate (model, and η when `eta_parametrization="raw"`) |
| `weight_decay` | `float` | `1e-3` | AdamW weight decay |
| `warmup_epochs` | `int` | `0` | Epochs of source-only pre-training before DA begins |
| `sigma_scale` | `float` | `0.05` | Scale factor for dynamic blur: σ = max(scale · max‖z‖, floor) |
| `sigma_floor` | `float` | `0.01` | Minimum blur value (prevents degenerate OT plans) |
| `grad_clip` | `float` | `10.0` | Gradient clipping max-norm, applied to the model parameters (η are not clipped) |
| `use_potentials` | `bool` | `False` | Experimental: weight the per-sample CE loss by softmax(-f/τ) of the current batch's source potentials (see [Potential-based reweighting](#potential-based-reweighting-experimental)) |
| `potential_temperature` | `float` | `1.0` | Temperature τ of that softmax |
| `weight_ot` | `bool` | `False` | Experimental: replace the uniform empirical measures by softmax(-f/τ), softmax(-g/τ) weights from the current batch (one extra Sinkhorn solve per batch) |
| `ot_ema_momentum` | `None` | `None` | **Deprecated, ignored** (passing a value emits a `FutureWarning`). The former cross-batch EMA was indexed by batch position and so mixed up unrelated samples |
| `scaling` | `float` | `0.5` | geomloss ε-scaling of the Sinkhorn solve (published code: 0.9). See [Sinkhorn scaling](#sinkhorn-scaling) |
| `eta_parametrization` | `str` | `"log"` | `"log"` (ShiftKit modification): log η trained in a separate Adam. `"raw"`: η trained in the model's AdamW, as in the published code |
| `eta_lr` | `float \| None` | `None` | η learning rate with `eta_parametrization="log"` (`None` → `1e-2`) |
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
| `tgt_acc` | Target accuracy in evaluation mode after each step (`nan` without target labels; never used for training) |
| `eta1` | Current η₁ (CE weight) at epoch end |
| `eta2` | Current η₂ (DA weight) at epoch end |
| `sigma` | Sinkhorn blur σ of the epoch's last batch (`0.0` during warmup) |
| `mean_potential` | Mean source dual potential f of the uniform solve, averaged over batches (`0.0` unless `use_potentials` or `weight_ot`). This is **not** the Sinkhorn divergence (which is ⟨α,f⟩+⟨β,g⟩) and is not a monotone alignment measure |

---

## `evaluate(loader, domain="source")`

Compute classification accuracy on any labelled DataLoader in evaluation mode
(raises `ValueError` for an empty or unlabelled loader).

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
    Warmup length is a tuning choice. As a rule of thumb, it should be long
    enough that source accuracy is reasonable but short enough to avoid
    overfitting the source domain before adaptation starts.

---

## Dynamic Sinkhorn blur

The Sinkhorn regularisation strength σ is adapted **each batch**:

$$\sigma = \max\!\left(0.05 \cdot \max_{i,j}\|z_i - z_j^*\|_2,\; 0.01\right)$$

The distances are between source and target latents, measured on the latents
themselves: the features the Sinkhorn solve runs on, as in the published code.

!!! note "LayerNorm on the latent"
    The published SIDDA model ends its encoder with a LayerNorm
    (`fc1 → nn.LayerNorm(256)`), so σ, the Sinkhorn solve and the classifier
    all see the normalised latent. ShiftKit's `MLP`, `CNN`, `MLPRegressor` and
    `SimpleGCN` do not include one. `GNN` applies `LayerNorm` after each
    convolution by default (`use_layernorm=True`), followed by ReLU, which is not
    the same placement as the published model. To mimic the published
    placement, end your model's `encode()` with `nn.LayerNorm(latent_dim)`,
    inside the model so it is also applied at test time.

σ is a heuristic scale: it follows the largest cross-domain latent distance in
the batch. That distance can decrease as the domains align, but it need not
decrease monotonically (for example if the encoder spreads the latents), so a
shrinking σ is neither guaranteed nor a convergence signal.

---

## Sinkhorn scaling

geomloss solves the entropic OT problem by ε-scaling: it starts from a large
blur and shrinks it by the factor `scaling` at each step until it reaches σ.
`scaling` therefore sets how exactly the Sinkhorn divergence, and its gradient,
is computed: values near 1 take more steps and solve more exactly, smaller
values are faster and coarser.

`scaling` can change the DA gradient, so treat it as a hyperparameter rather
than only a speed setting. ShiftKit's default is `0.5`, which differs from the
published `0.9`; no reproducible experiment record comparing the two is
included here, so the better value for a given problem is untested and worth
tuning.

`scaling=0.9` with `eta_parametrization="raw"` matches the published code's
Sinkhorn setting and η handling. It does **not** reproduce the published
experiments: architecture (e.g. LayerNorm placement), data and augmentation,
learning-rate schedule, validation and checkpoint selection, and other
settings also differ.

---

## Learnable loss weights η

η₁ and η₂ are scalar parameters, both initialised to 1. With the default
`eta_parametrization="log"` (a ShiftKit modification), log η₁ and log η₂ are
optimised in a separate Adam (`eta_lr`, no weight decay). No reproducible
comparison showing it is more stable than the published parametrisation is
recorded; treat that as untested. With `"raw"` they are optimised in the
model's AdamW, as in the published code. η gradients are not clipped. Hard
constraints are enforced at every step (with `"raw"` before the optimiser step,
as published; with `"log"` after it):

- η₁ ≥ 1e-3. This caps the CE coefficient 1/(2η₁²) at 5·10⁵. It does **not**
  prevent a *large* η₁ from making the CE term negligible.
- η₂ ≥ 0.25 · η₁. This bounds the ratio of the DA coefficient to the CE
  coefficient, (η₁/η₂)², by 16. The DA term can still dominate if the raw DA
  loss is much larger than the CE loss.

A training loss that becomes NaN or infinite raises `FloatingPointError`.

You can inspect how the balance evolves during training:

```python
import matplotlib.pyplot as plt

epochs = [h["epoch"] for h in history]
plt.plot(epochs, [h["eta1"] for h in history], label="η₁ (CE)")
plt.plot(epochs, [h["eta2"] for h in history], label="η₂ (DA)")
plt.xlabel("Epoch"); plt.ylabel("η"); plt.legend()
```

---

## Potential-based reweighting (experimental)

`use_potentials` and `weight_ot` are ShiftKit additions; they are not part of
the published SIDDA method. Both are off by default. Their benefit has **not**
been validated experimentally; treat them as heuristics to try, not
recommended settings.

### Dual potentials

For each batch, geomloss can return the Sinkhorn dual potentials (f, g):

- `f[i]` for source latent z_i, `g[j]` for target latent z*_j.

At the solution, the Sinkhorn divergence is recovered by integrating them
against the two measures:

$$\mathcal{S}_\sigma(\alpha,\beta) = \sum_i \alpha_i f_i + \sum_j \beta_j g_j$$

which is `f.mean() + g.mean()` for uniform measures. Potentials are defined up
to the data in the current batch; a dual potential is not a universally valid
per-point "distance to target", and the mean of f alone is not the Sinkhorn
divergence.

Both options use the heuristic weights

$$w_i = \text{softmax}\!\left(\frac{-f_i}{\tau}\right)$$

computed from **the current batch only**. No weights are carried between
batches, so every weight belongs to the sample it was computed for.

### `use_potentials=True` — reweight the supervised loss

$$\mathcal{L}_\text{CE} = \sum_i w_i \cdot \text{CE}(f(z_i), y_i)$$

The weights are detached (no gradient flows through them). This up-weights
source samples with low potential; whether that focuses learning on a
"transferable" subset is a hypothesis, not an established property.

### `weight_ot=True` — reweight the measures

A first, gradient-free solve on uniform measures gives (f, g). The DA loss is
then the Sinkhorn divergence between the weighted measures

$$\alpha = \sum_i \text{softmax}(-f/\tau)_i\, \delta_{z_i}, \quad \beta = \sum_j \text{softmax}(-g/\tau)_j\, \delta_{z^*_j}$$

This costs one extra Sinkhorn solve per batch. Earlier versions kept an
exponential moving average of the weights across batches indexed by batch
position, which attached weights to unrelated samples; that EMA was removed and
`ot_ema_momentum` is deprecated.

### Temperature τ

- **τ → ∞**: uniform weights (standard SIDDA)
- **τ → 0**: nearly all weight on the lowest-potential samples

### Usage

```python
# Supervised-loss reweighting only
trainer = SIDDATrainer(
    model=model, source_loader=train_src, target_loader=train_tgt,
    warmup_epochs=10, lr=1e-2,
    use_potentials=True,
    potential_temperature=1.0,
)

# Measure reweighting only
trainer = SIDDATrainer(
    model=model, source_loader=train_src, target_loader=train_tgt,
    warmup_epochs=10, lr=1e-2,
    weight_ot=True,
    potential_temperature=1.0,
)
history = trainer.fit(epochs=50)
```

Compare against standard SIDDA with the same seed(s) and a held-out labelled
target set before adopting either option.

---

## Regression variants

`SIDDARegressionTrainer` uses the MSE of the regression head instead of CE;
`SIDDAGaussianRegressionTrainer` expects a 2-column head (`predict_var=True`;
column 0 = mean, column 1 = raw value with variance `softplus(raw) + 1e-6`).
Its loss is

$$\mathcal{L} = \frac{1}{2\eta_1^2}\text{MSE}(\mu) + \frac{1}{2\eta_2^2}\mathcal{S}_\sigma + \log|\eta_1\eta_2| + \mathcal{L}_\text{var}$$

Only the non-negative MSE of the mean is η-weighted. The variance term
$\mathcal{L}_\text{var}$ is added unweighted and uses the detached mean, so it
trains the variance head (and shared encoder) without changing the mean
objective:

- `loss="nll"` (default): Gaussian NLL of y under N(μ.detach(), σ²)
  (`torch.nn.GaussianNLLLoss` semantics);
- `loss="moment"`: J1 = mean(((y − μ.detach())² − σ²)²), the Moment-Network
  variance objective (Jeffrey & Wandelt 2020).

The previous implementation η-weighted the logarithmic moment loss of
Villanueva-Domingo et al. (2022). That loss can be negative, which made η₁ fall
steadily toward its clamp. `evaluate()` reports RMSE/MAE/R² of the mean and
`nll`, `cov68`, `cov95` and `mean_sd`. These are diagnostics under a Gaussian
assumption on the evaluated data; they do not establish calibration under
other shifts. Regression history and `evaluate` share one metric convention
(`shiftkit.methods.metrics.regression_metrics`).
