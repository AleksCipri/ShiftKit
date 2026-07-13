# KLIEP — Kullback–Leibler Importance Estimation Procedure

KLIEP is an **instance-based** domain adaptation method.  Instead of aligning
feature distributions (as MMD, CORAL, and DANN do), it estimates how much more
or less likely each source sample is under the target distribution and reweights
the training loss accordingly.

> **Reference** — Sugiyama, M., Nakajima, S., Kashima, H., Bünau, P. V., &
> Kawanabe, M. (2008). Direct Importance Estimation with Model Selection and
> Its Application to Covariate Shift Adaptation. *NeurIPS 20*, 1433–1440.
> [[Paper]](https://papers.nips.cc/paper/3248-direct-importance-estimation-with-model-selection-and-its-application-to-covariate-shift-adaptation)

---

## Core idea: density ratio estimation

Under **covariate shift**, the label conditionals are the same across domains
(`p_src(y|x) = p_tgt(y|x)`) but the marginals differ (`p_src(x) ≠ p_tgt(x)`).
The Bayes-optimal correction is to reweight each source sample by the
**importance weight**:

$$w(\mathbf{x}) = \frac{p_\text{target}(\mathbf{x})}{p_\text{source}(\mathbf{x})}$$

A source sample that is common in the target gets high weight; one that is
rare in the target gets low weight.  Training on the reweighted distribution
approximates training directly on target-distributed data — without ever
seeing target labels.

---

## The KLIEP algorithm

KLIEP models the importance weights as a non-negative kernel expansion:

$$w(\mathbf{x};\,\boldsymbol{\theta}) = \sum_{l=1}^{m} \theta_l \, K_\sigma(\mathbf{x},\,\mathbf{c}_l), \quad \theta_l \geq 0$$

where $\mathbf{c}_l$ are RBF centres drawn from the **target** domain and
$K_\sigma$ is the RBF kernel with bandwidth $\sigma$.

The objective is to maximise the expected log-weight under the target:

$$\max_{\boldsymbol{\theta}} \; \frac{1}{n_\text{tgt}} \sum_{\mathbf{x} \in \mathcal{T}} \log w(\mathbf{x};\boldsymbol{\theta})$$

subject to the normalisation constraint that keeps the reweighted source
distribution a valid probability distribution:

$$\frac{1}{n_\text{src}} \sum_{\mathbf{x} \in \mathcal{S}} w(\mathbf{x};\boldsymbol{\theta}) = 1, \quad \theta_l \geq 0$$

Optimisation proceeds by gradient ascent, followed by non-negativity
projection and re-normalisation after each step.

Once weights are estimated, the model is trained with an importance-weighted
cross-entropy loss:

$$\mathcal{L} = \frac{\sum_i w(\mathbf{x}_i) \cdot \text{CE}(f(\mathbf{x}_i),\, y_i)}{\sum_i w(\mathbf{x}_i)}$$

---

## Feature-based vs instance-based DA

| | Feature-based (MMD, CORAL, DANN) | Instance-based (KLIEP) |
|--|----------------------------------|------------------------|
| **What is aligned** | Latent representation distributions | Training sample weights |
| **Model interface** | `encode()` + `classify()` required | Standard `forward()` only |
| **Alignment cost** | Each forward pass (every epoch) | Once at initialisation |
| **Best for** | Large distributional shifts | Covariate shift (same `p(y\|x)`) |
| **Input dimensionality** | Any (operates on latent space) | Low-to-medium (RBF in input space) |
| **Target labels needed** | No | No |

---

## Usage

### Basic

```python
from shiftkit import DataManager, CNN
from shiftkit import KLIEPTrainer

dm = DataManager(batch_size=256)
train_src, train_tgt = dm.load("mnist_noisy_mnist")
test_src,  test_tgt  = dm.load("mnist_noisy_mnist", train=False)

model = CNN(latent_dim=128, num_classes=10)

trainer = KLIEPTrainer(
    model, train_src, train_tgt,
    n_centers=200,      # RBF basis size
    kliep_iter=500,     # gradient-ascent steps
    weight_clip=10.0,   # prevent extreme weights
    lr=1e-3,
)
history = trainer.fit(epochs=20)

result = trainer.evaluate(test_tgt, domain="target")
print(f"Target accuracy: {result['accuracy']*100:.1f}%")
```

### Standalone weight estimation

`KLIEPWeightEstimator` can be used independently of any trainer — for example,
to inspect or diagnose the density ratio before committing to full training:

```python
import numpy as np
from shiftkit import KLIEPWeightEstimator

# Tabular arrays, shape (n, d)
estimator = KLIEPWeightEstimator(sigma=1.0, n_centers=100, n_iter=500)
estimator.fit(X_source, X_target)

weights = estimator.predict(X_source)   # shape (n_src,)
print(f"Weight distribution: mean={weights.mean():.3f}, max={weights.max():.3f}")
```

### Via TrainerRegistry

```python
from shiftkit.methods import TrainerRegistry

trainer = TrainerRegistry.create(
    "kliep",
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    n_centers=200,
    weight_clip=10.0,
)
```

---

## API reference

### `KLIEPTrainer`

```python
KLIEPTrainer(
    model,                  # nn.Module with standard forward()
    source_loader,
    target_loader,
    sigma=None,             # RBF bandwidth (None → median heuristic)
    n_centers=100,          # number of basis functions
    kliep_lr=0.01,          # KLIEP gradient-ascent step size
    kliep_iter=500,         # KLIEP optimisation steps
    weight_clip=None,       # max importance weight (recommended: 10–100)
    lr=1e-3,                # model optimiser learning rate
    device=None,            # auto-detected
)
```

**`fit(epochs)`** — returns `list[dict]` with per-epoch keys:

| Key | Description |
|-----|-------------|
| `epoch` | Epoch index (1-based) |
| `ce_loss` | Importance-weighted cross-entropy |
| `mmd_loss` | Always `0.0` (for history compatibility) |
| `total_loss` | Same as `ce_loss` |
| `src_acc` | Source accuracy |
| `tgt_acc` | Target accuracy (not optimised directly) |
| `mean_weight` | Mean importance weight across batches |
| `max_weight` | Maximum importance weight seen in epoch |

**`evaluate(loader, domain)`** — returns `{"domain", "accuracy", "n_samples"}`.

### `KLIEPWeightEstimator`

```python
KLIEPWeightEstimator(
    sigma=None,             # RBF bandwidth (None → median heuristic)
    n_centers=100,          # number of RBF centres sampled from target
    lr=0.01,                # gradient-ascent step size
    n_iter=500,             # number of optimisation iterations
    weight_clip=None,       # clip weights to [0, weight_clip]
    seed=0,
)

estimator.fit(X_src, X_tgt)   # (n, d) float32 numpy arrays
weights = estimator.predict(X)  # (n,) float32 numpy array
```

---

## Practical notes

**Bandwidth σ (median heuristic)**
: If `sigma=None`, KLIEP estimates σ from the median pairwise distance in a
  subsample of the combined data. This is usually a good default. Set σ manually
  if features have very different scales or the automatic estimate behaves poorly.

**Number of centres**
: Larger `n_centers` → more expressive model, slower estimation.
  100–500 centres is typically sufficient. Use fewer for high-dimensional inputs.

**Weight clipping**
: Extreme importance weights (w ≫ 1) can destabilise gradient updates.
  `weight_clip=10.0` is a safe default; tighten to `5.0` if training is noisy.

**Input dimensionality**
: KLIEP estimates density ratios in **raw input space**.  For low-dimensional
  tabular data this works well.  For images (e.g. 784-dim MNIST pixels) the
  RBF kernel becomes less discriminative — consider applying PCA or using a
  feature-based method (MMD, CORAL) instead.

**Covariate shift assumption**
: KLIEP is theoretically grounded when `p_src(y|x) = p_tgt(y|x)`.  If the
  labelling function itself changes across domains, feature-based methods
  that do not rely on this assumption may perform better.
