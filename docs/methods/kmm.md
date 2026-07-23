# KMMTrainer

**KMM** (Kernel Mean Matching) is an **instance-based** domain adaptation method.
Instead of aligning feature distributions (as MMD, CORAL, and DANN do), it estimates
how much more or less likely each source sample is under the target distribution and
reweights the training loss accordingly.

## Core idea: minimising MMD in weight space

Under **covariate shift**, the label conditionals are the same across domains
(`p_src(y|x) = p_tgt(y|x)`) but the marginals differ (`p_src(x) ≠ p_tgt(x)`).
The theoretically correct fix is to reweight each source sample by the **importance weight**:

$$w(\mathbf{x}) = \frac{p_\text{target}(\mathbf{x})}{p_\text{source}(\mathbf{x})}$$

KMM finds these weights by solving a convex **quadratic programme** that directly
minimises the MMD between the reweighted source distribution and the target in an RBF
RKHS, subject to non-negativity and a normalisation constraint:

$$\min_{\mathbf{w}} \quad \frac{1}{2}\mathbf{w}^\top K_{ss}\mathbf{w} - \boldsymbol{\kappa}^\top \mathbf{w}$$

$$\text{s.t.} \quad w_i \geq 0, \quad B \geq w_i, \quad \left|\frac{1}{n_s}\sum_i w_i - 1\right| \leq \varepsilon$$

where $K_{ss}[i,j] = k(x_i^s, x_j^s)$, $\kappa_i = \frac{n_s}{n_t}\sum_j k(x_i^s, x_j^t)$,
$B$ is an upper bound on the weights, and $\varepsilon$ controls the normalisation tolerance.

Once solved, the model is trained with an importance-weighted cross-entropy loss:

$$\mathcal{L} = \frac{\sum_i w(\mathbf{x}_i) \cdot \text{CE}(f(\mathbf{x}_i),\, y_i)}{\sum_i w(\mathbf{x}_i)}$$

> **Reference:** Huang, J., Smola, A. J., Gretton, A., Borgwardt, K. M., & Schölkopf, B. (2007).
> Correcting Sample Selection Bias by Unlabeled Data. *Advances in Neural Information Processing Systems*, 19.
> [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2006/file/a2186aa7c086b46ad4e8bf81e2a3a19b-Paper.pdf)

---

## KMM vs KLIEP

Both KMM and KLIEP are instance-based methods that estimate the same density ratio — the key difference is *how* they solve for the weights:

| | KMM | KLIEP |
|--|-----|-------|
| **Objective** | Minimise MMD between reweighted source and target | Maximise KL divergence from target to reweighted source |
| **Solver** | Convex QP (scipy SLSQP) | Gradient ascent with projection |
| **Weights defined on** | All source samples jointly (global solution) | Each source sample independently via basis expansion |
| **Scales with** | n_src² (kernel matrix) | n_centers (basis size) |
| **Upper bound on weights** | Yes — explicit parameter `B` | Via `weight_clip` |
| **Best for** | Small-to-medium datasets, when exact QP is tractable | Larger datasets where kernel matrix is expensive |

---

## Usage

### Basic

```python
from shiftkit import DataManager, CNN
from shiftkit import KMMTrainer

dm = DataManager(batch_size=256)
train_src, train_tgt = dm.load("mnist_noisy_mnist")
test_src,  test_tgt  = dm.load("mnist_noisy_mnist", train=False)

model = CNN(latent_dim=128, num_classes=10)

trainer = KMMTrainer(
    model, train_src, train_tgt,
    B=1000.0,        # upper bound on weights
    lr=1e-3,
)
history = trainer.fit(epochs=20)

result = trainer.evaluate(test_tgt, domain="target")
print(f"Target accuracy: {result['accuracy']*100:.1f}%")
```

### Standalone weight estimation

`KMMWeightEstimator` can be used independently to inspect the density ratio
before committing to full training:

```python
import numpy as np
from shiftkit import KMMWeightEstimator

estimator = KMMWeightEstimator(sigma=1.0, B=1000.0)
estimator.fit(X_source, X_target)   # (n, d) float32/64 numpy arrays

weights = estimator.predict(X_source)
print(f"Weight distribution: mean={weights.mean():.3f}, max={weights.max():.3f}")
```

### Via TrainerRegistry

```python
from shiftkit.methods import TrainerRegistry

trainer = TrainerRegistry.create(
    "kmm",
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    B=1000.0,
)
```

---

## API reference

### `KMMTrainer`

```python
KMMTrainer(
    model,                  # nn.Module with standard forward()
    source_loader,
    target_loader,
    sigma=None,             # RBF bandwidth (None → median heuristic)
    B=1000.0,               # upper bound on each importance weight
    epsilon=None,           # normalisation tolerance (None → (√n-1)/√n)
    weight_clip=None,       # hard clip on weights after solving
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

### `KMMWeightEstimator`

```python
KMMWeightEstimator(
    sigma=None,     # RBF bandwidth (None → median heuristic)
    B=1000.0,       # upper bound on weights
    epsilon=None,   # normalisation tolerance
    weight_clip=None,
)

estimator.fit(X_src, X_tgt)   # (n, d) float32/64 numpy arrays
weights = estimator.predict(X_src)  # returns the fitted weights (n_src,)
```

---

## Practical notes

**Bandwidth σ (median heuristic)**
: If `sigma=None`, KMM uses the median pairwise distance across a subsample of the
  combined data. Set manually if features have very different scales.

**Weight upper bound B**
: `B` caps individual weights to prevent extreme reweighting from a few source samples.
  The default `B=1000` is permissive — tighten to `10–100` if weights are
  destabilising training. The `weight_clip` parameter applies a hard clip *after* the
  QP solve, which is an alternative way to control the same thing.

**Scaling**
: KMM builds an n_src × n_src kernel matrix, so memory and QP solve time grow
  quadratically with source set size. For datasets larger than ~5 000 samples,
  consider KLIEP instead, which avoids the full kernel matrix.

**Covariate shift assumption**
: KMM is theoretically grounded when `p_src(y|x) = p_tgt(y|x)`. If the labelling
  function itself changes across domains, feature-based methods may perform better.
