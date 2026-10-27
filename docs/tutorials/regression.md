# Tutorial 6 — Regression with Domain Adaptation

Most ShiftKit examples use **classification** tasks, but domain shift affects regression
problems too — for example, a model trained to predict house prices in Northern California
may generalise poorly when deployed in Southern California.

This tutorial walks through two regression DA experiments using `SourceOnlyRegressionTrainer`
(baseline) and `MMDRegressionTrainer` (alignment).  The same `MLPRegressor` architecture and
`DataManager` interface are used throughout.

---

## New classes introduced

| Class | Module | Purpose |
|-------|--------|---------|
| `MLPRegressor` | `shiftkit.models` | Flexible MLP with `encode` / `regress` interface |
| `SourceOnlyRegressionTrainer` | `shiftkit.methods` | MSE baseline, no alignment |
| `MMDRegressionTrainer` | `shiftkit.methods` | MSE + λ·MMD² feature alignment |
| `SineWaveDataset` | `shiftkit.data` | Synthetic sine regression (used in the limitation example) |
| `CaliforniaHousingDataset` | `shiftkit.data` | Geographic CA housing split |

All regression trainers track **RMSE** and **R²** per epoch (source and target) and report
**MSE**, **RMSE**, **MAE** and **R²** from `evaluate()`. History and `evaluate()` use the same
convention: MSE/RMSE/MAE average over all output elements; R² is computed per output column
and averaged. Target labels, if present, are used only for monitoring (`tgt_rmse` is `None`
for an unlabelled target loader).

---

## Example 1 — Sine Wave (Covariate Shift)

### The domain gap

Both domains share the same labelling rule, `y = sin(x) + noise` (noise SD 0.05);
only the input distribution changes:

```
source: x ~ Uniform(−π, π)
target: x ~ Normal(π/2, 0.6²), clipped to (−π, π)
```

The target inputs lie inside the source range, so the shift is identifiable: a
model that fits `sin(x)` well on the source also fits the target. This example
builds the data inline with `TensorDataset`.

### Setup

```python
import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from shiftkit import MLPRegressor, SourceOnlyRegressionTrainer, MMDRegressionTrainer

torch.manual_seed(0)
rng = np.random.default_rng(0)

def make_domain(n, target):
    if target:   # target: x concentrated near π/2, inside the source range
        x = np.clip(rng.normal(np.pi / 2, 0.6, n), -np.pi, np.pi)
    else:        # source: x uniform on (−π, π)
        x = rng.uniform(-np.pi, np.pi, n)
    y = np.sin(x) + 0.05 * rng.standard_normal(n)
    return TensorDataset(torch.tensor(x, dtype=torch.float32).unsqueeze(1),
                         torch.tensor(y, dtype=torch.float32).unsqueeze(1))

train_src = DataLoader(make_domain(2000, False), batch_size=256, shuffle=True)
train_tgt = DataLoader(make_domain(2000, True),  batch_size=256, shuffle=True)
test_src  = DataLoader(make_domain(2000, False), batch_size=256)
test_tgt  = DataLoader(make_domain(2000, True),  batch_size=256)

initial = MLPRegressor(input_dim=1, latent_dim=32, hidden_dims=(64, 64))
```

### Source-only baseline

```python
trainer_so = SourceOnlyRegressionTrainer(copy.deepcopy(initial), train_src, train_tgt, lr=1e-3)
history_so = trainer_so.fit(epochs=30)

result_so = trainer_so.evaluate(test_tgt, domain="target")
print(f"Source-only → target  RMSE={result_so['rmse']:.4f}  R²={result_so['r2']:.3f}")
```

### MMD-aligned model

```python
trainer_mmd = MMDRegressionTrainer(
    copy.deepcopy(initial), train_src, train_tgt,
    mmd_weight=0.5, warmup_epochs=5, lr=1e-3
)
history_mmd = trainer_mmd.fit(epochs=30)

result_mmd = trainer_mmd.evaluate(test_tgt, domain="target")
print(f"MMD          → target  RMSE={result_mmd['rmse']:.4f}  R²={result_mmd['r2']:.3f}")
```

### Compare training curves

```python
import matplotlib.pyplot as plt

epochs = [e["epoch"] for e in history_so]

plt.figure(figsize=(8, 4))
plt.plot(epochs, [e["tgt_rmse"] for e in history_so],  label="Source-only (target)")
plt.plot(epochs, [e["tgt_rmse"] for e in history_mmd], label="MMD (target)")
plt.plot(epochs, [e["src_rmse"] for e in history_mmd], label="MMD (source)", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Sine wave — RMSE during training")
plt.legend()
plt.tight_layout()
plt.show()
```

### Results from one run

Running the code above on CPU (seed 0, 30 epochs, both models starting from the
same initial weights) gave:

| Model | Source RMSE | Target RMSE | Target R² |
|-------|-------------|-------------|-----------|
| Source-only | 0.059 | 0.059 | 0.922 |
| MMD (λ=0.5) | 0.106 | 0.075 | 0.874 |

This is a single seed, not an average. In this run the source-only model already
transferred, because the target inputs are covered by the source data. MMD
alignment did not help and slightly increased both errors. Marginal alignment
is not guaranteed to help regression: when the conditional `p(y|x)` is shared and
the source covers the target, aligning latent marginals can distort an encoder
that was already adequate. (Target R² is lower than source R² at a similar
RMSE because `y` varies less on the target.)

### Limitation example — a pure phase change

`DataManager().load("sine_wave")` gives a different problem: source
`y = sin(x)` and target `y = sin(x + π/3)`, with the **same** input
distribution `x ~ Uniform(−π, π)` in both domains. Only the labelling rule
`p(y|x)` changes. Unlabelled target inputs carry no information about the new
phase, and with identical input distributions there is no population-level
marginal shift for MMD to remove. A predictor that is perfect on the source has
target MSE `1 − cos(π/3) + 0.05² = 0.5025`, i.e. **RMSE ≈ 0.709**.

Training the same two models on that data (seed 0, 30 epochs, CPU) gave target
RMSE 0.711 (source-only) and 0.724 (MMD), both at or slightly above the 0.709
floor, with target R² ≈ 0. No method that uses only unlabelled target inputs can
fix this kind of shift; it needs target labels or prior knowledge of the change.

---

## Example 2 — California Housing (Geographic Split)

### The domain gap

The California Housing dataset is loaded with scikit-learn's
`fetch_california_housing`, which downloads it on first use and caches it locally
(so the first run needs internet access). It contains census-block-group housing
statistics. We split by **latitude**:

- **Source (North, lat ≥ 36°)**: e.g. Bay Area, Sacramento — 8,826 samples
- **Target (South, lat < 36°)**: e.g. Greater Los Angeles, San Diego — 11,814 samples

Each domain is further split 80/20 into training and test sets.

Northern and Southern California differ in price distributions, housing density and
income patterns. The change may involve `p(y|x)` as well as `p(x)`, so it is not
guaranteed to be a pure covariate shift.

The prediction target is **median house value** (in units of $100 k).

### Setup

```python
dm = DataManager(batch_size=128)
train_src, train_tgt = dm.load("california_housing")
test_src,  test_tgt  = dm.load("california_housing", train=False)
```

Features are standardised with statistics fitted on the **training split only**, and the
same transform is applied to both domains and to the test sets, so no test data
influence the normalisation.

### Source-only baseline

```python
model_so = MLPRegressor(
    input_dim=8,          # 8 housing features
    latent_dim=64,
    hidden_dims=(128, 64),
    dropout=0.1,
)

trainer_so = SourceOnlyRegressionTrainer(
    model_so, train_src, train_tgt, lr=1e-3
)
history_so = trainer_so.fit(epochs=40)

result_so = trainer_so.evaluate(test_tgt, domain="target")
print(f"Source-only → target  RMSE={result_so['rmse']:.4f}  R²={result_so['r2']:.3f}")
```

### MMD-aligned model

```python
model_mmd = MLPRegressor(
    input_dim=8,
    latent_dim=64,
    hidden_dims=(128, 64),
    dropout=0.1,
)

trainer_mmd = MMDRegressionTrainer(
    model_mmd, train_src, train_tgt,
    mmd_weight=0.3, warmup_epochs=5, lr=1e-3
)
history_mmd = trainer_mmd.fit(epochs=40)

result_mmd = trainer_mmd.evaluate(test_tgt, domain="target")
print(f"MMD          → target  RMSE={result_mmd['rmse']:.4f}  R²={result_mmd['r2']:.3f}")
```

### Evaluate on both domains

```python
for trainer, label in [(trainer_so, "Source-only"), (trainer_mmd, "MMD")]:
    r_src = trainer.evaluate(test_src, domain="source")
    r_tgt = trainer.evaluate(test_tgt, domain="target")
    print(f"{label:12s}  src R²={r_src['r2']:.3f}  tgt R²={r_tgt['r2']:.3f}")
```

Results depend on the seed, the number of epochs and the λ value; we do not report
reference numbers for this example here. Compare both models on the labelled target test
set over several seeds before concluding that MMD helps.

---

## Tuning tips

| Symptom | Likely cause | Try |
|---------|-------------|-----|
| Target RMSE worse with MMD than without | λ too large — MMD dominates MSE | Reduce `mmd_weight` (try 0.1–0.5) |
| Source RMSE rises sharply after warmup | Too aggressive alignment | Increase `warmup_epochs` |
| Both RMSE stay flat | Learning rate too low | Increase `lr` or reduce `hidden_dims` |
| Target RMSE oscillates | Batch size too small for stable MMD estimate | Increase `batch_size` (≥ 128 recommended) |

## API reference

### `MLPRegressor`

```python
MLPRegressor(
    input_dim,                  # number of input features (required)
    latent_dim=64,              # bottleneck size (passed to MMD)
    output_dim=1,               # regression output size
    hidden_dims=(128, 64),      # encoder hidden layer sizes
    dropout=0.1,
)
```

### `MMDRegressionTrainer`

```python
MMDRegressionTrainer(
    model, source_loader, target_loader,
    mmd_weight=1.0,             # λ for the MMD term
    lr=1e-3,
    warmup_epochs=0,            # source-only pre-training epochs
    device=None,                # auto-detected
    mmd_sigmas=None,            # RBF bandwidths (default: [0.1, 1, 5, 10, 50])
)
```

`evaluate(loader, domain)` returns `{"domain", "mse", "rmse", "mae", "r2", "n_samples"}`
(evaluation mode; raises `ValueError` for an empty or unlabelled loader).
