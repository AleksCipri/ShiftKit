# Tutorial 6 — Regression with Domain Adaptation

Most ShiftKit examples use **classification** tasks, but domain shift affects regression
problems just as commonly — a model trained to predict house prices in Northern California
or fit a sine wave at one phase will generalise poorly when deployed in a different
geographic region or with a shifted input signal.

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
| `SineWaveDataset` | `shiftkit.data` | Synthetic phase-shifted sine regression |
| `CaliforniaHousingDataset` | `shiftkit.data` | Geographic CA housing split |

All regression trainers track **RMSE** per epoch (source and target) and report **MSE**,
**RMSE**, and **R²** from `evaluate()`.

---

## Example 1 — Sine Wave (Phase Shift)

### The domain gap

The source domain is `y = sin(x)` sampled uniformly from `(−π, π)`.
The target domain has the same shape but is shifted by π/3 (60°):

```
source: y = sin(x)
target: y = sin(x + π/3)
```

A model trained only on source data will try to predict the *wrong* phase on the target,
leading to high RMSE.  MMD alignment encourages the encoder to map both signals to a
shared latent space, reducing the phase-induced gap.

### Setup

```python
import shiftkit
from shiftkit import DataManager, MLPRegressor
from shiftkit import SourceOnlyRegressionTrainer, MMDRegressionTrainer

dm = DataManager(batch_size=256)
train_src, train_tgt = dm.load("sine_wave")           # phase 0  vs  π/3
test_src,  test_tgt  = dm.load("sine_wave", train=False)
```

### Source-only baseline

```python
import copy

model_so = MLPRegressor(input_dim=1, latent_dim=32, hidden_dims=(64, 64))

trainer_so = SourceOnlyRegressionTrainer(
    model_so, train_src, train_tgt, lr=1e-3
)
history_so = trainer_so.fit(epochs=30)

result_so = trainer_so.evaluate(test_tgt, domain="target")
print(f"Source-only → target  RMSE={result_so['rmse']:.4f}  R²={result_so['r2']:.3f}")
```

### MMD-aligned model

```python
model_mmd = MLPRegressor(input_dim=1, latent_dim=32, hidden_dims=(64, 64))

trainer_mmd = MMDRegressionTrainer(
    model_mmd, train_src, train_tgt,
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
plt.title("Sine wave — target RMSE during training")
plt.legend()
plt.tight_layout()
plt.show()
```

### Typical results

| Model | Source RMSE | Target RMSE | Target R² |
|-------|-------------|-------------|-----------|
| Source-only | ~0.06 | ~0.47 | ~0.56 |
| MMD (λ=0.5) | ~0.07 | ~0.12 | ~0.93 |

The domain gap is large (target RMSE ≈ 8× source) without adaptation.
MMD alignment recovers most of the target performance.

---

## Example 2 — California Housing (Geographic Split)

### The domain gap

The California Housing dataset (scikit-learn builtin, no download required) contains
census-block-level housing statistics.  We split by **latitude**:

- **Source (North, lat ≥ 36°)**: Bay Area, Sacramento, Fresno — ~11 k samples
- **Target (South, lat < 36°)**: Greater Los Angeles, San Diego — ~9 k samples

Northern and Southern California have meaningfully different price distributions, housing
density, and income patterns — a realistic geographic covariate shift.

The prediction target is **median house value** (in units of $100 k).

### Setup

```python
dm = DataManager(batch_size=128)
train_src, train_tgt = dm.load("california_housing")
test_src,  test_tgt  = dm.load("california_housing", train=False)
```

Both domains are normalised with statistics computed from the **full** dataset, so
feature scales are consistent across domains.

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

### Typical results

| Model | Source R² | Target R² | Target RMSE |
|-------|-----------|-----------|-------------|
| Source-only | ~0.64 | ~0.55 | ~0.61 |
| MMD (λ=0.3) | ~0.63 | ~0.61 | ~0.57 |

> The improvement from MMD is modest but consistent.  California Housing has a relatively
> small geographic gap — the two regions share many feature correlations.  More pronounced
> shifts (e.g. different countries or years) would show larger gains.

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

`evaluate(loader, domain)` returns `{"domain", "mse", "rmse", "r2", "n_samples"}`.
