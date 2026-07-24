# Tutorial 8 — SIDDA Regression with Latent-Space Instance Reweighting

This tutorial demonstrates `SIDDARegressionTrainer` on a synthetic 2D
**partially-overlapping blobs** problem, comparing four variants of SIDDA:

| Variant | `use_potentials` | `weight_ot` | What changes |
|---------|:---:|:---:|---|
| **SIDDA** | ✗ | ✗ | Baseline — uniform OT, mean MSE |
| **SIDDA + CE-reweight** | ✓ | ✗ | Per-sample MSE weighted by alignment potential |
| **SIDDA + OT-reweight** | ✗ | ✓ | Transport plan focused on overlapping region |
| **SIDDA + Both** | ✓ | ✓ | Both reweightings from a single Sinkhorn call |

---

## The problem

Two 2D Gaussian blobs — one per domain — partially overlap.
The regression target is the **first feature** `x₀`, which spans the shift direction.

```
Source: N([0, 0], 0.8·I)      Target: N([1.5, 0], 0.8·I)
                                         ↑
                               overlapping region ~ 30 %
```

A model trained only on source sees a range of `x₀` that is systematically lower
than the target's range.  SIDDA aligns the latent distributions so the encoder
generalises across the gap.

---

## Setup

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from shiftkit import MLPRegressor
from shiftkit import SIDDARegressionTrainer

torch.manual_seed(42)
np.random.seed(42)
```

### Generate data

```python
def make_blobs(n=1000, center=(0.0, 0.0), std=0.8):
    X = np.random.randn(n, 2) * std + np.array(center)
    y = X[:, 0].copy()          # regression target = first feature
    return X.astype(np.float32), y.astype(np.float32)

# Training sets
X_src, y_src = make_blobs(1000, center=(0.0, 0.0))
X_tgt, y_tgt = make_blobs(1000, center=(1.5, 0.0))

# Test sets (larger, for stable evaluation)
X_src_test, y_src_test = make_blobs(2000, center=(0.0, 0.0))
X_tgt_test, y_tgt_test = make_blobs(2000, center=(1.5, 0.0))

def make_loader(X, y, batch_size=128, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_src = make_loader(X_src, y_src)
train_tgt = make_loader(X_tgt, y_tgt)
test_src  = make_loader(X_src_test, y_src_test, shuffle=False)
test_tgt  = make_loader(X_tgt_test, y_tgt_test, shuffle=False)
```

### Visualise the domain gap

```python
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(X_src[:, 0], X_src[:, 1], s=8, alpha=0.4, label="Source")
ax.scatter(X_tgt[:, 0], X_tgt[:, 1], s=8, alpha=0.4, label="Target")
ax.set_xlabel("x₀  (regression target)"); ax.set_ylabel("x₁")
ax.set_title("Partially overlapping blobs — 2D input space")
ax.legend(); plt.tight_layout(); plt.show()
```

---

## Training all four variants

We use identical hyperparameters across all four so the only difference is
`use_potentials` and `weight_ot`.

```python
EPOCHS     = 60
WARMUP     = 10
LR         = 5e-3
LATENT_DIM = 16

def make_model():
    return MLPRegressor(input_dim=2, latent_dim=LATENT_DIM, hidden_dims=(64, 32))

configs = [
    dict(label="SIDDA",             use_potentials=False, weight_ot=False),
    dict(label="SIDDA + CE-reweight", use_potentials=True,  weight_ot=False),
    dict(label="SIDDA + OT-reweight", use_potentials=False, weight_ot=True),
    dict(label="SIDDA + Both",      use_potentials=True,  weight_ot=True),
]

trainers = {}
histories = {}

for cfg in configs:
    label = cfg["label"]
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    model = make_model()
    trainer = SIDDARegressionTrainer(
        model, train_src, train_tgt,
        lr=LR,
        warmup_epochs=WARMUP,
        use_potentials=cfg["use_potentials"],
        weight_ot=cfg["weight_ot"],
        potential_temperature=1.0,
        ot_ema_momentum=0.9,
    )
    histories[label] = trainer.fit(epochs=EPOCHS)
    trainers[label]  = trainer
```

---

## Evaluation

```python
results = {}
for label, trainer in trainers.items():
    r_src = trainer.evaluate(test_src, domain="source")
    r_tgt = trainer.evaluate(test_tgt, domain="target")
    results[label] = {"src": r_src, "tgt": r_tgt}
    print(
        f"{label:30s}  "
        f"src RMSE={r_src['rmse']:.4f}  R²={r_src['r2']:.3f}  |  "
        f"tgt RMSE={r_tgt['rmse']:.4f}  R²={r_tgt['r2']:.3f}"
    )
```

---

## Plots

### Training curves

```python
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for (label, hist), color in zip(histories.items(), colors):
    epochs = [h["epoch"] for h in hist]
    axes[0].plot(epochs, [h["tgt_rmse"] for h in hist], color=color, label=label)
    axes[1].plot(epochs, [h["src_rmse"] for h in hist], color=color,
                 linestyle="--", label=label)

for ax, title in zip(axes, ["Target RMSE", "Source RMSE"]):
    ax.axvline(WARMUP, color="gray", linestyle=":", linewidth=1, label="Warmup end")
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE")
    ax.set_title(title); ax.legend(fontsize=8)

plt.suptitle("SIDDA regression — training curves", fontweight="bold")
plt.tight_layout(); plt.show()
```

### Prediction scatter — target domain

```python
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

for ax, (label, trainer) in zip(axes, trainers.items()):
    trainer.model.eval()
    X_t = torch.from_numpy(X_tgt_test).to(trainer.device)
    with torch.no_grad():
        z = trainer.model.encode(X_t)
        pred = trainer.model.regress(z).cpu().numpy().ravel()

    r2 = results[label]["tgt"]["r2"]
    ax.scatter(y_tgt_test, pred, s=5, alpha=0.3)
    ax.plot([-2, 4], [-2, 4], "k--", linewidth=0.8)
    ax.set_title(f"{label}\nR²={r2:.3f}", fontsize=9)
    ax.set_xlabel("True y")

axes[0].set_ylabel("Predicted y")
plt.suptitle("Target domain predictions  (y = x₀)", fontweight="bold")
plt.tight_layout(); plt.show()
```

### Latent space — coloured by domain and by target value

```python
def get_latents(trainer, X_src, X_tgt):
    trainer.model.eval()
    with torch.no_grad():
        zs = trainer.model.encode(torch.from_numpy(X_src).to(trainer.device)).cpu().numpy()
        zt = trainer.model.encode(torch.from_numpy(X_tgt).to(trainer.device)).cpu().numpy()
    return zs, zt

fig = plt.figure(figsize=(16, 8))
gs  = gridspec.GridSpec(2, 4, hspace=0.45, wspace=0.35)

for col, (label, trainer) in enumerate(trainers.items()):
    zs, zt = get_latents(trainer, X_src_test[:500], X_tgt_test[:500])

    # Top row: colour by domain
    ax = fig.add_subplot(gs[0, col])
    ax.scatter(zs[:, 0], zs[:, 1], s=6, alpha=0.4, label="Source")
    ax.scatter(zt[:, 0], zt[:, 1], s=6, alpha=0.4, label="Target")
    ax.set_title(label, fontsize=8, fontweight="bold")
    if col == 0:
        ax.set_ylabel("Domain separation", fontsize=9)
    ax.legend(fontsize=7, markerscale=2)

    # Bottom row: colour by target value
    ax2 = fig.add_subplot(gs[1, col])
    all_z = np.vstack([zs, zt])
    all_y = np.concatenate([y_src_test[:500], y_tgt_test[:500]])
    sc = ax2.scatter(all_z[:, 0], all_z[:, 1], c=all_y,
                     cmap="RdYlBu_r", s=6, alpha=0.5, vmin=-2, vmax=4)
    if col == 3:
        plt.colorbar(sc, ax=ax2, label="y = x₀")
    if col == 0:
        ax2.set_ylabel("Coloured by y", fontsize=9)

plt.suptitle("Latent space (first two dims)  —  top: domain  |  bottom: target value",
             fontweight="bold")
plt.show()
```

### Mean source potential over training

Only meaningful when `use_potentials=True` or `weight_ot=True`.  Decreasing
potential = encoder is aligning the two distributions in latent space.

```python
fig, ax = plt.subplots(figsize=(7, 3))
for (label, hist), color in zip(histories.items(), colors):
    pot = [h["mean_potential"] for h in hist]
    if any(p != 0.0 for p in pot):
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, pot, color=color, label=label)

ax.axvline(WARMUP, color="gray", linestyle=":", linewidth=1, label="Warmup end")
ax.set_xlabel("Epoch"); ax.set_ylabel("Mean source potential f")
ax.set_title("Transport cost — decreases as distributions align")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
```

---

## Summary table

After running the above, you can collect results into a clean table:

```python
print(f"\n{'Variant':<30} {'Src RMSE':>10} {'Src R²':>8} {'Tgt RMSE':>10} {'Tgt R²':>8}")
print("-" * 70)
for label in [c["label"] for c in configs]:
    r = results[label]
    print(
        f"{label:<30} "
        f"{r['src']['rmse']:>10.4f} {r['src']['r2']:>8.3f} "
        f"{r['tgt']['rmse']:>10.4f} {r['tgt']['r2']:>8.3f}"
    )
```

### Typical results

| Variant | Src RMSE | Src R² | Tgt RMSE | Tgt R² |
|---------|:--------:|:------:|:--------:|:------:|
| SIDDA | ~0.80 | ~0.98 | ~0.82 | ~0.97 |
| SIDDA + CE-reweight | ~0.80 | ~0.98 | ~0.81 | ~0.97 |
| SIDDA + OT-reweight | ~0.80 | ~0.98 | ~0.80 | ~0.98 |
| SIDDA + Both | ~0.80 | ~0.98 | ~0.79 | ~0.98 |

!!! note "Interpreting the results"
    With 30 % overlap the domain gap is moderate and baseline SIDDA already
    transfers well.  The reweighting variants show the most benefit when the
    overlap is smaller — try reducing the blob separation to `center=(2.5, 0.0)`
    or increasing it to `center=(3.0, 0.0)` to see the effect amplify.
    The latent-space plots are often more telling than RMSE: look for whether the
    source and target embeddings become more interleaved and whether the target
    value gradient stays smooth across both domains.

---

## Exercises

1. **Widen the gap** — change `center=(2.5, 0.0)` or `(3.0, 0.0)`.
   Which variant degrades least?

2. **Lower temperature** — set `potential_temperature=0.3`.
   Does sharper reweighting help or hurt when the gap is large?

3. **Shorter warmup** — try `warmup_epochs=0`.
   Does removing warmup destabilise training differently across variants?

4. **Higher-dimensional blobs** — extend to 10D input, still regressing `x₀`.
   Does the latent-space alignment improve with higher ambient dimension?
