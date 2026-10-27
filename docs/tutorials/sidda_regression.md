# Tutorial 8 — SIDDA Regression with Latent-Space Instance Reweighting

This tutorial demonstrates `SIDDARegressionTrainer` on a synthetic 2D
**partially-overlapping blobs** problem, comparing four variants of SIDDA:

| Variant | `use_potentials` | `weight_ot` | What changes |
|---------|:---:|:---:|---|
| **SIDDA** | ✗ | ✗ | Baseline — uniform measures, mean MSE |
| **SIDDA + CE-reweight** | ✓ | ✗ | Per-sample MSE weighted by softmax(−f/τ) of the batch's source potentials |
| **SIDDA + OT-reweight** | ✗ | ✓ | Sinkhorn measures reweighted by softmax(−f/τ), softmax(−g/τ) of the same batch |
| **SIDDA + Both** | ✓ | ✓ | Both reweightings |

The two reweighting options are experimental heuristics (see the
[SIDDA method page](../methods/sidda.md#potential-based-reweighting-experimental));
this tutorial shows how to run them, not that they help.

---

## The problem

Two 2D Gaussian blobs — one per domain — partially overlap.
Each coordinate has **standard deviation 0.8** (covariance 0.64·I). The
regression target is the **first feature** `x₀`, which spans the shift direction.

```
Source: N([0, 0], 0.64·I)     Target: N([1.5, 0], 0.64·I)
        (per-coordinate SD 0.8)        (centres 1.5 apart ≈ 1.9 SD)
```

A model trained only on source sees a range of `x₀` that is systematically lower
than the target's range. Because `y = x₀` is the same function in both domains,
this is a covariate shift: a source model that learns the identity well can
already extrapolate, and the source-only reference below is a strong baseline.
Aligning the latent distributions can also *hurt*, because matching the two
latent marginals can pull apart the mapping from latent to `y`.

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
    )
    histories[label] = trainer.fit(epochs=EPOCHS)
    trainers[label]  = trainer
```

As a reference, train the same architecture on source only:

```python
from shiftkit import SourceOnlyRegressionTrainer

ref = SourceOnlyRegressionTrainer(make_model(), train_src, train_tgt, lr=LR)
ref.fit(epochs=EPOCHS)
r_tgt = ref.evaluate(test_tgt, domain="target")
print(f"Source-only reference: tgt RMSE={r_tgt['rmse']:.4f}  R²={r_tgt['r2']:.3f}")
```

The history's `tgt_rmse`/`tgt_r2` use the target training labels for
monitoring only (evaluation mode, after each step); they never enter the loss.

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

### Latent space — t-SNE coloured by domain and by target value

t-SNE projects all `LATENT_DIM` dimensions to 2D jointly for source and target,
so the proximity structure reflects the full embedding rather than just two
arbitrarily chosen axes.

```python
from sklearn.manifold import TSNE

N_PLOT = 500   # points per domain — keep small for fast t-SNE

def get_latents(trainer, X_src, X_tgt):
    trainer.model.eval()
    with torch.no_grad():
        zs = trainer.model.encode(
            torch.from_numpy(X_src).to(trainer.device)).cpu().numpy()
        zt = trainer.model.encode(
            torch.from_numpy(X_tgt).to(trainer.device)).cpu().numpy()
    return zs, zt

fig = plt.figure(figsize=(16, 8))
gs  = gridspec.GridSpec(2, 4, hspace=0.45, wspace=0.35)

for col, (label, trainer) in enumerate(trainers.items()):
    zs, zt = get_latents(trainer, X_src_test[:N_PLOT], X_tgt_test[:N_PLOT])

    # Fit t-SNE on source + target jointly so both share the same embedding
    z_all = np.vstack([zs, zt])
    z2d   = TSNE(n_components=2, perplexity=30, random_state=42,
                 init="pca", learning_rate="auto").fit_transform(z_all)
    zs2d, zt2d = z2d[:N_PLOT], z2d[N_PLOT:]

    # Top row: colour by domain
    ax = fig.add_subplot(gs[0, col])
    ax.scatter(zs2d[:, 0], zs2d[:, 1], s=6, alpha=0.5,
               color="#4C72B0", label="Source")
    ax.scatter(zt2d[:, 0], zt2d[:, 1], s=6, alpha=0.5,
               color="#DD8452", label="Target")
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0:
        ax.set_ylabel("By domain", fontsize=9)
    if col == 3:
        ax.legend(fontsize=7, markerscale=2, loc="upper right")

    # Bottom row: colour by target value y = x₀
    ax2 = fig.add_subplot(gs[1, col])
    all_y = np.concatenate([y_src_test[:N_PLOT], y_tgt_test[:N_PLOT]])
    sc = ax2.scatter(z2d[:, 0], z2d[:, 1], c=all_y,
                     cmap="RdYlBu_r", s=6, alpha=0.6, vmin=-2, vmax=4)
    ax2.set_xticks([]); ax2.set_yticks([])
    if col == 0:
        ax2.set_ylabel("By target value y", fontsize=9)
    if col == 3:
        plt.colorbar(sc, ax=ax2, label="y = x₀", shrink=0.8)

plt.suptitle("t-SNE latent space  —  top: domain  |  bottom: target value y",
             fontweight="bold")
plt.show()
```

!!! note "Reading the t-SNE plots"
    **Top row (by domain):** interleaved source (blue) and target (orange)
    points indicate that the latent marginals overlap.

    **Bottom row (by target value y):** a smooth colour gradient across both
    domains is consistent with the encoder representing y, but it is only a
    visual hint: t-SNE distorts distances, and a smooth gradient does not show
    that predictions on the target are accurate. Use labelled target test
    metrics for that.

### Mean source potential over training

Only recorded when `use_potentials=True` or `weight_ot=True`. The mean source
potential f is **not** the Sinkhorn divergence (that is mean(f) + mean(g)), and
it depends on σ, which changes every batch, so it is not a monotone measure of
alignment. Plot `da_loss` if you want the divergence itself.

```python
fig, ax = plt.subplots(figsize=(7, 3))
for (label, hist), color in zip(histories.items(), colors):
    pot = [h["mean_potential"] for h in hist]
    if any(p != 0.0 for p in pot):
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, pot, color=color, label=label)

ax.axvline(WARMUP, color="gray", linestyle=":", linewidth=1, label="Warmup end")
ax.set_xlabel("Epoch"); ax.set_ylabel("Mean source potential f")
ax.set_title("Mean source potential f (not the Sinkhorn divergence)")
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

### Results from one run

The table below was produced by running exactly the code above on CPU with
`torch.manual_seed(42)` / `np.random.seed(42)`, 60 epochs (10 warmup), plus the
source-only reference. This is **one seed**. Label variance is 0.664 (source
test) and 0.644 (target test), so R² = 1 − MSE/0.644 on the target.

| Variant | Src RMSE | Src R² | Tgt RMSE | Tgt R² |
|---------|:--------:|:------:|:--------:|:------:|
| SIDDA | 0.103 | 0.984 | 0.898 | −0.253 |
| SIDDA + CE-reweight | 0.183 | 0.949 | 1.364 | −1.888 |
| SIDDA + OT-reweight | 0.068 | 0.993 | 0.414 | 0.734 |
| SIDDA + Both | 0.050 | 0.996 | 0.152 | 0.964 |
| Source-only (reference) | 0.055 | 0.995 | 0.144 | 0.968 |

!!! warning "Interpreting the results"
    Results vary strongly between seeds. Repeating the same script with seeds
    0–3 gave target RMSE 0.33–1.08 for SIDDA, 0.23–0.35 for CE-reweight,
    0.07–0.28 for OT-reweight, 0.04–0.27 for Both, and 0.16–0.23 for source-only.
    The ranking of the variants changed from seed to seed, so this example does
    **not** show that any reweighting option is better, and on this covariate
    shift plain SIDDA was often worse than training on source only. Compare
    methods over several seeds, with a labelled target test set, before drawing
    conclusions. Your numbers may differ with other hardware or library
    versions.

---

## Exercises

1. **Widen the gap** — move the target centre further away, e.g.
   `center=(2.5, 0.0)` or `(3.0, 0.0)` (this *increases* the separation and
   reduces the overlap). Which variant degrades least, over several seeds?

2. **Lower temperature** — set `potential_temperature=0.3`.
   Does sharper reweighting help or hurt when the gap is large?

3. **Shorter warmup** — try `warmup_epochs=0`.
   Does removing warmup destabilise training differently across variants?

4. **Higher-dimensional blobs** — extend to 10D input, still regressing `x₀`.
   Does the latent-space alignment improve with higher ambient dimension?
