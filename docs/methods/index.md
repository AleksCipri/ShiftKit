# Methods

`shiftkit.methods` provides domain adaptation training loops. Every trainer
returns a per-epoch list of history dicts. The classification trainers share
the core keys `epoch`, `ce_loss`, `total_loss`, `src_acc` and `tgt_acc` and add
method-specific keys; regression trainers report errors (e.g. `src_rmse`)
instead of accuracies. The schemas are therefore comparable on their shared
keys, not identical.

ShiftKit methods fall into two broad families:

---

## Feature-based methods

Feature-based methods work by transforming the model's learned representations so that source and target features become indistinguishable. The encoder is trained jointly on a supervised task loss (cross-entropy on source labels) and a domain alignment loss that penalises differences between the source and target latent distributions. Because alignment happens in latent space, these methods require a model with a separate `encode()` step and work regardless of the input modality.

| Trainer | DA mechanism | Key parameter | Page |
|---------|-------------|---------------|------|
| [`SourceOnlyTrainer`](source_only.md) | No adaptation (baseline) | — | [→](source_only.md) |
| [`MMDTrainer`](mmd.md) | Latent distribution matching via MMD | `mmd_weight` λ | [→](mmd.md) |
| [`LMMDTrainer`](lmmd.md) | Per-class subdomain alignment via local MMD | `lmmd_weight` λ | [→](lmmd.md) |
| [`CORALTrainer`](coral.md) | Covariance alignment (second-order statistics) | `coral_weight` λ | [→](coral.md) |
| [`DANNTrainer`](dann.md) | Adversarial discriminator + GRL | `domain_weight` λ | [→](dann.md) |
| [`SIDDATrainer`](sidda.md) | Sinkhorn optimal transport + learnable η weights | `warmup_epochs` | [→](sidda.md) |

---

## Instance-based methods

Instance-based methods do not modify the feature space. Instead, they estimate how much more or less likely each source sample is under the target distribution and reweight the training loss accordingly. This approach is theoretically grounded under the **covariate shift** assumption — that the label conditionals are the same across domains (`p_src(y|x) = p_tgt(y|x)`) while only the input marginals differ. These methods require only a standard `forward()` interface and compute importance weights once before training begins.

| Trainer | DA mechanism | Key parameter | Page |
|---------|-------------|---------------|------|
| [`KLIEPTrainer`](kliep.md) | Importance weighting via density ratio estimation (gradient ascent) | `n_centers`, `weight_clip` | [→](kliep.md) |
| [`KMMTrainer`](kmm.md) | Importance weighting via kernel mean matching (QP) | `B`, `weight_clip` | [→](kmm.md) |

All trainers share the same interface:

```python
trainer = AnyTrainer(model, source_loader, target_loader, ...)
history = trainer.fit(epochs=10)
result  = trainer.evaluate(test_loader, domain="target-test")
```

The eight registered classification trainers (`TrainerRegistry.available()`)
are all `BaseTrainer` instances: SourceOnly, MMD, LMMD, CORAL and DANN subclass
it directly; SIDDA, KLIEP and KMM are registered as virtual subclasses.

**Target labels.** For SourceOnly, MMD, LMMD, CORAL and DANN, target loaders
may yield `x` (or `(x,)`) without labels; then `tgt_acc` is `nan`. If labels are
present they are used only for monitoring, never for gradients. Other trainers
may still expect `(x, y)` target batches — see their pages.

**Epochs and monitoring (SourceOnly, MMD, LMMD, CORAL, DANN).** One epoch
covers the longer of the source and target loaders (the shorter one restarts);
empty loaders raise `ValueError`. `src_acc` uses training-mode predictions on
each batch before its update; `tgt_acc` is measured in evaluation mode after the
update, so monitoring never changes BatchNorm statistics. Trainers do not seed
random number generators; call `shiftkit.utils.set_seed(seed)` before building the model and
loaders for repeatable runs. It seeds Python, NumPy and PyTorch and asks PyTorch for
deterministic GPU kernels (without which the same seed can give different networks on a GPU).

---

## Method comparison

| | Source Only | MMD | LMMD | CORAL | DANN | SIDDA | KLIEP | KMM |
|--|:-----------:|:---:|:----:|:-----:|:----:|:-----:|:-----:|:---:|
| **DA family** | — | Feature-based | Feature-based | Feature-based | Feature-based | Feature-based | Instance-based | Instance-based |
| **Alignment target** | None | Full distribution | Per-class subdomains | Covariance matrix | Domain labels | Optimal transport plan | Sample weights (density ratio) | Sample weights (kernel mean matching) |
| **What is matched** | — | Kernel mean embeddings (all moments only for a characteristic kernel and in the large-sample limit) | Class-conditional kernel means (pseudo-labelled target) | Covariances only (means removed) | Domain membership | Entire marginal distribution | p_tgt(x) / p_src(x) | Weighted source vs target kernel mean (input space) |
| **Kernel required** | No | Yes — RBF, bandwidth σ | Yes — RBF, bandwidth σ | No | No | No (Sinkhorn entropic OT) | Yes — RBF in input space | Yes — RBF in input space |
| **Needs source labels** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **Needs target labels** | No | No | Pseudo-labels (soft) | No | No | No | No | No |
| **Adversarial training** | No | No | No | No | Yes (GRL) | No | No | No |
| **Learnable loss weights** | No | No | No | No | No | Yes (η₁, η₂) | No | No |
| **Alignment cost** | — | Every batch | Every batch | Every batch | Every batch | Every batch | Once at init | Once at init (QP) |
| **Model interface** | `forward()` | `encode()` + `classify()` | `encode()` + `classify()` | `encode()` + `classify()` | `encode()` + `classify()` | full SIDDA interface | `forward()` only | `forward()` only |
| **Computation per batch** | O(n·d) | O(n²) kernel matrices | O(n²) per class | O(n·d²) covariance | O(n·d) + discriminator | O(n²) Sinkhorn iterations | O(n·m) weight lookup | O(n) weight lookup |
| **Key hyperparameter** | — | `mmd_weight` λ | `lmmd_weight` λ | `coral_weight` λ | `domain_weight` λ | `warmup_epochs`, blur schedule | `n_centers`, `weight_clip` | `B`, `sigma`, `weight_clip` |
| **Warmup supported** | — | Yes | Yes | Yes | Yes | Mandatory | No | No |
| **Covariate shift assumption** | No | No | No | No | No | No | Yes | Yes |
| **Best suited for** | Reference baseline | General distribution shift | Class-level shift with label imbalance | Shift in feature scale / correlation | Strong covariate shift with large batches | Unknown shift type; automatically reweights objectives | Covariate shift on tabular / low-dim data | Covariate shift on small tabular / low-dim data |

> **Choosing a method:** Start with the Source-Only baseline to measure the domain gap.
> For most tasks, MMD or CORAL is a fast, strong first attempt.
> Use LMMD when class distributions differ across domains.
> Use DANN when the shift is severe and batch sizes are large enough to train the discriminator.
> Use SIDDA when you want automatic loss balancing without manual λ tuning.
> Use KLIEP when the covariate shift assumption holds and you prefer instance reweighting over feature alignment — especially effective on tabular data.

---

## Shared history format

Every `fit()` call returns a `list[dict]` with one entry per epoch. The table
lists keys used across the classification trainers; each trainer reports only
its own subset (see each method page), e.g. `domain_loss` appears only for
DANN, and LMMD/CORAL add `lmmd_loss`/`coral_loss`:

| Key | Type | Description |
|-----|------|-------------|
| `epoch` | `int` | Epoch index (1-based) |
| `ce_loss` | `float` | Cross-entropy loss |
| `mmd_loss` | `float` | MMD² loss (`0.0` if not applicable) |
| `domain_loss` | `float` | Adversarial domain loss (`0.0` if not applicable) |
| `da_loss` | `float` | Sinkhorn DA loss (`0.0` if not applicable) |
| `eta1` | `float` | Learned CE weight η₁ (SIDDA only) |
| `eta2` | `float` | Learned DA weight η₂ (SIDDA only) |
| `sigma` | `float` | Sinkhorn blur used (SIDDA only) |
| `mean_potential` | `float` | Mean source Kantorovich potential (SIDDA with `use_potentials=True` only) |
| `total_loss` | `float` | Total combined loss |
| `src_acc` | `float` | Source training accuracy (see each method page for when it is measured) |
| `tgt_acc` | `float` | Target accuracy (diagnostic, not optimised; `nan` when target labels are absent, where supported) |

---





