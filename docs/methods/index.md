# Methods

`shiftkit.methods` provides domain adaptation training loops. All trainers record
identical per-epoch history dicts so their results can be directly compared.

---

## Available methods

| Trainer | DA mechanism | Key parameter | Page |
|---------|-------------|---------------|------|
| [`SourceOnlyTrainer`](source_only.md) | No adaptation (baseline) | — | [→](source_only.md) |
| [`MMDTrainer`](mmd.md) | Latent distribution matching via MMD | `mmd_weight` λ | [→](mmd.md) |
| [`DANNTrainer`](dann.md) | Adversarial discriminator + GRL | `domain_weight` λ | [→](dann.md) |
| [`LMMDTrainer`](lmmd.md) | Per-class subdomain alignment via local MMD | `num_classes` | [→](lmmd.md) |
| [`CORALTrainer`](coral.md) | Covariance alignment (second-order statistics) | `coral_weight` λ | [→](coral.md) |
| [`SIDDATrainer`](sidda.md) | Sinkhorn optimal transport + learnable η weights | `warmup_epochs` | [→](sidda.md) |
| [`KLIEPTrainer`](kliep.md) | Instance-based importance weighting via density ratio estimation | `n_centers`, `weight_clip` | [→](kliep.md) |
| Custom | Your own method | — | [→](custom.md) |

All trainers share the same interface:

```python
trainer = AnyTrainer(model, source_loader, target_loader, ...)
history = trainer.fit(epochs=10)
result  = trainer.evaluate(test_loader, domain="target-test")
```

---

## Method comparison

| | Source Only | MMD | LMMD | CORAL | DANN | SIDDA | KLIEP |
|--|:-----------:|:---:|:----:|:-----:|:----:|:-----:|:-----:|
| **DA family** | — | Feature-based | Feature-based | Feature-based | Feature-based | Feature-based | Instance-based |
| **Alignment target** | None | Full distribution | Per-class subdomains | Covariance matrix | Domain labels | Optimal transport plan | Sample weights (density ratio) |
| **What is matched** | — | All moments (via kernel) | Class-conditional moments | 2nd-order statistics | Domain membership | Entire marginal distribution | p_tgt(x) / p_src(x) |
| **Kernel required** | No | Yes — RBF, bandwidth σ | Yes — RBF, bandwidth σ | No | No | No (Sinkhorn entropic OT) | Yes — RBF in input space |
| **Needs source labels** | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **Needs target labels** | No | No | Pseudo-labels (soft) | No | No | No | No |
| **Adversarial training** | No | No | No | No | Yes (GRL) | No | No |
| **Learnable loss weights** | No | No | No | No | No | Yes (η₁, η₂) | No |
| **Alignment cost** | — | Every batch | Every batch | Every batch | Every batch | Every batch | Once at init |
| **Model interface** | `forward()` | `encode()` + `classify()` | `encode()` + `classify()` | `encode()` + `classify()` | `encode()` + `classify()` | full SIDDA interface | `forward()` only |
| **Computation per batch** | O(n·d) | O(n²) kernel matrices | O(n²) per class | O(n·d²) covariance | O(n·d) + discriminator | O(n²) Sinkhorn iterations | O(n·m) weight lookup |
| **Key hyperparameter** | — | `mmd_weight` λ | `lmmd_weight` λ | `coral_weight` λ | `domain_weight` λ | `warmup_epochs`, blur schedule | `n_centers`, `weight_clip` |
| **Warmup supported** | — | Yes | Yes | Yes | Yes | Mandatory | No |
| **Covariate shift assumption** | No | No | No | No | No | No | Yes |
| **Best suited for** | Reference baseline | General distribution shift | Class-level shift with label imbalance | Shift in feature scale / correlation | Strong covariate shift with large batches | Unknown shift type; automatically reweights objectives | Covariate shift on tabular / low-dim data |

> **Choosing a method:** Start with the Source-Only baseline to measure the domain gap.
> For most tasks, MMD or CORAL is a fast, strong first attempt.
> Use LMMD when class distributions differ across domains.
> Use DANN when the shift is severe and batch sizes are large enough to train the discriminator.
> Use SIDDA when you want automatic loss balancing without manual λ tuning.
> Use KLIEP when the covariate shift assumption holds and you prefer instance reweighting over feature alignment — especially effective on tabular data.

---

## Shared history format

Every `fit()` call returns a `list[dict]` with one entry per epoch:

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
| `total_loss` | `float` | Total combined loss |
| `src_acc` | `float` | Source domain accuracy |
| `tgt_acc` | `float` | Target domain accuracy (tracked, not directly optimised) |

---





