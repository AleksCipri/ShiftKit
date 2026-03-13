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
| [`SIDDATrainer`](sidda.md) | Sinkhorn optimal transport + learnable η weights | `warmup_epochs` | [→](sidda.md) |

All trainers share the same interface:

```python
trainer = AnyTrainer(model, source_loader, target_loader, ...)
history = trainer.fit(epochs=10)
result  = trainer.evaluate(test_loader, domain="target-test")
```

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

## Comparing methods

Pass a `{label: history}` dict to overlay multiple runs on the same plot:

```python
from shiftkit.diagnostics import plot_training_history

plot_training_history({
    "Source Only": history_baseline,
    "MMD":         history_mmd,
    "DANN":        history_dann,
    "SIDDA":       history_sidda,
})
```

The right panel shows source accuracy (solid) and target accuracy (dashed) —
the gap between them quantifies the remaining domain shift.

!!! tip "Tuning λ"
    Start at `1.0`. Reduce to `0.1`–`0.5` if source accuracy degrades sharply;
    increase to `2.0`–`5.0` if source and target embeddings remain separated
    after training.
