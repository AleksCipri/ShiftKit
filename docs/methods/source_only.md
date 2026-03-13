# SourceOnlyTrainer

A **no-adaptation baseline** that trains exclusively on labelled source data
using cross-entropy loss. No domain alignment is applied.

Use this alongside an adaptation method (MMD, DANN, …) to quantify how much
benefit domain adaptation provides on your task.

---

## Usage

```python
from shiftkit.methods import SourceOnlyTrainer

trainer = SourceOnlyTrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,   # used for tgt_acc tracking only
    lr=1e-3,
)
history = trainer.fit(epochs=10)

result = trainer.evaluate(test_tgt, domain="target-test")
print(f"Target accuracy: {result['accuracy']*100:.1f}%")
```

---

## Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | Any model with a standard `forward()` method |
| `source_loader` | `DataLoader` | — | Labelled source DataLoader |
| `target_loader` | `DataLoader` | — | Target DataLoader (labels used for `tgt_acc` tracking only) |
| `lr` | `float` | `1e-3` | Adam learning rate |
| `device` | `str \| None` | `None` | `'cuda'`, `'mps'`, or `'cpu'`; auto-detected if `None` |

---

## `fit(epochs=10)`

Train for `epochs` epochs using cross-entropy on source data only.

**Returns:** `list[dict]` — one dict per epoch:

| Key | Description |
|-----|-------------|
| `epoch` | Epoch number (1-indexed) |
| `ce_loss` | Mean cross-entropy loss |
| `mmd_loss` | Always `0.0` (for history-format compatibility) |
| `total_loss` | Same as `ce_loss` |
| `src_acc` | Source domain training accuracy |
| `tgt_acc` | Target domain accuracy (no gradient signal — purely diagnostic) |

---

## `evaluate(loader, domain="source")`

Compute classification accuracy on any labelled DataLoader.

**Returns:** `dict` with keys `domain` (str), `accuracy` (float), `n_samples` (int).
