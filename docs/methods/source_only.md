# SourceOnlyTrainer

A **no-adaptation baseline** that trains exclusively on labelled source data
using cross-entropy loss. No domain alignment is applied.

Use this alongside an adaptation method (MMD, DANN, …) to quantify how much
benefit domain adaptation provides on your task.

Target batches are only used for `tgt_acc` monitoring, which runs in
evaluation mode with gradients disabled: target data does not update
BatchNorm running statistics or any other model state. (In earlier versions
monitoring ran in training mode and target batches updated BatchNorm
statistics.)

---

## Usage

```python
from shiftkit.methods import SourceOnlyTrainer

trainer = SourceOnlyTrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,   # used for tgt_acc monitoring only; labels optional
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
| `target_loader` | `DataLoader` | — | Target DataLoader yielding `x` or `(x, y)`; labels, if present, are used only for `tgt_acc` |
| `lr` | `float` | `1e-3` | Adam learning rate |
| `device` | `str \| None` | `None` | `'cuda'`, `'mps'`, or `'cpu'`; auto-detected if `None` |

---

## `fit(epochs=10)`

Train for `epochs` epochs using cross-entropy on source data only. One
epoch covers the longer of the two loaders (the shorter one restarts); empty
loaders raise `ValueError`. Trainers do not seed random number generators.

**Returns:** `list[dict]` — one dict per epoch:

| Key | Description |
|-----|-------------|
| `epoch` | Epoch number (1-indexed) |
| `ce_loss` | Mean cross-entropy loss |
| `mmd_loss` | Always `0.0` (for history-format compatibility) |
| `total_loss` | Same as `ce_loss` |
| `src_acc` | Source training accuracy: training-mode predictions on each batch, taken *before* that batch's update |
| `tgt_acc` | Target accuracy in evaluation mode, after each update; `nan` without target labels (purely diagnostic) |

---

## `evaluate(loader, domain="source")`

Compute classification accuracy on any labelled DataLoader, in evaluation
mode. An empty or unlabelled loader raises `ValueError`.

**Returns:** `dict` with keys `domain` (str), `accuracy` (float), `n_samples` (int).
