# Tutorial 4 — Registering a custom dataset

You can plug in any pair of PyTorch datasets without modifying the library.
`DataManager.register()` accepts a factory function that returns a
`(source_loader, target_loader)` tuple.

---

## Factory signature

```python
def my_factory(root, batch_size, train, num_workers, **kwargs):
    ...
    return source_loader, target_loader
```

| Argument | Provided by | Description |
|----------|-------------|-------------|
| `root` | `DataManager` | Root directory from `DataManager(root=...)` |
| `batch_size` | `DataManager` | Batch size from `DataManager(batch_size=...)` |
| `train` | `dm.load(train=...)` | `True` for train split, `False` for test |
| `num_workers` | `DataManager` | Number of DataLoader workers |
| `**kwargs` | `dm.load(...)` | Any extra keyword arguments from the `load()` call |

---

## Example — tensor dataset pair

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from shiftkit.data import DataManager

def tabular_factory(root, batch_size, train, num_workers, **kwargs):
    n = 1000 if train else 200
    X_src = torch.randn(n, 32)
    y_src = (X_src[:, 0] > 0).long()            # binary label

    X_tgt = torch.randn(n, 32) + 0.5            # shifted mean
    y_tgt = (X_tgt[:, 0] > 0).long()

    src_loader = DataLoader(
        TensorDataset(X_src, y_src),
        batch_size=batch_size, shuffle=train, num_workers=num_workers,
    )
    tgt_loader = DataLoader(
        TensorDataset(X_tgt, y_tgt),
        batch_size=batch_size, shuffle=train, num_workers=num_workers,
    )
    return src_loader, tgt_loader

DataManager.register("tabular_shift", tabular_factory)

dm = DataManager(batch_size=64)
train_src, train_tgt = dm.load("tabular_shift", train=True)
test_src,  test_tgt  = dm.load("tabular_shift", train=False)
```

Use a model whose input width matches the 32 features, e.g.
`MLP(input_dim=32, latent_dim=16, num_classes=2)` (the `MLP` default is 784).

!!! note "Use separate generators per split"
    The factory above draws train and test data from the global RNG, so the two
    calls produce different samples only by chance of RNG state. For reproducible,
    independent splits, seed a `torch.Generator` per split (e.g. seed 0 for train,
    1 for test) and pass it to `torch.randn(..., generator=g)`.

!!! note "Unlabelled target"
    In unsupervised DA the target training loader only needs inputs; the
    built-in trainers accept target batches that are `x` alone (the history's
    `tgt_acc` is then `nan`, i.e. not available). Keep target labels for evaluation loaders.

---

## Passing extra kwargs

Any keyword argument passed to `dm.load()` is forwarded to your factory:

```python
def noisy_factory(root, batch_size, train, num_workers, noise_std=0.1, **kwargs):
    ...

DataManager.register("my_noisy", noisy_factory)

# Override noise level at load time:
train_src, train_tgt = dm.load("my_noisy", train=True, noise_std=0.5)
```

---

## Listing available datasets

```python
print(DataManager.available())
# ['mnist_noisy_mnist', 'synthetic_graphs', 'tabular_shift', ...]
```
