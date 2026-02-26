# Data

`shiftkit.data` provides dataset loading and paired source/target DataLoader creation.

---

## DataManager

Central hub for loading source/target domain data. Maintains a registry of dataset-pair factories and returns paired `DataLoader` objects.

```python
from shiftkit.data import DataManager

dm = DataManager(root="./data", batch_size=64)
train_src, train_tgt = dm.load("mnist_noisy_mnist", train=True)
test_src,  test_tgt  = dm.load("mnist_noisy_mnist", train=False)
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root` | `str` | `"./data"` | Root directory where datasets are downloaded |
| `batch_size` | `int` | `64` | Batch size for both DataLoaders |
| `num_workers` | `int` | `0` | Number of DataLoader worker processes |

### Methods

#### `load(name, train=True, **kwargs)`

Return `(source_loader, target_loader)` for the named dataset pair.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Registered dataset key, e.g. `"mnist_noisy_mnist"` |
| `train` | `bool` | `True` | Load training split if `True`, test split if `False` |
| `**kwargs` | | | Forwarded to the factory (e.g. `noise_std=0.5`) |

**Returns:** `(DataLoader, DataLoader)` — source loader, target loader

**Raises:** `ValueError` if `name` is not registered.

#### `register(name, factory)` *(static)*

Register a custom dataset-pair factory.

```python
def my_factory(root, batch_size, train, num_workers, **kwargs):
    source_ds = ...  # your source dataset
    target_ds = ...  # your target dataset
    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=train)
    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=train)
    return source_loader, target_loader

DataManager.register("my_pair", my_factory)

# then use it:
dm = DataManager()
src, tgt = dm.load("my_pair")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Key to register under |
| `factory` | `callable` | Function with signature `(root, batch_size, train, num_workers, **kwargs)` → `(DataLoader, DataLoader)` |

#### `available()` *(static)*

Return a list of all registered dataset-pair names.

```python
print(DataManager.available())
# ['mnist_noisy_mnist', 'my_pair', ...]
```

---

## NoisyMNIST

A `torch.utils.data.Dataset` that wraps `torchvision.datasets.MNIST` and adds per-sample Gaussian noise. Used as the built-in synthetic target domain.

```python
from shiftkit.data.datasets import NoisyMNIST

ds = NoisyMNIST(root="./data", train=True, noise_std=0.3)
img, label = ds[0]   # img is a clipped noisy tensor in [0, 1]
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root` | `str` | — | Path to dataset directory |
| `train` | `bool` | `True` | Training split if `True`, test split if `False` |
| `noise_std` | `float` | `0.3` | Standard deviation of additive Gaussian noise |
| `transform` | `callable` | `None` | Additional transforms applied after noise injection |
| `download` | `bool` | `True` | Download the dataset if not present |

!!! note
    Noise is injected as `img = (img + N(0, noise_std²)).clamp(0, 1)` each time `__getitem__` is called, so each epoch sees different noise realisations.

---

## Built-in dataset pairs

| Key | Source | Target | Extra kwargs |
|-----|--------|--------|-------------|
| `"mnist_noisy_mnist"` | `torchvision.MNIST` | `NoisyMNIST` | `noise_std` (default `0.3`) |
