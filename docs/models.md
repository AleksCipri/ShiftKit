# Models

`shiftkit.models` provides neural network architectures with a shared **encoder / classifier** interface. Splitting these two components is central to how domain adaptation methods operate — the encoder produces a latent representation `z`, and DA losses (e.g. MMD) are computed on `z` directly.

All models implement:

| Method | Signature | Description |
|--------|-----------|-------------|
| `encode` | `(x: Tensor) → Tensor` | Map input to latent vector `z ∈ ℝᵈ` |
| `classify` | `(z: Tensor) → Tensor` | Map latent vector to class logits |
| `forward` | `(x: Tensor) → Tensor` | `classify(encode(x))` — standard `nn.Module` interface |

---

## CNN

A small convolutional network designed for **1×28×28 inputs** (MNIST-like). Two conv-pool blocks feed into a fully-connected bottleneck that produces the latent vector.

```
Input (1×28×28)
  → Conv2d(1→32, k=3) + BN + ReLU + MaxPool  →  32×14×14
  → Conv2d(32→64, k=3) + BN + ReLU + MaxPool  →  64×7×7
  → Flatten → Linear(3136→256) → ReLU → Dropout
  → Linear(256→latent_dim) → ReLU              →  z ∈ ℝᵈ
  → Linear(latent_dim→num_classes)              →  logits
```

```python
from shiftkit.models import CNN

model = CNN(latent_dim=128, num_classes=10, dropout=0.3)

z      = model.encode(x)    # (B, 128)
logits = model.classify(z)  # (B, 10)
logits = model(x)           # equivalent
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent_dim` | `int` | `128` | Dimensionality of the bottleneck embedding |
| `num_classes` | `int` | `10` | Number of output classes |
| `dropout` | `float` | `0.3` | Dropout probability before the final FC layer |

---

## MLP

A fully-connected network that flattens the input and passes it through configurable hidden layers before the bottleneck.

```
Input (1×28×28 → flattened 784)
  → Linear(784→h₁) + ReLU + Dropout
  → Linear(h₁→h₂)  + ReLU + Dropout
  → ...
  → Linear(hₙ→latent_dim) + ReLU    →  z ∈ ℝᵈ
  → Linear(latent_dim→num_classes)   →  logits
```

```python
from shiftkit.models import MLP

model = MLP(latent_dim=128, num_classes=10, hidden_dims=(512, 256), dropout=0.3)

z      = model.encode(x)    # (B, 128)
logits = model.classify(z)  # (B, 10)
logits = model(x)           # equivalent
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent_dim` | `int` | `128` | Dimensionality of the bottleneck embedding |
| `num_classes` | `int` | `10` | Number of output classes |
| `hidden_dims` | `Tuple[int, ...]` | `(512, 256)` | Sizes of hidden layers before the bottleneck |
| `dropout` | `float` | `0.3` | Dropout probability after each hidden layer |

---

## Choosing between CNN and MLP

| | CNN | MLP |
|--|-----|-----|
| Input type | 2-D images (preserves spatial structure) | Any flattened vector |
| Inductive bias | Translation equivariance | None |
| Parameters (default) | ~856 K | ~560 K |
| Speed | Slightly faster on GPU | Slightly faster on CPU |

For image inputs, **CNN is recommended**. MLP is useful when inputs are already feature vectors.

---

## Using a custom model

Any model that exposes `.encode(x)` and `.classify(z)` can be used with `MMDTrainer` and `SourceOnlyTrainer`:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU())
        self.classifier = nn.Linear(64, 10)

    def encode(self, x):
        return self.encoder(x)

    def classify(self, z):
        return self.classifier(z)

    def forward(self, x):
        return self.classify(self.encode(x))
```
