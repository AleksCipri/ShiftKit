"""
Neural network architectures for domain adaptation.

All models expose:
    encode(x)  -> latent tensor  (used by DA methods)
    classify(z)-> logit tensor   (used for supervised loss)
    forward(x) -> logits         (encode + classify, standard nn.Module interface)

Two-column regression heads (mean, variance)
--------------------------------------------
Every ShiftKit regression head built with ``predict_var=True`` (``MLPRegressor``,
``GNN``) returns two columns per sample, interpreted by
:func:`mean_var_from_head`:

* column 0 — predicted mean ``mu``;
* column 1 — an unconstrained raw value ``r`` mapped to a variance
  ``var = softplus(r) + eps`` (``eps = 1e-6`` by default).

Column 1 is neither a standard deviation nor a log-variance.  All Gaussian
trainers and ``predict_mean_var`` use this single mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

#: Default variance floor added after the softplus in the 2-column head contract.
VAR_EPS = 1e-6


def mean_var_from_head(out: torch.Tensor, eps: float = VAR_EPS) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map a ``[..., 2]`` regression-head output to ``(mean, var)``.

    ``mean = out[..., 0]``; ``var = softplus(out[..., 1]) + eps``.  This is the
    one output contract for 2-column heads in ShiftKit.
    """
    if out.dim() < 1 or out.size(-1) != 2:
        raise ValueError(
            "expected a 2-column (mean, raw-variance) head output, got shape "
            f"{tuple(out.shape)}; build the model with predict_var=True")
    return out[..., 0], F.softplus(out[..., 1]) + eps


class MLP(nn.Module):
    """
    Fully-connected encoder + linear classifier.

    Architecture
    ------------
    input (flattened, ``input_dim`` features; default 784 = 1×28×28)
        -> hidden layers -> latent_dim -> num_classes

    Parameters
    ----------
    latent_dim  : size of the bottleneck embedding
    num_classes : number of output classes
    hidden_dims : sizes of hidden layers before the bottleneck
    dropout     : dropout probability applied after each hidden layer
    input_dim   : number of input features after flattening (default 784,
                  i.e. 1×28×28 MNIST-like images)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 10,
        hidden_dims: Tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
        input_dim: int = 28 * 28,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True),
                       nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))
        layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.view(x.size(0), -1))

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(x))


class CNN(nn.Module):
    """
    Small convolutional encoder + linear classifier.

    Architecture
    ------------
    Conv block x2 -> flatten -> FC -> latent_dim -> num_classes

    Designed for 1×28×28 inputs (MNIST-like).

    Parameters
    ----------
    latent_dim  : size of the bottleneck embedding
    num_classes : number of output classes
    dropout     : dropout probability before the classifier
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1: 1×28×28 -> 32×14×14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32×14×14 -> 64×7×7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Flatten + FC -> latent
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(latent_dim, num_classes)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(x))


class GalaxyCNN(nn.Module):
    """
    Small CNN for multi-channel images such as the 3×100×100 Galaxy Zoo cutouts.

    Follows the plain (non-equivariant) CNN of the SIDDA reference code (Pandya et al., 2025):
    three conv blocks (8 / 16 / 32 channels, kernels 5 / 3 / 3, BatchNorm, ReLU, 2×2 max-pool,
    dropout 0.2), then a 256-unit bottleneck with LayerNorm (the latent space), then a linear
    classifier.  It is *not* the paper's D4-equivariant network and does not reproduce its
    reported accuracies.

    Parameters
    ----------
    in_channels : number of image channels (3 for RGB)
    num_classes : number of output classes
    input_size  : (height, width) of the images
    latent_dim  : size of the bottleneck embedding returned by ``encode`` (default 256)
    dropout     : dropout probability after each conv block
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 6,
                 input_size: Tuple[int, int] = (100, 100), latent_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        chans, kernels = (8, 16, 32), (5, 3, 3)
        blocks, c_in = [], in_channels
        for c_out, k in zip(chans, kernels):
            blocks += [nn.Conv2d(c_in, c_out, kernel_size=k, padding=k // 2), nn.BatchNorm2d(c_out),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout(dropout)]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        with torch.no_grad():
            flat = self.features(torch.zeros(1, in_channels, *input_size)).numel()
        self.fc = nn.Linear(flat, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc(self.features(x).flatten(1)))

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(x))


class MLPRegressor(nn.Module):
    """
    Fully-connected encoder + linear regression head.

    Drop-in replacement for MLP/CNN when the task is regression rather than
    classification.  Exposes the same ``encode`` / ``regress`` / ``forward``
    interface expected by ``SourceOnlyRegressionTrainer`` and
    ``MMDRegressionTrainer``.

    Architecture
    ------------
    input (input_dim) → hidden layers → latent_dim → output_dim

    Parameters
    ----------
    input_dim   : number of input features
    latent_dim  : size of the bottleneck embedding (used by DA methods)
    output_dim  : regression output dimensionality (1 for scalar regression)
    hidden_dims : sizes of hidden layers before the bottleneck
    dropout     : dropout probability applied after each hidden layer
    predict_var : if True, build a 2-column (mean, raw-variance) head for a
                  scalar target (requires ``output_dim`` in {1, 2}); column 1
                  maps to a variance via ``softplus(raw) + eps``
                  (see :func:`mean_var_from_head`)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        output_dim: int = 1,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        predict_var: bool = False,
    ):
        super().__init__()
        if predict_var:
            if output_dim not in (1, 2):
                raise ValueError("predict_var=True supports a scalar target only "
                                 "(output_dim must be 1 or 2)")
            output_dim = 2
        self.predict_var = predict_var

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True),
                       nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, latent_dim), nn.ReLU(inplace=True)]

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(latent_dim, output_dim)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def regress(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regress(self.encode(x))

    def predict_mean_var(self, x: torch.Tensor, eps: float = VAR_EPS) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, var)`` for a model built with ``predict_var=True``."""
        if not self.predict_var:
            raise RuntimeError("predict_mean_var requires MLPRegressor(predict_var=True)")
        return mean_var_from_head(self.forward(x), eps)
