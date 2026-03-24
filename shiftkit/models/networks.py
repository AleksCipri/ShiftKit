"""
Neural network architectures for domain adaptation.

All models expose:
    encode(x)  -> latent tensor  (used by DA methods)
    classify(z)-> logit tensor   (used for supervised loss)
    forward(x) -> logits         (encode + classify, standard nn.Module interface)
"""

import torch
import torch.nn as nn
from typing import Tuple


class MLP(nn.Module):
    """
    Fully-connected encoder + linear classifier.

    Architecture
    ------------
    input (1×28×28 flattened) -> hidden layers -> latent_dim -> num_classes

    Parameters
    ----------
    latent_dim  : size of the bottleneck embedding
    num_classes : number of output classes
    hidden_dims : sizes of hidden layers before the bottleneck
    dropout     : dropout probability applied after each hidden layer
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 10,
        hidden_dims: Tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        input_dim = 28 * 28

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
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        output_dim: int = 1,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()

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
