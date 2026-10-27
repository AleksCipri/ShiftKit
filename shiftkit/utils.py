"""Small helpers shared by examples and notebooks."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch; optionally request deterministic GPU kernels.

    With ``deterministic=True`` the same seed reproduces a training run exactly on the same
    hardware and library versions.  Without it, GPU convolutions are not bit-for-bit
    repeatable, and in unstable problems small differences can grow into very different
    results (see the SIDDA notebook).  Deterministic mode did not change run time in the
    notebooks' small networks, but some operations may be slower or unsupported on other
    models; unsupported ones raise a warning, not an error.

    Call it before building models and data loaders.  It cannot seed a ``DataLoader``'s
    shuffling for you: pass ``generator=torch.Generator().manual_seed(seed)`` to it, or call
    this function immediately before creating the loader (the loader draws its base seed from
    the global generator when it is first iterated).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Must be set before the first cuBLAS call to take effect; harmless if already set.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
