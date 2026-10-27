"""
Maximum Mean Discrepancy (MMD) domain adaptation.

MMDLoss
-------
Estimates MMD² between two sets of latent vectors with a **sum** of RBF
kernels over a fixed list of bandwidths.  By default it returns the biased
V-statistic (all pairs, including i = j); ``unbiased=True`` gives the
U-statistic of Gretton et al. (2012).  Bandwidths are fixed defaults, user
supplied, or (opt-in) ``sigmas="median"``, the median pairwise distance of the
pooled batch.  No other bandwidth selection is performed.

MMDTrainer
----------
Trains a model with:
    total_loss = cross_entropy(source) + mmd_weight * MMD²(z_src, z_tgt)

SourceOnlyTrainer
-----------------
Baseline trainer with no domain adaptation — cross-entropy on source only.
Useful for comparing against MMDTrainer to quantify the benefit of DA.

Both trainers record per-epoch 'src_acc' and 'tgt_acc'.  ``src_acc`` is the
training-mode prediction on each source batch *before* that batch's optimizer
step; ``tgt_acc`` is measured in evaluation mode (no BatchNorm update, no
dropout) *after* the step, and is ``nan`` when target batches carry no labels.
Neither trainer seeds random number generators; call ``torch.manual_seed``
yourself for repeatable runs.
"""

import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List, Union

from .base import BaseTrainer
from .node_batch import (
    unpack_batch,
    unpack_target_batch,
    batch_accuracy,
    encode_domains,
    evaluate_accuracy,
    node_classify_logits,
    paired_batches,
)


# ─── MMD Loss ────────────────────────────────────────────────────────────────

def _sq_dists(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances ``||x - y||²`` (always ≥ 0).

    Uses direct differences rather than ``|x|² + |y|² − 2x·y``, which suffers
    catastrophic cancellation for large-norm latents (it can return negative
    "squared distances" and kernel values > 1).
    """
    return torch.cdist(X, Y, compute_mode="donot_use_mm_for_euclid_dist").pow(2)


def _rbf_kernel(X: torch.Tensor, Y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute the RBF kernel matrix k(X, Y) = exp(-||x - y||² / (2σ²))."""
    return torch.exp(-_sq_dists(X, Y) / (2.0 * sigma ** 2))


def _median_sigma(source: torch.Tensor, target: torch.Tensor) -> float:
    """Median pairwise distance of the pooled batch (detached; 1.0 if degenerate)."""
    Z = torch.cat([source, target], dim=0).detach()
    d = _sq_dists(Z, Z)
    off = d[~torch.eye(Z.size(0), dtype=torch.bool, device=Z.device)]
    med = off.median().sqrt().item() if off.numel() else 0.0
    return med if math.isfinite(med) and med > 0 else 1.0


class MMDLoss(nn.Module):
    """
    MMD² with a sum of RBF kernels (one term per bandwidth).

    For each bandwidth σ, with source ``x_1..x_n`` and target ``y_1..y_m``:

    * biased V-statistic (default)::

        (1/n²) Σ_{i,j} k(x_i,x_j) + (1/m²) Σ_{i,j} k(y_i,y_j) − (2/nm) Σ_{i,j} k(x_i,y_j)

      always ≥ 0 (up to rounding); this is what most DA code uses.
    * unbiased U-statistic (``unbiased=True``)::

        1/(n(n−1)) Σ_{i≠j} k(x_i,x_j) + 1/(m(m−1)) Σ_{i≠j} k(y_i,y_j) − (2/nm) Σ_{i,j} k(x_i,y_j)

      can be negative; needs n, m ≥ 2.

    The returned value is the **sum** of these per-bandwidth estimates (not
    their average).

    Parameters
    ----------
    sigmas   : list of kernel bandwidths, or ``"median"`` for a single bandwidth
               equal to the median pairwise distance of the pooled batch
               (recomputed every call, not differentiated).  If None, uses
               [0.1, 1, 5, 10, 50].
    unbiased : use the U-statistic instead of the biased V-statistic.
    """

    def __init__(self, sigmas: Optional[Union[List[float], str]] = None, unbiased: bool = False):
        super().__init__()
        if isinstance(sigmas, str) and sigmas != "median":
            raise ValueError(f"sigmas must be a list of floats, None or 'median', got {sigmas!r}")
        self.sigmas = sigmas or [0.1, 1.0, 5.0, 10.0, 50.0]
        self.unbiased = unbiased

    @staticmethod
    def _within(K: torch.Tensor, unbiased: bool) -> torch.Tensor:
        n = K.size(0)
        if not unbiased:
            return K.mean()
        if n < 2:
            raise ValueError("unbiased MMD needs at least 2 samples per domain")
        return (K.sum() - K.diagonal().sum()) / (n * (n - 1))

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.size(0) == 0 or target.size(0) == 0:
            raise ValueError("MMDLoss needs non-empty source and target batches")
        sigmas = [_median_sigma(source, target)] if self.sigmas == "median" else self.sigmas
        mmd = torch.tensor(0.0, device=source.device)
        for sigma in sigmas:
            k_ss = self._within(_rbf_kernel(source, source, sigma), self.unbiased)
            k_tt = self._within(_rbf_kernel(target, target, sigma), self.unbiased)
            k_st = _rbf_kernel(source, target, sigma).mean()
            mmd = mmd + k_ss + k_tt - 2.0 * k_st
        return mmd


# ─── shared helpers ───────────────────────────────────────────────────────────

def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── MMD Trainer ─────────────────────────────────────────────────────────────

class MMDTrainer(BaseTrainer):
    """
    Domain adaptation trainer using Maximum Mean Discrepancy.

    Parameters
    ----------
    model          : network with .encode() and .classify() methods
    source_loader  : labelled source DataLoader
    target_loader  : target DataLoader yielding ``x`` or ``(x, y)``; labels,
                     if present, are used only for ``tgt_acc`` monitoring
    mmd_weight     : λ weighting the MMD term  (total = CE + λ·MMD²)
    lr             : learning rate
    warmup_epochs  : epochs of source-only CE pre-training before MMD DA begins
    device         : 'cuda', 'mps', or 'cpu' (auto-detected if None)
    mmd_sigmas     : bandwidths for the RBF kernel mixture (passed to MMDLoss)

    An epoch covers the longer of the two loaders (the shorter one restarts).
    During DA epochs, tensor source and target batches are encoded in one
    joint forward pass, so BatchNorm sees both domains together.

    History keys
    ------------
    epoch, ce_loss, mmd_loss, total_loss, src_acc, tgt_acc (nan if unlabelled)
    """

    def __init__(
        self,
        model: nn.Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        mmd_weight: float = 1.0,
        lr: float = 1e-3,
        warmup_epochs: int = 0,
        device: Optional[str] = None,
        mmd_sigmas: Optional[Union[List[float], str]] = None,
    ):
        self.device = torch.device(device) if device else _auto_device()
        self.model = model.to(self.device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.mmd_weight = mmd_weight
        self.warmup_epochs = warmup_epochs

        self.ce_loss = nn.CrossEntropyLoss()
        self.mmd_loss = MMDLoss(sigmas=mmd_sigmas)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        self.history: List[dict] = []

    def fit(self, epochs: int = 10) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        for epoch in range(1, epochs + 1):
            is_warmup = epoch <= self.warmup_epochs
            stats = self._train_epoch(epoch, epochs, is_warmup)
            self.history.append(stats)
            phase = "warmup" if is_warmup else "MMD  "
            mmd_str = f"  MMD={stats['mmd_loss']:.4f}" if not is_warmup else ""
            print(
                f"[{epoch:>3}/{epochs}] [{phase}]  "
                f"CE={stats['ce_loss']:.4f}{mmd_str}  "
                f"Total={stats['total_loss']:.4f}  "
                f"Src={stats['src_acc']*100:.1f}%  "
                f"Tgt={stats['tgt_acc']*100:.1f}%"
            )
        return self.history

    def _train_epoch(self, epoch: int, total_epochs: int, warmup: bool = False) -> dict:
        self.model.train()
        total_ce = total_mmd = total_loss_sum = 0.0
        src_correct = tgt_correct = n_src = n_tgt = 0

        loader, n_batches = paired_batches(self.source_loader, self.target_loader)

        for (x_src, y_src), tgt_batch in tqdm(
            loader, total=n_batches,
            desc=f"Epoch {epoch}/{total_epochs}", leave=False
        ):
            x_src, y_src, _ = unpack_batch(x_src, y_src, self.device)
            x_tgt, y_tgt = unpack_target_batch(tgt_batch, self.device)

            # joint source+target encode for tensor batches (shared BN stats)
            z_src, z_tgt = encode_domains(self.model, x_src, None if warmup else x_tgt)
            logits = self.model.classify(z_src)
            ce = self.ce_loss(logits, y_src)
            if warmup:
                loss = ce
                mmd_val = 0.0
            else:
                mmd = self.mmd_loss(z_src, z_tgt)
                loss = ce + self.mmd_weight * mmd
                mmd_val = mmd.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_ce       += ce.item()
            total_mmd      += mmd_val
            total_loss_sum += loss.item()
            # src_acc: training-mode logits of this batch, before the update
            src_correct    += (logits.argmax(1) == y_src).sum().item()
            n_src          += y_src.size(0)

            # tgt_acc: eval mode (no BN update / dropout), after the update
            if y_tgt is not None:
                tgt_correct += batch_accuracy(self.model, x_tgt, y_tgt)
                n_tgt       += y_tgt.size(0)

        return {
            "epoch":      epoch,
            "ce_loss":    total_ce       / n_batches,
            "mmd_loss":   total_mmd      / n_batches,
            "total_loss": total_loss_sum / n_batches,
            "src_acc":    src_correct / n_src,
            "tgt_acc":    tgt_correct / n_tgt if n_tgt else float("nan"),
        }

    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """Compute accuracy on a labelled DataLoader (evaluation mode)."""
        return evaluate_accuracy(self.model, loader, self.device, domain)


# ─── Source-Only Baseline Trainer ────────────────────────────────────────────

class SourceOnlyTrainer(BaseTrainer):
    """
    Baseline trainer: cross-entropy on source data only, no domain adaptation.

    Produces the same history format as MMDTrainer (with mmd_loss=0.0 always)
    so histories can be directly compared via plot_training_history.

    Target batches are used only for ``tgt_acc`` monitoring, which runs in
    evaluation mode, so target data never updates BatchNorm statistics or any
    other model state.

    Parameters
    ----------
    model         : network with standard forward() method
    source_loader : labelled source DataLoader
    target_loader : target DataLoader yielding ``x`` or ``(x, y)``; labels,
                    if present, are used only for ``tgt_acc`` monitoring
    lr            : learning rate
    device        : 'cuda', 'mps', or 'cpu' (auto-detected if None)

    History keys
    ------------
    epoch, ce_loss, mmd_loss (always 0), total_loss, src_acc, tgt_acc
    """

    def __init__(
        self,
        model: nn.Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device) if device else _auto_device()
        self.model = model.to(self.device)
        self.source_loader = source_loader
        self.target_loader = target_loader

        self.ce_loss  = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        self.history: List[dict] = []

    def fit(self, epochs: int = 10) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        for epoch in range(1, epochs + 1):
            stats = self._train_epoch(epoch, epochs)
            self.history.append(stats)
            print(
                f"[{epoch:>3}/{epochs}] "
                f"CE={stats['ce_loss']:.4f}  "
                f"Src={stats['src_acc']*100:.1f}%  "
                f"Tgt={stats['tgt_acc']*100:.1f}%"
            )
        return self.history

    def _train_epoch(self, epoch: int, total_epochs: int) -> dict:
        self.model.train()
        total_ce = 0.0
        src_correct = tgt_correct = n_src = n_tgt = 0

        loader, n_batches = paired_batches(self.source_loader, self.target_loader)

        for (x_src, y_src), tgt_batch in tqdm(
            loader, total=n_batches,
            desc=f"Epoch {epoch}/{total_epochs}", leave=False
        ):
            x_src, y_src, node_batch = unpack_batch(x_src, y_src, self.device)
            x_tgt, y_tgt = unpack_target_batch(tgt_batch, self.device)

            if node_batch:
                logits = node_classify_logits(self.model, x_src)
            else:
                logits = self.model(x_src)
            loss = self.ce_loss(logits, y_src)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_ce    += loss.item()
            # src_acc: training-mode logits of this batch, before the update
            src_correct += (logits.argmax(1) == y_src).sum().item()
            n_src       += y_src.size(0)

            # tgt_acc: eval mode (no BN update / dropout), after the update
            if y_tgt is not None:
                tgt_correct += batch_accuracy(self.model, x_tgt, y_tgt)
                n_tgt       += y_tgt.size(0)

        return {
            "epoch":      epoch,
            "ce_loss":    total_ce / n_batches,
            "mmd_loss":   0.0,
            "total_loss": total_ce / n_batches,
            "src_acc":    src_correct / n_src,
            "tgt_acc":    tgt_correct / n_tgt if n_tgt else float("nan"),
        }

    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """Compute accuracy on a labelled DataLoader (evaluation mode)."""
        return evaluate_accuracy(self.model, loader, self.device, domain)
