"""
Deep CORAL (CORrelation ALignment) domain adaptation.

CORALLoss
---------
Aligns the second-order statistics (covariance matrices) of source and target
feature distributions.  The loss is the squared Frobenius norm between the
covariance matrices of the two domains, normalised by feature dimensionality:

    L_CORAL = (1 / 4d²) ||C_S - C_T||²_F

where C_S and C_T are the unbiased sample covariance matrices of the source
and target latent vectors, and d is the feature (latent) dimensionality.

Centering removes each domain's mean, so CORAL does **not** align means:
``CORALLoss(Z, Z + c) == 0`` for any constant shift c.  The 1/(4d²) factor is
a dimensional normalisation, not scale invariance: scaling all features by a
multiplies the loss by a⁴.  Covariances need at least two samples per domain.

This is computationally lightweight (O(n·d²)) compared to kernel methods,
and does not require choosing a kernel or bandwidth.

CORALTrainer
------------
Trains a model with:
    total_loss = cross_entropy(source) + coral_weight * L_CORAL(z_src, z_tgt)

Reference
---------
Sun, B., & Saenko, K. (2016).
Deep CORAL: Correlation Alignment for Deep Domain Adaptation.
ECCV Workshops 2016, LNCS 9915, 443–450.
https://arxiv.org/abs/1607.01719
"""

import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List

from .base import BaseTrainer
from .mmd import _auto_device
from .node_batch import (
    unpack_batch,
    unpack_target_batch,
    batch_accuracy,
    encode_domains,
    evaluate_accuracy,
    paired_batches,
)


# ─── CORAL Loss ───────────────────────────────────────────────────────────────

def _covariance(X: torch.Tensor) -> torch.Tensor:
    """Unbiased sample covariance matrix of a batch of row-vectors (n ≥ 2)."""
    n = X.size(0)
    if n < 2:
        raise ValueError(f"CORAL covariance needs at least 2 samples per domain, got {n}")
    X_c = X - X.mean(0, keepdim=True)          # centre
    return (X_c.t() @ X_c) / (n - 1)


class CORALLoss(nn.Module):
    """
    Deep CORAL loss: squared Frobenius norm between source and target covariances.

    L_CORAL = (1 / 4d²) ||C_S - C_T||²_F

    Only second-order statistics are compared (means are removed by
    centering).  Raises ``ValueError`` if either domain has fewer than 2 rows.

    Parameters
    ----------
    (none — the loss is parameter-free)
    """

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        source : (n_s, d) source latent vectors
        target : (n_t, d) target latent vectors

        Returns
        -------
        Scalar CORAL loss.
        """
        d = source.size(1)
        C_s = _covariance(source)
        C_t = _covariance(target)
        diff = C_s - C_t
        return (diff * diff).sum() / (4 * d * d)


# ─── CORAL Trainer ────────────────────────────────────────────────────────────

class CORALTrainer(BaseTrainer):
    """
    Domain adaptation trainer using Deep CORAL.

    Aligns the covariance matrices of source and target latent representations.
    Unlike MMD, CORAL requires no kernel bandwidth selection and has a
    computational cost of O(n·d²) per batch.

    Parameters
    ----------
    model         : network with .encode() and .classify() methods
    source_loader : labelled source DataLoader
    target_loader : target DataLoader yielding ``x`` or ``(x, y)``; labels,
                    if present, are used only for ``tgt_acc`` monitoring
    coral_weight  : λ weighting the CORAL term  (total = CE + λ·L_CORAL)
    lr            : Adam learning rate
    warmup_epochs : epochs of source-only CE pre-training before CORAL begins
    device        : 'cuda', 'mps', or 'cpu' (auto-detected if None)

    An epoch covers the longer loader; during CORAL epochs tensor source and
    target batches are encoded jointly (shared BatchNorm batch statistics).
    If a source or target batch has fewer than 2 samples, the CORAL term is
    skipped for that step (CE only) with a warning, instead of producing NaN.
    ``src_acc`` uses training-mode logits before each update; ``tgt_acc`` is
    measured in eval mode after it (``nan`` without target labels).

    History keys
    ------------
    epoch, ce_loss, coral_loss, mmd_loss (always 0.0), total_loss, src_acc, tgt_acc
    """

    def __init__(
        self,
        model: nn.Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        coral_weight: float = 1.0,
        lr: float = 1e-3,
        warmup_epochs: int = 0,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device) if device else _auto_device()
        self.model = model.to(self.device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.coral_weight = coral_weight
        self.warmup_epochs = warmup_epochs

        self.ce_loss    = nn.CrossEntropyLoss()
        self.coral_loss = CORALLoss()
        self.optimizer  = optim.Adam(model.parameters(), lr=lr)

        self.history: List[dict] = []

    def fit(self, epochs: int = 10) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        for epoch in range(1, epochs + 1):
            is_warmup = epoch <= self.warmup_epochs
            stats = self._train_epoch(epoch, epochs, is_warmup)
            self.history.append(stats)
            phase = "warmup" if is_warmup else "CORAL"
            coral_str = f"  CORAL={stats['coral_loss']:.4f}" if not is_warmup else ""
            print(
                f"[{epoch:>3}/{epochs}] [{phase}]  "
                f"CE={stats['ce_loss']:.4f}{coral_str}  "
                f"Total={stats['total_loss']:.4f}  "
                f"Src={stats['src_acc']*100:.1f}%  "
                f"Tgt={stats['tgt_acc']*100:.1f}%"
            )
        return self.history

    def _train_epoch(self, epoch: int, total_epochs: int, warmup: bool = False) -> dict:
        self.model.train()
        total_ce = total_coral = total_loss_sum = 0.0
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
            ce     = self.ce_loss(logits, y_src)

            if warmup:
                loss      = ce
                coral_val = 0.0
            elif z_src.size(0) < 2 or z_tgt.size(0) < 2:
                warnings.warn(
                    f"CORAL term skipped for a batch with {z_src.size(0)} source / "
                    f"{z_tgt.size(0)} target samples (covariance needs ≥ 2)",
                    RuntimeWarning,
                )
                loss      = ce
                coral_val = 0.0
            else:
                coral     = self.coral_loss(z_src, z_tgt)
                loss      = ce + self.coral_weight * coral
                coral_val = coral.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_ce       += ce.item()
            total_coral    += coral_val
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
            "coral_loss": total_coral    / n_batches,
            "mmd_loss":   0.0,   # history-format compatibility
            "total_loss": total_loss_sum / n_batches,
            "src_acc":    src_correct / n_src,
            "tgt_acc":    tgt_correct / n_tgt if n_tgt else float("nan"),
        }

    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """Compute accuracy on a labelled DataLoader (evaluation mode)."""
        return evaluate_accuracy(self.model, loader, self.device, domain)
