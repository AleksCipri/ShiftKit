"""
Maximum Mean Discrepancy (MMD) domain adaptation.

MMDLoss
-------
Computes the unbiased MMD² between two sets of latent vectors using
a mixture of RBF kernels (bandwidth selected automatically or manually).

MMDTrainer
----------
Trains a model with:
    total_loss = cross_entropy(source) + mmd_weight * MMD²(z_src, z_tgt)

SourceOnlyTrainer
-----------------
Baseline trainer with no domain adaptation — cross-entropy on source only.
Useful for comparing against MMDTrainer to quantify the benefit of DA.

Both trainers record per-epoch 'src_acc' and 'tgt_acc' so training histories
can be compared directly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List


# ─── MMD Loss ────────────────────────────────────────────────────────────────

def _rbf_kernel(X: torch.Tensor, Y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute the RBF kernel matrix k(X, Y) = exp(-||x - y||² / (2σ²))."""
    XX = (X * X).sum(dim=1, keepdim=True)
    YY = (Y * Y).sum(dim=1, keepdim=True)
    dist = XX + YY.t() - 2.0 * X @ Y.t()
    return torch.exp(-dist / (2.0 * sigma ** 2))


class MMDLoss(nn.Module):
    """
    Unbiased MMD² with a mixture of RBF kernels.

    Parameters
    ----------
    sigmas : list of kernel bandwidths.  If None, uses [0.1, 1, 5, 10, 50].
             Using multiple bandwidths captures structure at different scales.
    """

    def __init__(self, sigmas: Optional[List[float]] = None):
        super().__init__()
        self.sigmas = sigmas or [0.1, 1.0, 5.0, 10.0, 50.0]

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mmd = torch.tensor(0.0, device=source.device)
        for sigma in self.sigmas:
            k_ss = _rbf_kernel(source, source, sigma).mean()
            k_tt = _rbf_kernel(target, target, sigma).mean()
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


@torch.no_grad()
def _batch_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
    """Return number of correct predictions for a single batch (no grad)."""
    return (model(x).argmax(1) == y).sum().item()


# ─── MMD Trainer ─────────────────────────────────────────────────────────────

class MMDTrainer:
    """
    Domain adaptation trainer using Maximum Mean Discrepancy.

    Parameters
    ----------
    model          : network with .encode() and .classify() methods
    source_loader  : labelled source DataLoader
    target_loader  : target DataLoader (labels used only for accuracy tracking)
    mmd_weight     : λ weighting the MMD term  (total = CE + λ·MMD²)
    lr             : learning rate
    device         : 'cuda', 'mps', or 'cpu' (auto-detected if None)
    mmd_sigmas     : bandwidths for the RBF kernel mixture (passed to MMDLoss)

    History keys
    ------------
    epoch, ce_loss, mmd_loss, total_loss, src_acc, tgt_acc
    """

    def __init__(
        self,
        model: nn.Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        mmd_weight: float = 1.0,
        lr: float = 1e-3,
        device: Optional[str] = None,
        mmd_sigmas: Optional[List[float]] = None,
    ):
        self.device = torch.device(device) if device else _auto_device()
        self.model = model.to(self.device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.mmd_weight = mmd_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.mmd_loss = MMDLoss(sigmas=mmd_sigmas)
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
                f"MMD={stats['mmd_loss']:.4f}  "
                f"Total={stats['total_loss']:.4f}  "
                f"Src={stats['src_acc']*100:.1f}%  "
                f"Tgt={stats['tgt_acc']*100:.1f}%"
            )
        return self.history

    def _train_epoch(self, epoch: int, total_epochs: int) -> dict:
        self.model.train()
        total_ce = total_mmd = total_loss_sum = 0.0
        src_correct = tgt_correct = n_src = n_tgt = 0

        n_batches = min(len(self.source_loader), len(self.target_loader))
        loader = zip(self.source_loader, self.target_loader)

        for (x_src, y_src), (x_tgt, y_tgt) in tqdm(
            loader, total=n_batches,
            desc=f"Epoch {epoch}/{total_epochs}", leave=False
        ):
            x_src, y_src = x_src.to(self.device), y_src.to(self.device)
            x_tgt, y_tgt = x_tgt.to(self.device), y_tgt.to(self.device)

            z_src = self.model.encode(x_src)
            z_tgt = self.model.encode(x_tgt)

            logits = self.model.classify(z_src)
            ce  = self.ce_loss(logits, y_src)
            mmd = self.mmd_loss(z_src, z_tgt)
            loss = ce + self.mmd_weight * mmd

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_ce       += ce.item()
            total_mmd      += mmd.item()
            total_loss_sum += loss.item()
            src_correct    += (logits.argmax(1) == y_src).sum().item()
            n_src          += y_src.size(0)

            # target accuracy (no grad, reuses current weights)
            tgt_correct += _batch_accuracy(self.model, x_tgt, y_tgt)
            n_tgt       += y_tgt.size(0)

        return {
            "epoch":      epoch,
            "ce_loss":    total_ce       / n_batches,
            "mmd_loss":   total_mmd      / n_batches,
            "total_loss": total_loss_sum / n_batches,
            "src_acc":    src_correct / n_src,
            "tgt_acc":    tgt_correct / n_tgt,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """Compute accuracy on a labelled DataLoader."""
        self.model.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            correct += (self.model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
        return {"domain": domain, "accuracy": correct / total, "n_samples": total}


# ─── Source-Only Baseline Trainer ────────────────────────────────────────────

class SourceOnlyTrainer:
    """
    Baseline trainer: cross-entropy on source data only, no domain adaptation.

    Produces the same history format as MMDTrainer (with mmd_loss=0.0 always)
    so histories can be directly compared via plot_training_history.

    Parameters
    ----------
    model         : network with standard forward() method
    source_loader : labelled source DataLoader
    target_loader : target DataLoader (labels used only for accuracy tracking)
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

        n_batches = min(len(self.source_loader), len(self.target_loader))
        loader = zip(self.source_loader, self.target_loader)

        for (x_src, y_src), (x_tgt, y_tgt) in tqdm(
            loader, total=n_batches,
            desc=f"Epoch {epoch}/{total_epochs}", leave=False
        ):
            x_src, y_src = x_src.to(self.device), y_src.to(self.device)
            x_tgt, y_tgt = x_tgt.to(self.device), y_tgt.to(self.device)

            logits = self.model(x_src)
            loss   = self.ce_loss(logits, y_src)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_ce    += loss.item()
            src_correct += (logits.argmax(1) == y_src).sum().item()
            n_src       += y_src.size(0)

            tgt_correct += _batch_accuracy(self.model, x_tgt, y_tgt)
            n_tgt       += y_tgt.size(0)

        return {
            "epoch":      epoch,
            "ce_loss":    total_ce / n_batches,
            "mmd_loss":   0.0,
            "total_loss": total_ce / n_batches,
            "src_acc":    src_correct / n_src,
            "tgt_acc":    tgt_correct / n_tgt,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """Compute accuracy on a labelled DataLoader."""
        self.model.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            correct += (self.model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
        return {"domain": domain, "accuracy": correct / total, "n_samples": total}
