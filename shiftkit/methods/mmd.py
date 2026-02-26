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

The classifier head is only supervised on source labels; the encoder is
pulled toward domain-invariant representations via MMD.
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
    XX = (X * X).sum(dim=1, keepdim=True)          # (n, 1)
    YY = (Y * Y).sum(dim=1, keepdim=True)           # (m, 1)
    dist = XX + YY.t() - 2.0 * X @ Y.t()           # (n, m)
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
        """
        Parameters
        ----------
        source : (n, d) latent vectors from source domain
        target : (m, d) latent vectors from target domain

        Returns
        -------
        scalar MMD² estimate
        """
        mmd = torch.tensor(0.0, device=source.device)
        for sigma in self.sigmas:
            k_ss = _rbf_kernel(source, source, sigma).mean()
            k_tt = _rbf_kernel(target, target, sigma).mean()
            k_st = _rbf_kernel(source, target, sigma).mean()
            mmd = mmd + k_ss + k_tt - 2.0 * k_st
        return mmd


# ─── MMD Trainer ─────────────────────────────────────────────────────────────

class MMDTrainer:
    """
    Domain adaptation trainer using Maximum Mean Discrepancy.

    Parameters
    ----------
    model          : network with .encode() and .classify() methods
    source_loader  : labelled source DataLoader
    target_loader  : unlabelled target DataLoader (labels ignored during DA)
    mmd_weight     : λ weighting the MMD term  (total = CE + λ·MMD²)
    lr             : learning rate
    device         : 'cuda', 'mps', or 'cpu' (auto-detected if None)
    mmd_sigmas     : bandwidths for the RBF kernel mixture (passed to MMDLoss)
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
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.mmd_weight = mmd_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.mmd_loss = MMDLoss(sigmas=mmd_sigmas)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        self.history: List[dict] = []   # filled each epoch

    # ------------------------------------------------------------------
    def fit(self, epochs: int = 10) -> List[dict]:
        """
        Train for *epochs* epochs.

        Returns
        -------
        history : list of per-epoch dicts with keys
                  'epoch', 'ce_loss', 'mmd_loss', 'total_loss', 'src_acc'
        """
        for epoch in range(1, epochs + 1):
            stats = self._train_epoch(epoch, epochs)
            self.history.append(stats)
            print(
                f"[{epoch:>3}/{epochs}] "
                f"CE={stats['ce_loss']:.4f}  "
                f"MMD={stats['mmd_loss']:.4f}  "
                f"Total={stats['total_loss']:.4f}  "
                f"Src-acc={stats['src_acc']*100:.1f}%"
            )
        return self.history

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int, total_epochs: int) -> dict:
        self.model.train()
        total_ce = total_mmd = total_loss_sum = 0.0
        correct = n_src = 0

        # Zip loaders; stop when the shorter one is exhausted
        loader = zip(self.source_loader, self.target_loader)
        n_batches = min(len(self.source_loader), len(self.target_loader))

        for (x_src, y_src), (x_tgt, _) in tqdm(
            loader, total=n_batches,
            desc=f"Epoch {epoch}/{total_epochs}", leave=False
        ):
            x_src, y_src = x_src.to(self.device), y_src.to(self.device)
            x_tgt = x_tgt.to(self.device)

            z_src = self.model.encode(x_src)
            z_tgt = self.model.encode(x_tgt)

            logits = self.model.classify(z_src)
            ce = self.ce_loss(logits, y_src)
            mmd = self.mmd_loss(z_src, z_tgt)
            loss = ce + self.mmd_weight * mmd

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_ce += ce.item()
            total_mmd += mmd.item()
            total_loss_sum += loss.item()
            correct += (logits.argmax(1) == y_src).sum().item()
            n_src += y_src.size(0)

        return {
            "epoch": epoch,
            "ce_loss": total_ce / n_batches,
            "mmd_loss": total_mmd / n_batches,
            "total_loss": total_loss_sum / n_batches,
            "src_acc": correct / n_src,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """
        Compute accuracy on a labelled DataLoader.

        Returns
        -------
        dict with keys 'domain', 'accuracy', 'n_samples'
        """
        self.model.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            preds = self.model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        return {"domain": domain, "accuracy": correct / total, "n_samples": total}
