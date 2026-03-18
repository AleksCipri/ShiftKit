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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List


# ─── shared helpers ───────────────────────────────────────────────────────────

def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _batch_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
    return (model(x).argmax(1) == y).sum().item()


# ─── CORAL Loss ───────────────────────────────────────────────────────────────

def _covariance(X: torch.Tensor) -> torch.Tensor:
    """Unbiased sample covariance matrix of a batch of row-vectors."""
    n = X.size(0)
    X_c = X - X.mean(0, keepdim=True)          # centre
    return (X_c.t() @ X_c) / (n - 1)


class CORALLoss(nn.Module):
    """
    Deep CORAL loss: squared Frobenius norm between source and target covariances.

    L_CORAL = (1 / 4d²) ||C_S - C_T||²_F

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

class CORALTrainer:
    """
    Domain adaptation trainer using Deep CORAL.

    Aligns the covariance matrices of source and target latent representations.
    Unlike MMD, CORAL requires no kernel bandwidth selection and has a
    computational cost of O(n·d²) per batch.

    Parameters
    ----------
    model         : network with .encode() and .classify() methods
    source_loader : labelled source DataLoader
    target_loader : target DataLoader (labels used only for accuracy tracking)
    coral_weight  : λ weighting the CORAL term  (total = CE + λ·L_CORAL)
    lr            : Adam learning rate
    warmup_epochs : epochs of source-only CE pre-training before CORAL begins
    device        : 'cuda', 'mps', or 'cpu' (auto-detected if None)

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

        n_batches = min(len(self.source_loader), len(self.target_loader))
        loader = zip(self.source_loader, self.target_loader)

        for (x_src, y_src), (x_tgt, y_tgt) in tqdm(
            loader, total=n_batches,
            desc=f"Epoch {epoch}/{total_epochs}", leave=False
        ):
            x_src, y_src = x_src.to(self.device), y_src.to(self.device)
            x_tgt, y_tgt = x_tgt.to(self.device), y_tgt.to(self.device)

            z_src  = self.model.encode(x_src)
            logits = self.model.classify(z_src)
            ce     = self.ce_loss(logits, y_src)

            if warmup:
                loss      = ce
                coral_val = 0.0
            else:
                z_tgt     = self.model.encode(x_tgt)
                coral     = self.coral_loss(z_src, z_tgt)
                loss      = ce + self.coral_weight * coral
                coral_val = coral.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_ce       += ce.item()
            total_coral    += coral_val
            total_loss_sum += loss.item()
            src_correct    += (logits.argmax(1) == y_src).sum().item()
            n_src          += y_src.size(0)

            tgt_correct += _batch_accuracy(self.model, x_tgt, y_tgt)
            n_tgt       += y_tgt.size(0)

        return {
            "epoch":      epoch,
            "ce_loss":    total_ce       / n_batches,
            "coral_loss": total_coral    / n_batches,
            "mmd_loss":   0.0,   # history-format compatibility
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
