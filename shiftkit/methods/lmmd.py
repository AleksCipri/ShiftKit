"""
Local Maximum Mean Discrepancy (LMMD) domain adaptation.

LMMDLoss
--------
Class-conditional extension of MMD that aligns per-class subdomains instead
of the overall marginal distributions.  For each class c the source and target
subdomains are weighted by:

  w_i^(src, c) = y_ic  / Σ_j y_jc          (one-hot normalised)
  w_i^(tgt, c) = p̂_ic  / Σ_j p̂_jc          (softmax normalised)

where y is the source ground-truth label (one-hot) and p̂ is the predicted
softmax probability for target samples.

The LMMD loss is:

  d̂(p, q) = (1/C) Σ_c  [ Σ_{i,j} w_i^s w_j^s k(z_i^s, z_j^s)
                         + Σ_{i,j} w_i^t w_j^t k(z_i^t, z_j^t)
                         - 2 Σ_{i,j} w_i^s w_j^t k(z_i^s, z_j^t) ]

where k is an RBF kernel.  The per-class kernel sums are vectorised as:

  Σ_c w_c^T K w_c  =  (K ⊙ (W W^T)).sum()

LMMDTrainer
-----------
Trains a model with:  total_loss = CE(source) + lmmd_weight * LMMD(z_src, z_tgt)

Target soft labels are produced from the model's current predictions and
treated as fixed weights (stop-gradient), matching the original DSAN paper.

Reference
---------
Zhu, Y., Zhuang, F., & Wang, D. (2020).
Deep Subdomain Adaptation Network for Image Classification.
IEEE Transactions on Neural Networks and Learning Systems, 32(4), 1713–1722.
https://arxiv.org/abs/2106.09388
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List


# ─── shared helpers (same as mmd.py, kept local to avoid circular imports) ───

def _rbf_kernel(X: torch.Tensor, Y: torch.Tensor, sigma: float) -> torch.Tensor:
    """RBF kernel k(X, Y) = exp(-||x - y||² / 2σ²)."""
    XX = (X * X).sum(dim=1, keepdim=True)
    YY = (Y * Y).sum(dim=1, keepdim=True)
    dist = XX + YY.t() - 2.0 * X @ Y.t()
    return torch.exp(-dist / (2.0 * sigma ** 2))


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _batch_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
    return (model(x).argmax(1) == y).sum().item()


# ─── LMMD Loss ────────────────────────────────────────────────────────────────

class LMMDLoss(nn.Module):
    """
    Local (class-conditional) Maximum Mean Discrepancy.

    Parameters
    ----------
    num_classes : number of classes C
    sigmas      : RBF kernel bandwidths; defaults to [0.1, 1.0, 5.0, 10.0, 50.0]
    """

    def __init__(self, num_classes: int, sigmas: Optional[List[float]] = None):
        super().__init__()
        self.num_classes = num_classes
        self.sigmas = sigmas or [0.1, 1.0, 5.0, 10.0, 50.0]

    def forward(
        self,
        z_src: torch.Tensor,
        z_tgt: torch.Tensor,
        y_src: torch.Tensor,
        y_tgt_prob: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_src      : (n_s, d) source latent vectors
        z_tgt      : (n_t, d) target latent vectors
        y_src      : (n_s,)   source ground-truth class labels (long)
        y_tgt_prob : (n_t, C) target softmax probabilities (stop-gradient)

        Returns
        -------
        Scalar LMMD loss (averaged over C classes and kernel bandwidths).
        """
        C = self.num_classes

        # ── per-class sample weights ──────────────────────────────────────────
        # source: one-hot, then normalise each column so weights sum to 1
        y_src_oh = F.one_hot(y_src, C).float()              # (n_s, C)
        ws = y_src_oh / (y_src_oh.sum(0, keepdim=True) + 1e-8)   # (n_s, C)

        # target: softmax probabilities, normalise each column
        wt = y_tgt_prob / (y_tgt_prob.sum(0, keepdim=True) + 1e-8)  # (n_t, C)

        # ── weight outer products (used for vectorised kernel sums) ───────────
        # Σ_c w_c^T K w_c  =  (K ⊙ (W W^T)).sum()
        W_ss = ws @ ws.t()   # (n_s, n_s)
        W_tt = wt @ wt.t()   # (n_t, n_t)
        W_st = ws @ wt.t()   # (n_s, n_t)

        lmmd = torch.tensor(0.0, device=z_src.device)
        for sigma in self.sigmas:
            K_ss = _rbf_kernel(z_src, z_src, sigma)
            K_tt = _rbf_kernel(z_tgt, z_tgt, sigma)
            K_st = _rbf_kernel(z_src, z_tgt, sigma)

            lmmd = lmmd + (
                (K_ss * W_ss).sum()
                + (K_tt * W_tt).sum()
                - 2.0 * (K_st * W_st).sum()
            )

        return lmmd / C


# ─── LMMD Trainer ─────────────────────────────────────────────────────────────

class LMMDTrainer:
    """
    Domain adaptation trainer using Local Maximum Mean Discrepancy (LMMD).

    Unlike global MMD which aligns marginal distributions, LMMD aligns
    per-class subdomains.  Source samples are weighted by their ground-truth
    class labels; target samples are weighted by the model's current softmax
    predictions (treated as stop-gradient soft pseudo-labels).

    Parameters
    ----------
    model         : network with .encode() and .classify() methods
    source_loader : labelled source DataLoader
    target_loader : target DataLoader (labels used only for accuracy tracking)
    num_classes   : number of output classes
    lmmd_weight   : λ weighting the LMMD term  (total = CE + λ·LMMD)
    lr            : Adam learning rate
    warmup_epochs : epochs of source-only CE pre-training before LMMD begins
    device        : 'cuda', 'mps', or 'cpu' (auto-detected if None)
    lmmd_sigmas   : RBF kernel bandwidths (passed to LMMDLoss)

    History keys
    ------------
    epoch, ce_loss, lmmd_loss, mmd_loss (always 0.0), total_loss, src_acc, tgt_acc
    """

    def __init__(
        self,
        model: nn.Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        num_classes: int,
        lmmd_weight: float = 1.0,
        lr: float = 1e-3,
        warmup_epochs: int = 0,
        device: Optional[str] = None,
        lmmd_sigmas: Optional[List[float]] = None,
    ):
        self.device = torch.device(device) if device else _auto_device()
        self.model = model.to(self.device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.num_classes = num_classes
        self.lmmd_weight = lmmd_weight
        self.warmup_epochs = warmup_epochs

        self.ce_loss   = nn.CrossEntropyLoss()
        self.lmmd_loss = LMMDLoss(num_classes=num_classes, sigmas=lmmd_sigmas)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        self.history: List[dict] = []

    def fit(self, epochs: int = 10) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        for epoch in range(1, epochs + 1):
            is_warmup = epoch <= self.warmup_epochs
            stats = self._train_epoch(epoch, epochs, is_warmup)
            self.history.append(stats)
            phase = "warmup" if is_warmup else "LMMD "
            lmmd_str = f"  LMMD={stats['lmmd_loss']:.4f}" if not is_warmup else ""
            print(
                f"[{epoch:>3}/{epochs}] [{phase}]  "
                f"CE={stats['ce_loss']:.4f}{lmmd_str}  "
                f"Total={stats['total_loss']:.4f}  "
                f"Src={stats['src_acc']*100:.1f}%  "
                f"Tgt={stats['tgt_acc']*100:.1f}%"
            )
        return self.history

    def _train_epoch(self, epoch: int, total_epochs: int, warmup: bool = False) -> dict:
        self.model.train()
        total_ce = total_lmmd = total_loss_sum = 0.0
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
                loss     = ce
                lmmd_val = 0.0
            else:
                z_tgt = self.model.encode(x_tgt)
                # target soft labels: stop-gradient, used as alignment weights
                with torch.no_grad():
                    y_tgt_prob = torch.softmax(
                        self.model.classify(z_tgt), dim=1
                    )
                lmmd     = self.lmmd_loss(z_src, z_tgt, y_src, y_tgt_prob)
                loss     = ce + self.lmmd_weight * lmmd
                lmmd_val = lmmd.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_ce       += ce.item()
            total_lmmd     += lmmd_val
            total_loss_sum += loss.item()
            src_correct    += (logits.argmax(1) == y_src).sum().item()
            n_src          += y_src.size(0)

            tgt_correct += _batch_accuracy(self.model, x_tgt, y_tgt)
            n_tgt       += y_tgt.size(0)

        return {
            "epoch":      epoch,
            "ce_loss":    total_ce       / n_batches,
            "lmmd_loss":  total_lmmd     / n_batches,
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
