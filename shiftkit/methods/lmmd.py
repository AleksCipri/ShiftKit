"""
Local Maximum Mean Discrepancy (LMMD) domain adaptation.

LMMDLoss
--------
Class-conditional extension of MMD that compares per-class subdomains instead
of the overall marginal distributions.  For each class c the source and target
subdomains are weighted by:

  w_i^(src, c) = y_ic  / Σ_j y_jc          (one-hot normalised)
  w_i^(tgt, c) = p̂_ic  / Σ_j p̂_jc          (softmax normalised)

where y is the source ground-truth label (one-hot) and p̂ is the predicted
softmax probability for target samples.

Missing-class policy: a class contributes only if it is **present in the
source batch** and its total target probability mass Σ_j p̂_jc exceeds
``min_target_mass``.  Other classes have no well-defined (normalised) source
or target measure in this batch and are skipped.  Let A be the set of active
classes; the loss is

  d̂(p, q) = (1/|A|) Σ_{c∈A} Σ_σ [ Σ_{i,j} w_i^s w_j^s k_σ(z_i^s, z_j^s)
                                 + Σ_{i,j} w_i^t w_j^t k_σ(z_i^t, z_j^t)
                                 - 2 Σ_{i,j} w_i^s w_j^t k_σ(z_i^s, z_j^t) ]

i.e. kernels are **summed** over bandwidths σ and averaged over active
classes (0 if no class is active).  The per-class kernel sums are vectorised:

  Σ_c w_c^T K w_c  =  (K ⊙ (W W^T)).sum()

LMMDTrainer
-----------
Trains a model with:  total_loss = CE(source) + lmmd_weight * LMMD(z_src, z_tgt)

Target soft labels are produced from the model's current predictions and
treated as fixed weights (stop-gradient), as in DSAN.  This is a DSAN-style
variant, not an exact reproduction of the paper's training recipe: there is
no λ warm-up ramp, no learning-rate annealing, and the kernel mixture and
missing-class handling are ShiftKit choices.

Reference
---------
Zhu, Y., Zhuang, F., Wang, J., Ke, G., Chen, J., Bian, J., Xiong, H., & He, Q. (2021).
Deep Subdomain Adaptation Network for Image Classification.
IEEE Transactions on Neural Networks and Learning Systems, 32(4), 1713–1722.
https://doi.org/10.1109/TNNLS.2020.2988928  (preprint: https://arxiv.org/abs/2106.09388)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List

from .base import BaseTrainer
from .mmd import _rbf_kernel, _auto_device
from .node_batch import (
    unpack_batch,
    unpack_target_batch,
    batch_accuracy,
    encode_domains,
    evaluate_accuracy,
    paired_batches,
)


# ─── LMMD Loss ────────────────────────────────────────────────────────────────

class LMMDLoss(nn.Module):
    """
    Local (class-conditional) Maximum Mean Discrepancy.

    Parameters
    ----------
    num_classes     : number of classes C
    sigmas          : RBF kernel bandwidths; defaults to [0.1, 1.0, 5.0, 10.0, 50.0]
    min_target_mass : a class is skipped when its summed target probability in
                      the batch is ≤ this value (avoids giving a negligible-mass
                      class full normalised weight)
    """

    def __init__(self, num_classes: int, sigmas: Optional[List[float]] = None,
                 min_target_mass: float = 1e-3):
        super().__init__()
        self.num_classes = num_classes
        self.sigmas = sigmas or [0.1, 1.0, 5.0, 10.0, 50.0]
        self.min_target_mass = min_target_mass

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
        Scalar LMMD loss: kernel terms summed over bandwidths, averaged over
        the active classes (present in the source batch with target mass above
        ``min_target_mass``); 0 if no class is active.
        """
        C = self.num_classes
        if y_src.numel() and (y_src.min() < 0 or y_src.max() >= C):
            raise ValueError(f"source labels must lie in [0, {C}), got range "
                             f"[{y_src.min().item()}, {y_src.max().item()}]")

        # ── per-class sample weights, active classes only ─────────────────────
        y_src_oh = F.one_hot(y_src, C).float()              # (n_s, C)
        src_mass = y_src_oh.sum(0)                          # (C,)
        tgt_mass = y_tgt_prob.sum(0)                        # (C,)
        active = (src_mass > 0) & (tgt_mass > self.min_target_mass)
        n_active = int(active.sum().item())
        if n_active == 0:
            return z_src.sum() * 0.0                        # keeps the graph

        # normalise each active column so weights sum to 1
        ws = y_src_oh[:, active] / src_mass[active]          # (n_s, A)
        wt = y_tgt_prob[:, active] / tgt_mass[active]        # (n_t, A)

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

        return lmmd / n_active


# ─── LMMD Trainer ─────────────────────────────────────────────────────────────

class LMMDTrainer(BaseTrainer):
    """
    Domain adaptation trainer using Local Maximum Mean Discrepancy (LMMD).

    Unlike global MMD which compares marginal distributions, LMMD compares
    per-class subdomains.  Source samples are weighted by their ground-truth
    class labels; target samples are weighted by the model's current softmax
    predictions (treated as stop-gradient soft pseudo-labels), so the target
    side of each class is only as reliable as those predictions.

    Parameters
    ----------
    model         : network with .encode() and .classify() methods
    source_loader : labelled source DataLoader
    target_loader : target DataLoader yielding ``x`` or ``(x, y)``; labels,
                    if present, are used only for ``tgt_acc`` monitoring
    num_classes   : number of output classes
    lmmd_weight   : λ weighting the LMMD term  (total = CE + λ·LMMD)
    lr            : Adam learning rate
    warmup_epochs : epochs of source-only CE pre-training before LMMD begins
    device        : 'cuda', 'mps', or 'cpu' (auto-detected if None)
    lmmd_sigmas   : RBF kernel bandwidths (passed to LMMDLoss)

    An epoch covers the longer loader; during LMMD epochs tensor source and
    target batches are encoded jointly (shared BatchNorm batch statistics).
    ``src_acc`` uses training-mode logits before each update; ``tgt_acc`` is
    measured in eval mode after it (``nan`` without target labels).

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
                loss     = ce
                lmmd_val = 0.0
            else:
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
            "lmmd_loss":  total_lmmd     / n_batches,
            "mmd_loss":   0.0,   # history-format compatibility
            "total_loss": total_loss_sum / n_batches,
            "src_acc":    src_correct / n_src,
            "tgt_acc":    tgt_correct / n_tgt if n_tgt else float("nan"),
        }

    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """Compute accuracy on a labelled DataLoader (evaluation mode)."""
        return evaluate_accuracy(self.model, loader, self.device, domain)
