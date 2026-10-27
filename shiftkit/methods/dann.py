"""
Domain-Adversarial Neural Networks (DANN) with optional Semantic Centroid Alignment.

Introduces a domain discriminator connected to the encoder via a
Gradient Reversal Layer (GRL). The GRL negates gradients during backprop,
forcing the encoder to learn domain-invariant features that fool the
discriminator while still classifying source labels correctly.

Architecture
------------
                 ┌─── classify(z_src) ──► CE loss (label)
encoder(x) ──► z
                 └─── GRL ──► discriminator(z) ──► BCE loss (domain)

The gradient from the domain loss is reversed before it reaches the encoder,
so the encoder is trained to *maximise* domain confusion (minimise the
discriminator's ability to tell source from target).

Optionally, an MSTN-style semantic centroid alignment (a variant of Xie et
al., 2018) can be enabled via ``semantic_weight > 0``. Per-batch source class
centroids (from labels) are pulled toward target class centroids maintained as
a detached exponential moving average over pseudo-labeled target features.
Unlike MSTN, source centroids are not moving averages and target features get
no direct gradient from this term; classes whose target centroid has never
been observed are skipped.

The GRL strength follows an epoch-level sigmoid ramp over the DA epochs (a
variant of the paper's per-iteration training-progress schedule); the
learning rate is constant (no annealing).

References
----------
Ganin, Y., et al. (2016). Domain-Adversarial Training of Neural Networks.
Journal of Machine Learning Research, 17(59), 1–35.
https://jmlr.org/papers/volume17/15-239/15-239.pdf

Xie, S., Zheng, Z., Chen, L., & Chen, C. (2018).
Learning Semantic Representations for Unsupervised Domain Adaptation.
ICML 2018, PMLR 80:5423–5432.
https://proceedings.mlr.press/v80/xie18c.html
"""

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


# ─── Gradient Reversal Layer ──────────────────────────────────────────────────

class _GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Identity in the forward pass; negates and scales gradients in backward.

    Parameters
    ----------
    alpha : reversal strength λ (can be updated during training via .alpha)
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFunction.apply(x, self.alpha)


# ─── Domain Discriminator ─────────────────────────────────────────────────────

class DomainDiscriminator(nn.Module):
    """
    Small MLP that predicts domain (source=0, target=1) from a latent vector.

    Parameters
    ----------
    latent_dim : input size (must match model.latent_dim)
    hidden_dim : hidden layer size
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ─── DANN Trainer ─────────────────────────────────────────────────────────────

class DANNTrainer(BaseTrainer):
    """
    Domain-Adversarial Neural Network trainer (Ganin et al., 2016) with optional
    semantic centroid alignment (Xie et al., 2018).

    A domain discriminator is attached to the encoder output via a Gradient
    Reversal Layer. The encoder is trained jointly to minimise task loss
    (cross-entropy on source labels) and maximise domain confusion (via the
    reversed domain loss), producing domain-invariant latent representations.

    When ``semantic_weight > 0``, an additional centroid alignment loss is added
    (an MSTN-style variant): per-batch source centroids (from ground-truth
    labels) are pulled toward per-class target centroids kept as a detached
    exponential moving average over pseudo-labeled target features.  Only
    classes present in the source batch *and* whose target centroid has been
    initialised (seen at least once) contribute.

    Total loss
    ----------
    L = CE(src) + λ_d · BCE(domain via GRL)
        + λ_s · (1/|K|) Σ_{k∈K} (1/d) ||c_k^src - c_k^tgt||²

    where K is the set of contributing classes and d = latent_dim (the code
    uses the mean squared difference over latent dimensions).

    Parameters
    ----------
    model               : network with .encode() and .classify() methods
                          (must also expose .latent_dim)
    source_loader       : labelled source DataLoader
    target_loader       : target DataLoader yielding ``x`` or ``(x, y)``;
                          labels, if present, are used only for ``tgt_acc``
    domain_weight       : λ_d — weight on the domain adversarial loss
    lr                  : Adam learning rate (shared by model + discriminator)
    alpha               : GRL reversal strength (final value if scheduled,
                          constant after warmup otherwise)
    schedule_alpha      : if True, ramp alpha per epoch with the sigmoid
                          α(p) = alpha · (2/(1+exp(−10p)) − 1), where
                          p = (epoch − warmup_epochs) / (epochs − warmup_epochs)
                          counts DA epochs only (an epoch-level variant of
                          the paper's per-iteration progress schedule).
                          If False, α = alpha for every DA epoch.
    discriminator_hidden: hidden dim of the domain discriminator MLP
    warmup_epochs       : epochs of source-only CE pre-training before
                          adversarial DA begins; alpha is held at 0 during
                          warmup regardless of schedule_alpha
    semantic_weight     : λ_s — weight on the centroid alignment loss.
                          Set to 0.0 (default) to disable.
    centroid_momentum   : β — EMA momentum for updating target centroids.
                          Larger β tracks the current batch more closely.
    num_classes         : number of output classes (required when
                          semantic_weight > 0; source labels must be
                          < num_classes)
    device              : 'cuda', 'mps', or 'cpu' (auto-detected if None)

    An epoch covers the longer loader; during DA epochs tensor source and
    target batches are encoded jointly (shared BatchNorm batch statistics).
    ``src_acc`` uses training-mode logits before each update; ``tgt_acc`` is
    measured in eval mode after it (``nan`` without target labels).  The
    learning rate is constant (no annealing).

    History keys
    ------------
    epoch, ce_loss, domain_loss, semantic_loss, mmd_loss (always 0),
    total_loss, src_acc, tgt_acc
    """

    def __init__(
        self,
        model: nn.Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        domain_weight: float = 1.0,
        lr: float = 1e-3,
        alpha: float = 1.0,
        schedule_alpha: bool = True,
        discriminator_hidden: int = 128,
        warmup_epochs: int = 0,
        semantic_weight: float = 0.0,
        centroid_momentum: float = 0.1,
        num_classes: int = 10,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device) if device else _auto_device()
        self.model = model.to(self.device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.domain_weight = domain_weight
        self.alpha = alpha
        self.schedule_alpha = schedule_alpha
        self.warmup_epochs = warmup_epochs
        self.semantic_weight = semantic_weight
        self.centroid_momentum = centroid_momentum
        self.num_classes = num_classes

        self.grl = GradientReversalLayer(alpha=alpha if not schedule_alpha else 0.0)
        self.discriminator = DomainDiscriminator(
            latent_dim=model.latent_dim,
            hidden_dim=discriminator_hidden,
        ).to(self.device)

        self.ce_loss  = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(
            list(model.parameters()) + list(self.discriminator.parameters()),
            lr=lr,
        )

        # Target centroid buffer — shape (num_classes, latent_dim), plus a mask
        # of classes whose centroid has been initialised from real features
        if self.semantic_weight > 0.0:
            self.tgt_centroids = torch.zeros(
                num_classes, model.latent_dim, device=self.device
            )
            self.tgt_centroid_seen = torch.zeros(
                num_classes, dtype=torch.bool, device=self.device
            )
        else:
            self.tgt_centroids = None
            self.tgt_centroid_seen = None

        self.history: List[dict] = []

    # ------------------------------------------------------------------
    def fit(self, epochs: int = 10) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        da_epochs = max(epochs - self.warmup_epochs, 1)
        for epoch in range(1, epochs + 1):
            is_warmup = epoch <= self.warmup_epochs
            if is_warmup:
                self.grl.alpha = 0.0
            elif self.schedule_alpha:
                # ramp α only over the DA phase
                p = (epoch - self.warmup_epochs) / da_epochs
                self.grl.alpha = self.alpha * (
                    2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p)).item()) - 1.0
                )
            else:
                # fixed strength: restore it after any warmup epochs
                self.grl.alpha = self.alpha
            stats = self._train_epoch(epoch, epochs, is_warmup)
            self.history.append(stats)
            phase = "warmup" if is_warmup else "DANN "
            da_str = ""
            if not is_warmup:
                da_str = f"  Domain={stats['domain_loss']:.4f}"
                if self.semantic_weight > 0.0:
                    da_str += f"  Sem={stats['semantic_loss']:.4f}"
            print(
                f"[{epoch:>3}/{epochs}] [{phase}]  "
                f"CE={stats['ce_loss']:.4f}{da_str}  "
                f"Total={stats['total_loss']:.4f}  "
                f"Src={stats['src_acc']*100:.1f}%  "
                f"Tgt={stats['tgt_acc']*100:.1f}%"
            )
        return self.history

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int, total_epochs: int, warmup: bool = False) -> dict:
        self.model.train()
        self.discriminator.train()

        total_ce = total_domain = total_semantic = total_loss_sum = 0.0
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
                # ── warmup: source-only CE loss ───────────────────────────
                loss = ce
                domain_val = 0.0
                semantic_loss = torch.tensor(0.0, device=self.device)

            else:
                # domain adversarial loss — source=0, target=1
                z_all    = torch.cat([z_src, z_tgt], dim=0)
                z_rev    = self.grl(z_all)
                d_logits = self.discriminator(z_rev).squeeze(1)
                d_labels = torch.cat([
                    torch.zeros(z_src.size(0), device=self.device),
                    torch.ones( z_tgt.size(0), device=self.device),
                ])
                domain_loss = self.bce_loss(d_logits, d_labels)
                domain_val  = domain_loss.item()
                loss = ce + self.domain_weight * domain_loss

                # ── semantic centroid alignment (MSTN-style variant) ──────
                semantic_loss = torch.tensor(0.0, device=self.device)
                if self.semantic_weight > 0.0:
                    if y_src.numel() and (y_src.min() < 0 or y_src.max() >= self.num_classes):
                        raise ValueError(
                            f"source labels must lie in [0, num_classes={self.num_classes}), "
                            f"got range [{y_src.min().item()}, {y_src.max().item()}]"
                        )
                    with torch.no_grad():
                        pseudo_y = self.model.classify(z_tgt).argmax(1)
                        for k in range(self.num_classes):
                            mask_t = (pseudo_y == k)
                            if mask_t.any():
                                batch_mean = z_tgt[mask_t].detach().mean(0)
                                if self.tgt_centroid_seen[k]:
                                    self.tgt_centroids[k].mul_(1.0 - self.centroid_momentum).add_(
                                        batch_mean * self.centroid_momentum
                                    )
                                else:
                                    # first observation initialises the centroid
                                    self.tgt_centroids[k].copy_(batch_mean)
                                    self.tgt_centroid_seen[k] = True
                    n_present = 0
                    for k in range(self.num_classes):
                        mask_s = (y_src == k)
                        # skip classes with no source sample or no target centroid yet
                        if mask_s.any() and self.tgt_centroid_seen[k]:
                            c_src = z_src[mask_s].mean(0)
                            semantic_loss = semantic_loss + (
                                (c_src - self.tgt_centroids[k]) ** 2
                            ).mean()
                            n_present += 1
                    if n_present > 0:
                        semantic_loss = semantic_loss / n_present
                    loss = loss + self.semantic_weight * semantic_loss
                # ──────────────────────────────────────────────────────────

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_ce         += ce.item()
            total_domain     += domain_val
            total_semantic   += semantic_loss.item()
            total_loss_sum   += loss.item()
            # src_acc: training-mode logits of this batch, before the update
            src_correct      += (logits.argmax(1) == y_src).sum().item()
            n_src            += y_src.size(0)

            # tgt_acc: eval mode (no BN update / dropout), after the update
            if y_tgt is not None:
                tgt_correct += batch_accuracy(self.model, x_tgt, y_tgt)
                n_tgt       += y_tgt.size(0)

        return {
            "epoch":          epoch,
            "ce_loss":        total_ce         / n_batches,
            "domain_loss":    total_domain     / n_batches,
            "semantic_loss":  total_semantic   / n_batches,
            "mmd_loss":       0.0,              # compatibility with plot_training_history
            "total_loss":     total_loss_sum   / n_batches,
            "src_acc":        src_correct / n_src,
            "tgt_acc":        tgt_correct / n_tgt if n_tgt else float("nan"),
        }

    # ------------------------------------------------------------------
    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """Compute accuracy on a labelled DataLoader (evaluation mode)."""
        return evaluate_accuracy(self.model, loader, self.device, domain)
