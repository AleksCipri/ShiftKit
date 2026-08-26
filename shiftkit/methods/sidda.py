"""
SIDDA: SInkhorn Dynamic Domain Adaptation (Ciprijanovic et al., 2025).

Uses the Sinkhorn divergence as the domain alignment loss with two key ideas:

1. **Dynamic regularisation** — the Sinkhorn blur parameter σ is adapted each
   batch based on the maximum pairwise distance between source and target latent
   features, so the optimal-transport plan automatically adjusts to the current
   state of the encoder.

2. **Learnable loss weighting** — two scalar parameters η₁ (CE) and η₂ (DA) are
   jointly optimised with the model.  The multi-task loss formulation:

       ℒ = (1/2η₁²)ℒ_CE + (1/2η₂²)ℒ_DA + log(|η₁||η₂|)

   automatically balances classification and domain alignment without a fixed λ.
   The log term prevents η from collapsing to zero or growing unboundedly.

3. **Optional latent-space instance reweighting** (``use_potentials=True``) —
   geomloss computes Kantorovich dual potentials (F, G) as a byproduct of the
   Sinkhorn iterations.  F[i] measures how expensive it is to move source point
   z_i to match the target distribution in latent space.  With
   ``use_potentials=True`` these potentials reweight the per-sample CE loss:

       w_i = softmax(-F / τ)  →  ℒ_CE = Σ w_i · CE(f(z_i), y_i)

   This upweights source samples already close to the target (low transport
   cost), focusing the classifier on the transferable subset of source data.
   The weighting is purely in latent space and updates every batch as the
   encoder trains — no separate QP or input-space computation required.

An optional **warmup phase** trains the encoder on source classification only
(no DA loss) for a fixed number of epochs before domain adaptation begins.
This ensures the encoder produces meaningful representations before alignment
is attempted — equivariant networks typically need a shorter warmup than CNNs.

Architecture
------------
encoder(x) ──► z_src ──► classify(z_src) ──► CE loss  ──┐
encoder(x) ──► z_tgt ──►                                  ├─► weighted sum
               Sinkhorn(z_src, z_tgt) ──► DA loss  ───────┘

Reference
---------
Ciprijanovic, A., Lewis, A., Pedro, K., Downey, E., Nord, B., & Stark, A. (2025).
SIDDA: SInkhorn Dynamic Domain Adaptation for Image Classification with
Equivariant Neural Networks.
arXiv:2501.14048.
https://arxiv.org/abs/2501.14048

Sinkhorn divergence (optimal transport background):
Feydy, J., Séjourné, T., Vialard, F.-X., Amari, S., Trouvé, A., &
Peyré, G. (2019). Interpolating between Optimal Transport and MMD using
Sinkhorn Divergences. AISTATS 2019. https://arxiv.org/abs/1810.08278
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List
from shiftkit.methods.node_batch import (
    is_node_graph_batch,
    latents_and_targets,
    node_regress_preds,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dynamic_blur(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    scale: float = 0.05,
    floor: float = 0.01,
) -> float:
    """
    Compute the per-batch Sinkhorn blur σ.

    Following the paper:  σ = max(scale · max_{i,j}||z_i − z*_j||₂, floor)

    Layer normalisation is applied before computing distances to prevent
    outlier features from inflating σ and dominating the OT plan.
    """
    with torch.no_grad():
        z_s = F.layer_norm(z_src.detach().float(), [z_src.shape[-1]])
        z_t = F.layer_norm(z_tgt.detach().float(), [z_tgt.shape[-1]])
        max_dist = torch.cdist(z_s, z_t).max().item()
    return max(scale * max_dist, floor)


# ─── SIDDA Trainer ────────────────────────────────────────────────────────────

class SIDDATrainer:
    """
    SIDDA trainer (Ciprijanovic et al., 2025).

    Trains the encoder jointly with a Sinkhorn domain alignment loss and two
    learnable loss-weighting parameters (η₁, η₂).  The Sinkhorn regularisation
    strength σ is adapted each batch from the latent feature distances.

    Total loss (after warmup)
    -------------------------
    ℒ = (1/2η₁²)·CE(z_src, y_src)
      + (1/2η₂²)·Sinkhorn_σ(z_src, z_tgt)
      + log(|η₁|·|η₂|)

    Warmup phase
    ------------
    For the first ``warmup_epochs`` epochs only CE loss is used (no DA).
    This builds a good source-domain representation before alignment begins.
    η₁ and η₂ are not updated during warmup.

    Parameters
    ----------
    model                : network with .encode() and .classify() methods
                           (must expose .latent_dim)
    source_loader        : labelled source DataLoader
    target_loader        : target DataLoader (labels used for tgt_acc tracking only)
    lr                   : AdamW learning rate
    weight_decay         : AdamW weight decay
    warmup_epochs        : epochs of source-only pre-training before DA begins
    sigma_scale          : scale factor for dynamic blur (default 0.05 from paper)
    sigma_floor          : minimum blur value to prevent degenerate OT (default 0.01)
    grad_clip            : gradient clipping max-norm (default 10.0 from paper)
    use_potentials       : if True, use Kantorovich dual potentials from geomloss
                           to reweight per-sample CE loss in latent space
                           (default False)
    potential_temperature: temperature τ for the softmax reweighting
                           w_i = softmax(-F_i / τ).  Lower τ = sharper focus on
                           already-aligned source samples.  (default 1.0)
    weight_ot            : if True, reweight the Sinkhorn transport plan itself
                           using an EMA of the previous batch's potentials.
                           This focuses OT alignment on the overlapping region of
                           the two distributions rather than aligning everything
                           equally.  Requires one extra Sinkhorn solve only on the
                           very first batch of each epoch (to seed the EMA).
                           Can be used alone or combined with use_potentials.
                           (default False)
    ot_ema_momentum      : EMA momentum α for updating OT weights between batches.
                           w_t = α·w_{t-1} + (1-α)·softmax(-f_t / τ).
                           Higher α = smoother, slower-changing weights.
                           (default 0.9)
    device               : 'cuda', 'mps', or 'cpu' (auto-detected if None)

    History keys
    ------------
    epoch, ce_loss, da_loss, mmd_loss (always 0), total_loss,
    src_acc, tgt_acc, eta1, eta2, sigma, mean_potential
    """

    def __init__(
        self,
        model: nn.Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        warmup_epochs: int = 0,
        sigma_scale: float = 0.05,
        sigma_floor: float = 0.01,
        grad_clip: float = 10.0,
        use_potentials: bool = False,
        potential_temperature: float = 1.0,
        weight_ot: bool = False,
        ot_ema_momentum: float = 0.9,
        device: Optional[str] = None,
    ):
        try:
            from geomloss import SamplesLoss
            self._SamplesLoss = SamplesLoss
        except ImportError as e:
            raise ImportError(
                "geomloss is required for SIDDATrainer. "
                "Install it with:  pip install geomloss"
            ) from e

        self.device = torch.device(device) if device else _auto_device()
        self.model = model.to(self.device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.warmup_epochs = warmup_epochs
        self.sigma_scale = sigma_scale
        self.sigma_floor = sigma_floor
        self.grad_clip = grad_clip
        self.use_potentials = use_potentials
        self.potential_temperature = potential_temperature
        self.weight_ot = weight_ot
        self.ot_ema_momentum = ot_ema_momentum

        # EMA buffers for OT weighting — lazily initialised on first DA batch
        self._ema_w_src: Optional[torch.Tensor] = None
        self._ema_w_tgt: Optional[torch.Tensor] = None

        # Learnable loss-weighting scalars, initialised to 1
        self.eta1 = nn.Parameter(torch.ones(1, device=self.device))  # CE
        self.eta2 = nn.Parameter(torch.ones(1, device=self.device))  # DA

        self.ce_loss = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            list(model.parameters()) + [self.eta1, self.eta2],
            lr=lr,
            weight_decay=weight_decay,
        )

        self.history: List[dict] = []

    # ------------------------------------------------------------------
    def fit(self, epochs: int = 50) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        for epoch in range(1, epochs + 1):
            is_warmup = epoch <= self.warmup_epochs
            stats = self._train_epoch(epoch, epochs, is_warmup)
            self.history.append(stats)

            phase = "warmup" if is_warmup else "SIDDA "
            da_str = (
                f"  DA={stats['da_loss']:.4f}"
                f"  η₁={stats['eta1']:.3f}  η₂={stats['eta2']:.3f}"
                f"  σ={stats['sigma']:.4f}"
                if not is_warmup else ""
            )
            print(
                f"[{epoch:>3}/{epochs}] [{phase}]  "
                f"CE={stats['ce_loss']:.4f}{da_str}  "
                f"Total={stats['total_loss']:.4f}  "
                f"Src={stats['src_acc']*100:.1f}%  "
                f"Tgt={stats['tgt_acc']*100:.1f}%"
            )
        return self.history

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int, total_epochs: int, warmup: bool) -> dict:
        self.model.train()

        total_ce = total_da = total_loss_sum = 0.0
        src_correct = tgt_correct = n_src = n_tgt = 0
        last_sigma = 0.0
        total_potential = 0.0

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
            ce = self.ce_loss(logits, y_src)

            if warmup:
                # ── warmup: source-only CE loss, no DA ────────────────
                loss = ce
                da_val = 0.0

            else:
                # ── SIDDA loss ─────────────────────────────────────────
                # 1. Dynamic Sinkhorn blur
                sigma = _dynamic_blur(z_src, z_tgt, self.sigma_scale, self.sigma_floor)
                last_sigma = sigma

                # need_potentials: True if either CE reweighting or EMA OT weighting is on
                need_potentials = self.use_potentials or self.weight_ot

                # 2. Sinkhorn divergence S_σ(z_src, z_tgt)
                sinkhorn = self._SamplesLoss(
                    "sinkhorn", p=2, blur=sigma, scaling=0.9,
                    potentials=need_potentials,
                )

                if need_potentials:
                    # ── weighted transport plan (EMA) ──────────────────
                    ema_ready = (
                        self.weight_ot
                        and self._ema_w_src is not None
                        and self._ema_w_src.shape[-1] == z_src.shape[0]
                        and self._ema_w_tgt.shape[-1] == z_tgt.shape[0]
                    )
                    if ema_ready:
                        # Use previous-batch EMA weights as the discrete measures:
                        # α = Σ w_src[i] δ_{z_src[i]},  β = Σ w_tgt[j] δ_{z_tgt[j]}
                        # geomloss normalises internally, so raw softmax values work.
                        f_pot, g_pot = sinkhorn(
                            self._ema_w_src, z_src,
                            self._ema_w_tgt, z_tgt,
                        )
                    else:
                        # First batch, or batch size changed — run uniform, seed EMA below
                        f_pot, g_pot = sinkhorn(z_src, z_tgt)

                    da_loss = f_pot.mean() + g_pot.mean()

                    # ── update EMA buffers from this batch's potentials ─
                    if self.weight_ot:
                        new_w_src = f_pot.detach().reshape(-1).div(-self.potential_temperature).softmax(dim=-1)
                        new_w_tgt = g_pot.detach().reshape(-1).div(-self.potential_temperature).softmax(dim=-1)
                        if ema_ready:
                            mom = self.ot_ema_momentum
                            self._ema_w_src = mom * self._ema_w_src + (1 - mom) * new_w_src
                            self._ema_w_tgt = mom * self._ema_w_tgt + (1 - mom) * new_w_tgt
                        else:
                            # seed or reset after batch-size change
                            self._ema_w_src = new_w_src
                            self._ema_w_tgt = new_w_tgt

                    # ── reweight per-sample CE (if requested) ──────────
                    if self.use_potentials:
                        w_ce = f_pot.detach().reshape(-1).div(-self.potential_temperature).softmax(dim=-1)
                        ce_per_sample = F.cross_entropy(logits, y_src, reduction="none")
                        ce = (w_ce * ce_per_sample).sum()

                    total_potential += f_pot.detach().mean().item()

                else:
                    da_loss = sinkhorn(z_src, z_tgt)

                da_val = da_loss.item()

                # 3. Weighted combination with learnable etas
                loss = (
                    (1.0 / (2.0 * self.eta1 ** 2)) * ce
                    + (1.0 / (2.0 * self.eta2 ** 2)) * da_loss
                    + torch.log(self.eta1.abs() * self.eta2.abs())
                )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + [self.eta1, self.eta2],
                self.grad_clip,
            )
            self.optimizer.step()

            # Enforce η constraints: η₁ ≥ 1e-3,  η₂ ≥ 0.25·η₁
            if not warmup:
                with torch.no_grad():
                    self.eta1.clamp_(min=1e-3)
                    self.eta2.clamp_(min=0.25 * self.eta1.item())

            total_ce       += ce.item()
            total_da       += da_val
            total_loss_sum += loss.item()
            src_correct    += (logits.argmax(1) == y_src).sum().item()
            n_src          += y_src.size(0)

            with torch.no_grad():
                tgt_correct += (self.model(x_tgt).argmax(1) == y_tgt).sum().item()
                n_tgt       += y_tgt.size(0)

        return {
            "epoch":           epoch,
            "ce_loss":         total_ce       / n_batches,
            "da_loss":         total_da       / n_batches,
            "mmd_loss":        0.0,            # history-format compatibility
            "total_loss":      total_loss_sum / n_batches,
            "src_acc":         src_correct / n_src,
            "tgt_acc":         tgt_correct / n_tgt,
            "eta1":            self.eta1.item(),
            "eta2":            self.eta2.item(),
            "sigma":           last_sigma,
            "mean_potential":  total_potential / n_batches,
        }

    # ------------------------------------------------------------------
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


# ─── SIDDA Regression Trainer ─────────────────────────────────────────────────

class SIDDARegressionTrainer:
    """
    SIDDA trainer for regression tasks.

    Identical to SIDDATrainer but uses MSE regression loss instead of
    cross-entropy, and expects a model with .encode() and .regress() methods
    (e.g. MLPRegressor).  The Sinkhorn OT alignment and optional latent-space
    instance reweighting (use_potentials, weight_ot) work identically.

    Total loss (after warmup)
    -------------------------
    ℒ = (1/2η₁²)·MSE(z_src, y_src)
      + (1/2η₂²)·Sinkhorn_σ(z_src, z_tgt)
      + log(|η₁|·|η₂|)

    Parameters
    ----------
    model                : network with .encode() and .regress() methods
    source_loader        : labelled source DataLoader with float targets
    target_loader        : target DataLoader (labels used for tgt_rmse only)
    lr                   : AdamW learning rate
    weight_decay         : AdamW weight decay
    warmup_epochs        : epochs of source-only MSE pre-training before DA
    sigma_scale          : scale factor for dynamic blur (default 0.05)
    sigma_floor          : minimum blur value (default 0.01)
    grad_clip            : gradient clipping max-norm (default 10.0)
    use_potentials       : reweight per-sample MSE loss via Kantorovich potentials
    potential_temperature: temperature τ for softmax reweighting (default 1.0)
    weight_ot            : reweight transport plan via EMA of previous potentials
    ot_ema_momentum      : EMA momentum for OT weight updates (default 0.9)
    device               : auto-detected if None

    History keys
    ------------
    epoch, mse_loss, da_loss, total_loss, src_rmse, tgt_rmse,
    eta1, eta2, sigma, mean_potential
    """

    def __init__(
        self,
        model: nn.Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        warmup_epochs: int = 0,
        sigma_scale: float = 0.05,
        sigma_floor: float = 0.01,
        grad_clip: float = 10.0,
        use_potentials: bool = False,
        potential_temperature: float = 1.0,
        weight_ot: bool = False,
        ot_ema_momentum: float = 0.9,
        device: Optional[str] = None,
    ):
        try:
            from geomloss import SamplesLoss
            self._SamplesLoss = SamplesLoss
        except ImportError as e:
            raise ImportError(
                "geomloss is required for SIDDARegressionTrainer. "
                "Install it with:  pip install geomloss"
            ) from e

        self.device = torch.device(device) if device else _auto_device()
        self.model = model.to(self.device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.warmup_epochs = warmup_epochs
        self.sigma_scale = sigma_scale
        self.sigma_floor = sigma_floor
        self.grad_clip = grad_clip
        self.use_potentials = use_potentials
        self.potential_temperature = potential_temperature
        self.weight_ot = weight_ot
        self.ot_ema_momentum = ot_ema_momentum

        self._ema_w_src: Optional[torch.Tensor] = None
        self._ema_w_tgt: Optional[torch.Tensor] = None

        self.eta1 = nn.Parameter(torch.ones(1, device=self.device))
        self.eta2 = nn.Parameter(torch.ones(1, device=self.device))

        self.mse_loss = nn.MSELoss()

        self.optimizer = optim.AdamW(
            list(model.parameters()) + [self.eta1, self.eta2],
            lr=lr,
            weight_decay=weight_decay,
        )

        self.history: List[dict] = []

    # ------------------------------------------------------------------
    def fit(self, epochs: int = 50) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        for epoch in range(1, epochs + 1):
            is_warmup = epoch <= self.warmup_epochs
            stats = self._train_epoch(epoch, epochs, is_warmup)
            self.history.append(stats)

            phase = "warmup" if is_warmup else "SIDDA "
            da_str = (
                f"  DA={stats['da_loss']:.4f}"
                f"  η₁={stats['eta1']:.3f}  η₂={stats['eta2']:.3f}"
                f"  σ={stats['sigma']:.4f}"
                if not is_warmup else ""
            )
            print(
                f"[{epoch:>3}/{epochs}] [{phase}]  "
                f"MSE={stats['mse_loss']:.4f}{da_str}  "
                f"Total={stats['total_loss']:.4f}  "
                f"SrcRMSE={stats['src_rmse']:.4f}  "
                f"TgtRMSE={stats['tgt_rmse']:.4f}"
            )
        return self.history

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int, total_epochs: int, warmup: bool) -> dict:
        self.model.train()

        total_mse = total_da = total_loss_sum = 0.0
        src_se = tgt_se = 0.0
        n_src = n_tgt = 0
        last_sigma = 0.0
        total_potential = 0.0

        n_batches = min(len(self.source_loader), len(self.target_loader))
        loader = zip(self.source_loader, self.target_loader)

        for (x_src, y_src), (x_tgt, y_tgt) in tqdm(
            loader, total=n_batches,
            desc=f"Epoch {epoch}/{total_epochs}", leave=False
        ):
            # One dispatch for both batch kinds: for a NodeGraphBatch the graph
            # is encoded once and z/y are indexed to the split's nodes.
            z_src, y_src = latents_and_targets(self.model, x_src, y_src, self.device)
            z_tgt, y_tgt = latents_and_targets(self.model, x_tgt, y_tgt, self.device)
            y_src, y_tgt = y_src.float(), y_tgt.float()

            pred_src = self.model.regress(z_src).view_as(y_src)

            mse = self.mse_loss(pred_src, y_src)

            if warmup:
                loss = mse
                da_val = 0.0

            else:
                sigma = _dynamic_blur(z_src, z_tgt, self.sigma_scale, self.sigma_floor)
                last_sigma = sigma

                need_potentials = self.use_potentials or self.weight_ot
                sinkhorn = self._SamplesLoss(
                    "sinkhorn", p=2, blur=sigma, scaling=0.9,
                    potentials=need_potentials,
                )

                if need_potentials:
                    ema_ready = (
                        self.weight_ot
                        and self._ema_w_src is not None
                        and self._ema_w_src.shape[-1] == z_src.shape[0]
                        and self._ema_w_tgt.shape[-1] == z_tgt.shape[0]
                    )
                    if ema_ready:
                        f_pot, g_pot = sinkhorn(
                            self._ema_w_src, z_src,
                            self._ema_w_tgt, z_tgt,
                        )
                    else:
                        f_pot, g_pot = sinkhorn(z_src, z_tgt)

                    da_loss = f_pot.mean() + g_pot.mean()

                    if self.weight_ot:
                        new_w_src = f_pot.detach().reshape(-1).div(-self.potential_temperature).softmax(dim=-1)
                        new_w_tgt = g_pot.detach().reshape(-1).div(-self.potential_temperature).softmax(dim=-1)
                        if ema_ready:
                            mom = self.ot_ema_momentum
                            self._ema_w_src = mom * self._ema_w_src + (1 - mom) * new_w_src
                            self._ema_w_tgt = mom * self._ema_w_tgt + (1 - mom) * new_w_tgt
                        else:
                            self._ema_w_src = new_w_src
                            self._ema_w_tgt = new_w_tgt

                    if self.use_potentials:
                        w_mse = f_pot.detach().reshape(-1).div(-self.potential_temperature).softmax(dim=-1)
                        mse_per = F.mse_loss(pred_src.squeeze(), y_src.squeeze(), reduction="none")
                        mse = (w_mse * mse_per).sum()

                    total_potential += f_pot.detach().mean().item()

                else:
                    da_loss = sinkhorn(z_src, z_tgt)

                da_val = da_loss.item()

                loss = (
                    (1.0 / (2.0 * self.eta1 ** 2)) * mse
                    + (1.0 / (2.0 * self.eta2 ** 2)) * da_loss
                    + torch.log(self.eta1.abs() * self.eta2.abs())
                )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + [self.eta1, self.eta2],
                self.grad_clip,
            )
            self.optimizer.step()

            if not warmup:
                with torch.no_grad():
                    self.eta1.clamp_(min=1e-3)
                    self.eta2.clamp_(min=0.25 * self.eta1.item())

            total_mse      += mse.item()
            total_da       += da_val
            total_loss_sum += loss.item()
            src_se += ((pred_src.detach() - y_src) ** 2).sum().item()
            n_src  += y_src.size(0)

            with torch.no_grad():
                pred_tgt = self.model.regress(z_tgt).view_as(y_tgt)
                tgt_se  += ((pred_tgt - y_tgt) ** 2).sum().item()
                n_tgt   += y_tgt.size(0)

        return {
            "epoch":          epoch,
            "mse_loss":       total_mse       / n_batches,
            "da_loss":        total_da        / n_batches,
            "total_loss":     total_loss_sum  / n_batches,
            "src_rmse":       (src_se / n_src) ** 0.5,
            "tgt_rmse":       (tgt_se / n_tgt) ** 0.5,
            "eta1":           self.eta1.item(),
            "eta2":           self.eta2.item(),
            "sigma":          last_sigma,
            "mean_potential": total_potential / n_batches,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """Compute MSE, RMSE, and R² on a labelled DataLoader."""
        self.model.eval()
        ys, preds = [], []
        for x, y in loader:
            if is_node_graph_batch(x):
                # Predict on the split's nodes only — encoding x.graph alone
                # would return one prediction per node in the whole graph.
                x = x.to(self.device)
                preds.append(node_regress_preds(self.model, x))
                ys.append(x.y.float())
                continue
            x, y = x.to(self.device), y.to(self.device).float()
            preds.append(self.model.regress(self.model.encode(x)).view_as(y))
            ys.append(y)
        ys    = torch.cat(ys)
        preds = torch.cat(preds)
        mse    = ((preds - ys) ** 2).mean().item()
        ss_res = ((preds - ys) ** 2).sum().item()
        ss_tot = ((ys - ys.mean()) ** 2).sum().item()
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            "domain":    domain,
            "mse":       mse,
            "rmse":      mse ** 0.5,
            "r2":        r2,
            "n_samples": ys.size(0),
        }


# ─── SIDDA Regression Trainer with heteroscedastic (mean, variance) head ──────

def _split_mean_var(out: torch.Tensor, eps: float = 1e-8):
    """Split a [..., 2] regression-head output into (mean, var).

    The second column is the raw predicted std (Jeffrey & Wandelt 2020
    parameterization) — squaring it gives a variance that's always >= 0
    without needing an exp/softplus transform.
    """
    mean = out[..., 0]
    var = out[..., 1].pow(2) + eps
    return mean, var


def _moment_network_loss(mean, var, y, weights=None, eps: float = 1e-8):
    """
    Likelihood-free moment-matching loss (L1, L2) from Jeffrey & Wandelt (2020),
    as used by Villaescusa-Navarro et al. 2022 for halo-mass posterior moments.
    See shiftkit.methods.regression._moment_network_loss for the full derivation
    (duplicated here to keep this module's SIDDA-specific dependencies self-contained).
    """
    if weights is None:
        weights = torch.full_like(y, 1.0 / y.numel())
    sq_resid = (y - mean) ** 2
    l1 = torch.log((weights * sq_resid).sum() + eps)
    l2 = torch.log((weights * (sq_resid - var) ** 2).sum() + eps)
    return l1, l2


class SIDDAGaussianRegressionTrainer:
    """
    SIDDA regression trainer with a heteroscedastic (mean, variance) head.

    Identical to SIDDARegressionTrainer except the regression term is the
    likelihood-free moment-matching loss (L1 + L2, see _moment_network_loss)
    on (mean, variance) instead of MSE — the model must be built with
    predict_var=True (regress(z) returns 2 columns: [mean, std]).
    The Sinkhorn OT alignment, η₁/η₂ loss weighting, and optional
    use_potentials/weight_ot reweighting are unchanged in structure, just fed
    a per-sample-weighted L1+L2 instead of per-sample MSE where reweighting
    is used.

    Total loss (after warmup)
    -------------------------
    ℒ = (1/2η₁²)·(L1 + L2)(mean_src, var_src, y_src)
      + (1/2η₂²)·Sinkhorn_σ(z_src, z_tgt)
      + log(|η₁|·|η₂|)

    History keys
    ------------
    epoch, l1_loss, l2_loss, da_loss, total_loss, src_rmse, tgt_rmse,
    eta1, eta2, sigma, mean_potential
    """

    def __init__(
        self,
        model: nn.Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        warmup_epochs: int = 0,
        sigma_scale: float = 0.05,
        sigma_floor: float = 0.01,
        grad_clip: float = 10.0,
        use_potentials: bool = False,
        potential_temperature: float = 1.0,
        weight_ot: bool = False,
        ot_ema_momentum: float = 0.9,
        var_eps: float = 1e-6,
        device: Optional[str] = None,
    ):
        try:
            from geomloss import SamplesLoss
            self._SamplesLoss = SamplesLoss
        except ImportError as e:
            raise ImportError(
                "geomloss is required for SIDDAGaussianRegressionTrainer. "
                "Install it with:  pip install geomloss"
            ) from e

        self.device = torch.device(device) if device else _auto_device()
        self.model = model.to(self.device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.warmup_epochs = warmup_epochs
        self.sigma_scale = sigma_scale
        self.sigma_floor = sigma_floor
        self.grad_clip = grad_clip
        self.use_potentials = use_potentials
        self.potential_temperature = potential_temperature
        self.weight_ot = weight_ot
        self.ot_ema_momentum = ot_ema_momentum
        self.var_eps = var_eps

        self._ema_w_src: Optional[torch.Tensor] = None
        self._ema_w_tgt: Optional[torch.Tensor] = None

        self.eta1 = nn.Parameter(torch.ones(1, device=self.device))
        self.eta2 = nn.Parameter(torch.ones(1, device=self.device))

        self.optimizer = optim.AdamW(
            list(model.parameters()) + [self.eta1, self.eta2],
            lr=lr,
            weight_decay=weight_decay,
        )

        self.history: List[dict] = []

    # ------------------------------------------------------------------
    def fit(self, epochs: int = 50) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        for epoch in range(1, epochs + 1):
            is_warmup = epoch <= self.warmup_epochs
            stats = self._train_epoch(epoch, epochs, is_warmup)
            self.history.append(stats)

            phase = "warmup" if is_warmup else "SIDDA "
            da_str = (
                f"  DA={stats['da_loss']:.4f}"
                f"  eta1={stats['eta1']:.3f}  eta2={stats['eta2']:.3f}"
                f"  sigma={stats['sigma']:.4f}"
                if not is_warmup else ""
            )
            print(
                f"[{epoch:>3}/{epochs}] [{phase}]  "
                f"L1={stats['l1_loss']:.4f}  L2={stats['l2_loss']:.4f}{da_str}  "
                f"Total={stats['total_loss']:.4f}  "
                f"SrcRMSE={stats['src_rmse']:.4f}  "
                f"TgtRMSE={stats['tgt_rmse']:.4f}"
            )
        return self.history

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int, total_epochs: int, warmup: bool) -> dict:
        self.model.train()

        total_l1 = total_l2 = total_da = total_loss_sum = 0.0
        src_se = tgt_se = 0.0
        n_src = n_tgt = 0
        last_sigma = 0.0
        total_potential = 0.0

        n_batches = min(len(self.source_loader), len(self.target_loader))
        loader = zip(self.source_loader, self.target_loader)

        for (x_src, y_src), (x_tgt, y_tgt) in tqdm(
            loader, total=n_batches,
            desc=f"Epoch {epoch}/{total_epochs}", leave=False
        ):
            z_src, y_src = latents_and_targets(self.model, x_src, y_src, self.device)
            z_tgt, y_tgt = latents_and_targets(self.model, x_tgt, y_tgt, self.device)
            y_src, y_tgt = y_src.float(), y_tgt.float()

            out_src = self.model.regress(z_src)
            mean_src, var_src = _split_mean_var(out_src, self.var_eps)
            y_src = y_src.view_as(mean_src)

            l1, l2 = _moment_network_loss(mean_src, var_src, y_src, eps=self.var_eps)
            reg_loss = l1 + l2

            if warmup:
                loss = reg_loss
                da_val = 0.0

            else:
                sigma = _dynamic_blur(z_src, z_tgt, self.sigma_scale, self.sigma_floor)
                last_sigma = sigma

                need_potentials = self.use_potentials or self.weight_ot
                sinkhorn = self._SamplesLoss(
                    "sinkhorn", p=2, blur=sigma, scaling=0.9,
                    potentials=need_potentials,
                )

                if need_potentials:
                    ema_ready = (
                        self.weight_ot
                        and self._ema_w_src is not None
                        and self._ema_w_src.shape[-1] == z_src.shape[0]
                        and self._ema_w_tgt.shape[-1] == z_tgt.shape[0]
                    )
                    if ema_ready:
                        f_pot, g_pot = sinkhorn(
                            self._ema_w_src, z_src,
                            self._ema_w_tgt, z_tgt,
                        )
                    else:
                        f_pot, g_pot = sinkhorn(z_src, z_tgt)

                    da_loss = f_pot.mean() + g_pot.mean()

                    if self.weight_ot:
                        new_w_src = f_pot.detach().reshape(-1).div(-self.potential_temperature).softmax(dim=-1)
                        new_w_tgt = g_pot.detach().reshape(-1).div(-self.potential_temperature).softmax(dim=-1)
                        if ema_ready:
                            mom = self.ot_ema_momentum
                            self._ema_w_src = mom * self._ema_w_src + (1 - mom) * new_w_src
                            self._ema_w_tgt = mom * self._ema_w_tgt + (1 - mom) * new_w_tgt
                        else:
                            self._ema_w_src = new_w_src
                            self._ema_w_tgt = new_w_tgt

                    if self.use_potentials:
                        w_reg = f_pot.detach().reshape(-1).div(-self.potential_temperature).softmax(dim=-1)
                        l1, l2 = _moment_network_loss(mean_src, var_src, y_src, weights=w_reg, eps=self.var_eps)
                        reg_loss = l1 + l2

                    total_potential += f_pot.detach().mean().item()

                else:
                    da_loss = sinkhorn(z_src, z_tgt)

                da_val = da_loss.item()

                loss = (
                    (1.0 / (2.0 * self.eta1 ** 2)) * reg_loss
                    + (1.0 / (2.0 * self.eta2 ** 2)) * da_loss
                    + torch.log(self.eta1.abs() * self.eta2.abs())
                )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + [self.eta1, self.eta2],
                self.grad_clip,
            )
            self.optimizer.step()

            if not warmup:
                with torch.no_grad():
                    self.eta1.clamp_(min=1e-3)
                    self.eta2.clamp_(min=0.25 * self.eta1.item())

            total_l1       += l1.item()
            total_l2       += l2.item()
            total_da       += da_val
            total_loss_sum += loss.item()
            src_se += ((mean_src.detach() - y_src) ** 2).sum().item()
            n_src  += y_src.size(0)

            with torch.no_grad():
                out_tgt = self.model.regress(z_tgt)
                mean_tgt, _ = _split_mean_var(out_tgt, self.var_eps)
                y_tgt = y_tgt.view_as(mean_tgt)
                tgt_se += ((mean_tgt - y_tgt) ** 2).sum().item()
                n_tgt  += y_tgt.size(0)

        return {
            "epoch":          epoch,
            "l1_loss":        total_l1        / n_batches,
            "l2_loss":        total_l2        / n_batches,
            "da_loss":        total_da        / n_batches,
            "total_loss":     total_loss_sum  / n_batches,
            "src_rmse":       (src_se / n_src) ** 0.5,
            "tgt_rmse":       (tgt_se / n_tgt) ** 0.5,
            "eta1":           self.eta1.item(),
            "eta2":           self.eta2.item(),
            "sigma":          last_sigma,
            "mean_potential": total_potential / n_batches,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """Compute MSE, RMSE, and R2 (on the predicted mean) on a labelled DataLoader."""
        means, _, ys = self._predict_raw(loader)
        mse    = ((means - ys) ** 2).mean().item()
        ss_res = ((means - ys) ** 2).sum().item()
        ss_tot = ((ys - ys.mean()) ** 2).sum().item()
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            "domain":    domain,
            "mse":       mse,
            "rmse":      mse ** 0.5,
            "r2":        r2,
            "n_samples": ys.size(0),
        }

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        """Return (true, pred_mean, pred_std) numpy arrays for every sample in *loader*."""
        means, stds, ys = self._predict_raw(loader)
        return ys.cpu().numpy(), means.cpu().numpy(), stds.cpu().numpy()

    def _predict_raw(self, loader: DataLoader):
        self.model.eval()
        ys, means, stds = [], [], []
        for x, y in loader:
            if is_node_graph_batch(x):
                x = x.to(self.device)
                out = node_regress_preds(self.model, x)
                y_ = x.y.float()
            else:
                x, y_ = x.to(self.device), y.to(self.device).float()
                out = self.model.regress(self.model.encode(x))
            mean, var = _split_mean_var(out, self.var_eps)
            ys.append(y_.view(-1))
            means.append(mean.view(-1))
            stds.append(var.sqrt().view(-1))
        return torch.cat(means), torch.cat(stds), torch.cat(ys)
