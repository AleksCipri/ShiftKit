"""
SIDDA: SInkhorn Dynamic Domain Adaptation (Pandya et al., 2025).

Uses the Sinkhorn divergence as the domain alignment loss with two key ideas:

1. **Dynamic regularisation** — the Sinkhorn blur parameter σ is adapted each
   batch based on the maximum pairwise distance between source and target latent
   features, so the optimal-transport plan automatically adjusts to the current
   state of the encoder.

2. **Learnable loss weighting** — two scalar parameters η₁ (CE) and η₂ (DA) are
   jointly optimised with the model.  The multi-task loss formulation:

       ℒ = (1/2η₁²)ℒ_CE + (1/2η₂²)ℒ_DA + log(|η₁||η₂|)

   replaces a hand-tuned λ.  For a *non-negative* loss term ℒ the η-optimum is
   η² = ℒ, so the log term keeps η away from 0 and ∞; for a loss that can be
   negative there is no such optimum (η is driven to its lower clamp), so
   SIDDA's inverse-η² weighting is only ever applied to non-negative terms here.
   The η-clamps (η₁ ≥ 1e-3, η₂ ≥ 0.25·η₁) bound the coefficients but do not
   guarantee a useful balance.

3. **Optional latent-space instance reweighting** (``use_potentials`` /
   ``weight_ot``; ShiftKit additions, not part of the published method) —
   geomloss returns the Sinkhorn dual potentials (f, g) of the *current* batch.
   The heuristic ``w_i = softmax(-f_i / τ)`` up-weights source samples with a
   low potential.  A dual potential is not a validated per-sample
   "distance-to-target" score, and the benefit of this weighting has not been
   established experimentally; treat both options as experimental.  Weights
   are computed from each batch's own potentials only (no state is carried
   between batches).

An optional **warmup phase** trains the encoder on source classification only
(no DA loss) for a fixed number of epochs before domain adaptation begins.
This ensures the encoder produces meaningful representations before alignment
is attempted — equivariant networks typically need a shorter warmup than CNNs.

Architecture
------------
encoder([x_src; x_tgt]) ──► [z_src; z_tgt]       (one joint forward pass)
z_src ──► classify/regress ──► supervised loss ──┐
Sinkhorn_σ(z_src, z_tgt)   ──► DA loss  ─────────┴─► η-weighted sum

Reference
---------
Pandya, S., Patel, P., Nord, B. D., Walmsley, M., & Ćiprijanović, A. (2025). 
SIDDA: SInkhorn Dynamic Domain Adaptation for image classification with equivariant neural networks. 
Machine Learning: Science and Technology, 6(3), 035032.
arXiv:2501.14048.
https://arxiv.org/abs/2501.14048

Sinkhorn divergence (optimal transport background):
Feydy, J., Séjourné, T., Vialard, F.-X., Amari, S., Trouvé, A., &
Peyré, G. (2019). Interpolating between Optimal Transport and MMD using
Sinkhorn Divergences. AISTATS 2019. https://arxiv.org/abs/1810.08278
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List

from ..models.networks import VAR_EPS
from .base import BaseTrainer
from .node_batch import (
    batch_accuracy,
    encode_domains,
    evaluate_accuracy,
    paired_batches,
    unpack_batch,
    unpack_target_batch,
)
from .regression import (
    _GaussianMixin,
    _RegressionTrainerBase,
    _check_gaussian_loss,
    _fmt,
    _metrics_or_none,
    _per_sample,
    _weighted_mean,
    gaussian_nll,
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

    Distances are measured on z itself, the features the Sinkhorn solve runs
    on, as in the published SIDDA code.  This is a heuristic scale choice: the
    maximum distance need not decrease monotonically as alignment improves.
    """
    with torch.no_grad():
        max_dist = torch.cdist(z_src.detach().float(), z_tgt.detach().float()).max().item()
    return max(scale * max_dist, floor)


def _init_etas(trainer, parametrization: str, eta_lr: Optional[float]) -> list:
    """
    Create the learnable loss weights η₁, η₂ (both η = 1 at the start).

    "log" (default): ``trainer.log_eta1``/``trainer.log_eta2`` are log η,
    trained in a separate Adam (``trainer.eta_optimizer``, lr ``eta_lr``,
    default 1e-2, no weight decay); returns [].  This is a ShiftKit
    modification; no reproducible comparison with the published
    parametrisation is recorded.
    "raw" (published code): ``trainer.eta1``/``trainer.eta2`` are η and are
    trained in the model's AdamW; returns them, to be added to it.
    """
    if parametrization not in ("raw", "log"):
        raise ValueError(f"eta_parametrization must be 'raw' or 'log', got {parametrization!r}")
    trainer.eta_parametrization = parametrization
    if parametrization == "raw":
        trainer.eta1 = nn.Parameter(torch.ones(1, device=trainer.device))  # CE
        trainer.eta2 = nn.Parameter(torch.ones(1, device=trainer.device))  # DA
        trainer.eta_optimizer = None
        return [trainer.eta1, trainer.eta2]
    trainer.log_eta1 = nn.Parameter(torch.zeros(1, device=trainer.device))
    trainer.log_eta2 = nn.Parameter(torch.zeros(1, device=trainer.device))
    trainer.eta_optimizer = optim.Adam(
        [trainer.log_eta1, trainer.log_eta2], lr=1e-2 if eta_lr is None else eta_lr
    )
    return []


def _eta_values(trainer):
    """Current (η₁, η₂) as tensors, for either parametrisation."""
    if trainer.eta_parametrization == "log":
        return trainer.log_eta1.exp(), trainer.log_eta2.exp()
    return trainer.eta1, trainer.eta2


def _clamp_etas(trainer) -> None:
    """Enforce η₁ ≥ 1e-3,  η₂ ≥ 0.25·η₁.

    η₁ ≥ 1e-3 caps the supervised coefficient 1/(2η₁²) at 5e5; it does not
    stop a *large* η₁ from silencing the supervised term.  η₂ ≥ 0.25·η₁ bounds
    the ratio of the DA to the supervised coefficient by 16; the DA term can
    still dominate when the raw losses differ greatly.
    """
    with torch.no_grad():
        if trainer.eta_parametrization == "log":
            trainer.log_eta1.clamp_(min=math.log(1e-3))
            trainer.log_eta2.clamp_(min=trainer.log_eta1.item() + math.log(0.25))
        else:
            trainer.eta1.clamp_(min=1e-3)
            trainer.eta2.clamp_(min=0.25 * trainer.eta1.item())


def _optimizer_step(trainer, loss: torch.Tensor, warmup: bool) -> None:
    """
    Backward, clip, step, and enforce the η constraints.

    Only the model's gradients are clipped, as in the published code; η are
    not clipped.  "raw" follows the published order (clamp η, then step);
    "log" clamps after its own step.
    """
    if not torch.isfinite(loss).all():
        raise FloatingPointError(
            f"non-finite SIDDA training loss ({loss.item()}); try a lower learning "
            "rate, standardised targets, or a larger sigma_floor")
    trainer.optimizer.zero_grad()
    if trainer.eta_optimizer is not None:
        trainer.eta_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.grad_clip)
    if trainer.eta_optimizer is None:
        if not warmup:
            _clamp_etas(trainer)
        trainer.optimizer.step()
    else:
        trainer.optimizer.step()
        if not warmup:
            trainer.eta_optimizer.step()
            _clamp_etas(trainer)


# ─── shared SIDDA machinery ───────────────────────────────────────────────────

def _potential_weights(pot: torch.Tensor, temperature: float) -> torch.Tensor:
    """Heuristic per-sample weights ``softmax(-pot / τ)`` (sum to 1, no gradient)."""
    return pot.detach().reshape(-1).div(-temperature).softmax(dim=-1)


class _SIDDAMixin:
    """Sinkhorn alignment, η weighting and option handling shared by all SIDDA trainers."""

    def _sidda_setup(
        self, model, source_loader, target_loader, *, lr, weight_decay,
        warmup_epochs, sigma_scale, sigma_floor, grad_clip, use_potentials,
        potential_temperature, weight_ot, ot_ema_momentum, scaling,
        eta_parametrization, eta_lr, device,
    ) -> None:
        try:
            from geomloss import SamplesLoss
            self._SamplesLoss = SamplesLoss
        except ImportError as e:
            raise ImportError(
                f"geomloss is required for {type(self).__name__}. "
                "Install it with:  pip install geomloss"
            ) from e
        if ot_ema_momentum is not None:
            warnings.warn(
                "ot_ema_momentum is deprecated and ignored: the previous EMA carried "
                "OT weights between unrelated samples of different batches. "
                "weight_ot now derives weights from the current batch only.",
                FutureWarning, stacklevel=3,
            )
        if potential_temperature <= 0:
            raise ValueError("potential_temperature must be > 0")

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
        self.ot_ema_momentum = None
        self.scaling = scaling

        # Learnable loss-weighting scalars, initialised to η = 1
        shared_etas = _init_etas(self, eta_parametrization, eta_lr)
        self.optimizer = optim.AdamW(
            list(model.parameters()) + shared_etas,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.history: List[dict] = []

    # ------------------------------------------------------------------
    def _sinkhorn_da(self, z_src: torch.Tensor, z_tgt: torch.Tensor) -> dict:
        """
        Sinkhorn DA loss for one batch, plus optional potential-based weights.

        Returns a dict with ``da_loss`` (tensor), ``sigma`` (float),
        ``w_src`` (per-source-sample weights for the supervised loss, or None)
        and ``mean_potential`` (mean source potential of the uniform solve,
        or None).

        All weights are computed from this batch's own potentials.  With
        ``weight_ot=True`` a first, gradient-free uniform solve gives (f, g);
        the DA loss is then the Sinkhorn divergence between the reweighted
        measures α = Σ softmax(-f/τ)_i δ_{z_i}, β = Σ softmax(-g/τ)_j δ_{z*_j}
        (one extra Sinkhorn solve per batch).
        """
        sigma = _dynamic_blur(z_src, z_tgt, self.sigma_scale, self.sigma_floor)
        need_potentials = self.use_potentials or self.weight_ot
        sinkhorn = self._SamplesLoss(
            "sinkhorn", p=2, blur=sigma, scaling=self.scaling,
            potentials=need_potentials,
        )
        if not need_potentials:
            return {"da_loss": sinkhorn(z_src, z_tgt), "sigma": sigma,
                    "w_src": None, "mean_potential": None}

        if self.weight_ot:
            with torch.no_grad():
                f_pot, g_pot = sinkhorn(z_src.detach(), z_tgt.detach())
            a = _potential_weights(f_pot, self.potential_temperature)
            b = _potential_weights(g_pot, self.potential_temperature)
            f_w, g_w = sinkhorn(a, z_src, b, z_tgt)
            # S(α, β) = <α, f> + <β, g> for the weighted measures
            da_loss = (a * f_w.reshape(-1)).sum() + (b * g_w.reshape(-1)).sum()
        else:
            f_pot, g_pot = sinkhorn(z_src, z_tgt)
            # uniform measures: S = mean(f) + mean(g)
            da_loss = f_pot.mean() + g_pot.mean()

        w_src = _potential_weights(f_pot, self.potential_temperature) if self.use_potentials else None
        return {"da_loss": da_loss, "sigma": sigma, "w_src": w_src,
                "mean_potential": f_pot.detach().mean().item()}

    def _eta_weighted(self, sup_loss: torch.Tensor, da_loss: torch.Tensor) -> torch.Tensor:
        """(1/2η₁²)·sup + (1/2η₂²)·DA + log|η₁η₂|; both terms must be ≥ 0."""
        eta1, eta2 = _eta_values(self)
        return (
            (1.0 / (2.0 * eta1 ** 2)) * sup_loss
            + (1.0 / (2.0 * eta2 ** 2)) * da_loss
            + torch.log(eta1.abs() * eta2.abs())
        )

    def _eta_stats(self) -> dict:
        eta1, eta2 = _eta_values(self)
        return {"eta1": eta1.item(), "eta2": eta2.item()}


_SIDDA_PARAM_DOC = """
    lr                   : AdamW learning rate (model, and η when
                           ``eta_parametrization="raw"``)
    weight_decay         : AdamW weight decay
    warmup_epochs        : epochs of source-only supervised training before DA
                           begins (η not updated during warmup)
    sigma_scale          : scale factor for dynamic blur (default 0.05, as published)
    sigma_floor          : minimum blur value (default 0.01, as published)
    grad_clip            : max-norm clipping of model gradients (default 10.0)
    use_potentials       : experimental: weight the per-sample supervised loss by
                           ``softmax(-f/τ)`` of this batch's source potentials
    potential_temperature: τ for that softmax (default 1.0)
    weight_ot            : experimental: replace the uniform empirical measures by
                           ``softmax(-f/τ)``/``softmax(-g/τ)`` weights from a
                           gradient-free uniform solve on the same batch (one
                           extra Sinkhorn solve per batch)
    ot_ema_momentum      : deprecated and ignored (emits a FutureWarning); the
                           former cross-batch EMA was position-indexed
    scaling              : geomloss ε-scaling (default 0.5; published code 0.9)
    eta_parametrization  : "log" (default, ShiftKit modification: log η in a
                           separate Adam, lr ``eta_lr``) or "raw" (published: η
                           in the model's AdamW)
    eta_lr               : η learning rate for "log" (default 1e-2)
    device               : 'cuda', 'mps', or 'cpu' (auto-detected if None)

    An epoch covers the longer loader (the shorter restarts).  After warmup,
    tensor source and target batches are encoded in one joint forward pass, as
    in the published code; node-level/PyG batches are encoded per domain.
    Target loaders may yield ``x``, ``(x,)`` or ``(x, y)``; labels are used only
    for evaluation-mode monitoring after each step.
"""


# ─── SIDDA Trainer ────────────────────────────────────────────────────────────

class SIDDATrainer(_SIDDAMixin, BaseTrainer):
    __doc__ = """
    SIDDA trainer for classification (Pandya et al., 2025).

    Total loss (after warmup)
    -------------------------
    ℒ = (1/2η₁²)·CE(z_src, y_src) + (1/2η₂²)·S_σ(z_src, z_tgt) + log(|η₁|·|η₂|)

    Parameters
    ----------
    model                : network with .encode() and .classify()
    source_loader        : labelled source DataLoader
    target_loader        : target DataLoader (labels only for ``tgt_acc``)
    """ + _SIDDA_PARAM_DOC + """
    History keys
    ------------
    epoch, ce_loss, da_loss, mmd_loss (always 0), total_loss, src_acc,
    tgt_acc (nan if unlabelled), eta1, eta2, sigma, mean_potential
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
        ot_ema_momentum: Optional[float] = None,
        scaling: float = 0.5,
        eta_parametrization: str = "log",
        eta_lr: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self._sidda_setup(
            model, source_loader, target_loader, lr=lr, weight_decay=weight_decay,
            warmup_epochs=warmup_epochs, sigma_scale=sigma_scale,
            sigma_floor=sigma_floor, grad_clip=grad_clip,
            use_potentials=use_potentials, potential_temperature=potential_temperature,
            weight_ot=weight_ot, ot_ema_momentum=ot_ema_momentum, scaling=scaling,
            eta_parametrization=eta_parametrization, eta_lr=eta_lr, device=device,
        )
        self.ce_loss = nn.CrossEntropyLoss()

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

        loader, n_batches = paired_batches(self.source_loader, self.target_loader)

        for (x_src, y_src), tgt_batch in tqdm(
            loader, total=n_batches,
            desc=f"Epoch {epoch}/{total_epochs}", leave=False
        ):
            x_src, y_src, _ = unpack_batch(x_src, y_src, self.device)
            x_tgt, y_tgt = unpack_target_batch(tgt_batch, self.device)

            # As in the published code: source and target in one forward pass
            # (BatchNorm sees both domains together); source only in warmup.
            z_src, z_tgt = encode_domains(self.model, x_src, None if warmup else x_tgt)
            logits = self.model.classify(z_src)

            if warmup:
                ce = self.ce_loss(logits, y_src)
                loss = ce
                da_val = 0.0
            else:
                da = self._sinkhorn_da(z_src, z_tgt)
                last_sigma = da["sigma"]
                if da["w_src"] is not None:
                    ce = _weighted_mean(F.cross_entropy(logits, y_src, reduction="none"), da["w_src"])
                else:
                    ce = self.ce_loss(logits, y_src)
                if da["mean_potential"] is not None:
                    total_potential += da["mean_potential"]
                da_val = da["da_loss"].item()
                loss = self._eta_weighted(ce, da["da_loss"])

            _optimizer_step(self, loss, warmup)

            total_ce       += ce.item()
            total_da       += da_val
            total_loss_sum += loss.item()
            src_correct    += (logits.argmax(1) == y_src).sum().item()
            n_src          += y_src.size(0)

            # Eval mode: monitoring updates no BatchNorm statistics and
            # draws no dropout randomness.
            if y_tgt is not None:
                tgt_correct += batch_accuracy(self.model, x_tgt, y_tgt)
                n_tgt       += y_tgt.size(0)

        return {
            "epoch":           epoch,
            "ce_loss":         total_ce       / n_batches,
            "da_loss":         total_da       / n_batches,
            "mmd_loss":        0.0,            # history-format compatibility
            "total_loss":      total_loss_sum / n_batches,
            "src_acc":         src_correct / n_src,
            "tgt_acc":         tgt_correct / n_tgt if n_tgt else float("nan"),
            **self._eta_stats(),
            "sigma":           last_sigma,
            "mean_potential":  total_potential / n_batches,
        }

    # ------------------------------------------------------------------
    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """Compute accuracy on a labelled DataLoader (evaluation mode)."""
        return evaluate_accuracy(self.model, loader, self.device, domain)


# ─── SIDDA Regression Trainers ────────────────────────────────────────────────

class SIDDARegressionTrainer(_SIDDAMixin, _RegressionTrainerBase):
    __doc__ = """
    SIDDA trainer for regression.

    As SIDDATrainer, with MSE instead of cross-entropy; the model needs
    ``.encode()`` and ``.regress()`` (e.g. ``MLPRegressor``).

    Total loss (after warmup)
    -------------------------
    ℒ = (1/2η₁²)·MSE(regress(z_src), y_src) + (1/2η₂²)·S_σ(z_src, z_tgt)
      + log(|η₁|·|η₂|)

    Parameters
    ----------
    model                : network with .encode() and .regress()
    source_loader        : labelled source DataLoader with float targets
    target_loader        : target DataLoader (labels only for monitoring)
    """ + _SIDDA_PARAM_DOC + """
    History keys
    ------------
    epoch, mse_loss, da_loss, total_loss, src_rmse, src_r2, tgt_rmse, tgt_r2
    (None if unlabelled), eta1, eta2, sigma, mean_potential.  RMSE/R² follow
    :func:`shiftkit.methods.metrics.regression_metrics`, as in ``evaluate``.
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
        ot_ema_momentum: Optional[float] = None,
        scaling: float = 0.5,
        eta_parametrization: str = "log",
        eta_lr: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self._sidda_setup(
            model, source_loader, target_loader, lr=lr, weight_decay=weight_decay,
            warmup_epochs=warmup_epochs, sigma_scale=sigma_scale,
            sigma_floor=sigma_floor, grad_clip=grad_clip,
            use_potentials=use_potentials, potential_temperature=potential_temperature,
            weight_ot=weight_ot, ot_ema_momentum=ot_ema_momentum, scaling=scaling,
            eta_parametrization=eta_parametrization, eta_lr=eta_lr, device=device,
        )
        self.var_eps = VAR_EPS

    # ------------------------------------------------------------------
    def fit(self, epochs: int = 50) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        for epoch in range(1, epochs + 1):
            is_warmup = epoch <= self.warmup_epochs
            stats = self._train_epoch(epoch, epochs, is_warmup)
            self.history.append(stats)
            self._print_epoch(stats, epoch, epochs, is_warmup)
        return self.history

    def _print_epoch(self, stats: dict, epoch: int, epochs: int, is_warmup: bool) -> None:
        phase = "warmup" if is_warmup else "SIDDA "
        da_str = (
            f"  DA={stats['da_loss']:.4f}"
            f"  η₁={stats['eta1']:.3f}  η₂={stats['eta2']:.3f}"
            f"  σ={stats['sigma']:.4f}"
            if not is_warmup else ""
        )
        extra = f"  VarLoss={stats['var_loss']:.4f}" if "var_loss" in stats else ""
        print(
            f"[{epoch:>3}/{epochs}] [{phase}]  "
            f"MSE={stats['mse_loss']:.4f}{extra}{da_str}  "
            f"Total={stats['total_loss']:.4f}  "
            f"SrcRMSE={_fmt(stats['src_rmse'])}  "
            f"TgtRMSE={_fmt(stats['tgt_rmse'])}"
        )

    def _supervised(self, out, y_src, w_src):
        """Return ``(eta_weighted_term, unweighted_term_or_None, point_pred, y)``."""
        pred, y, _ = self._head(out, y_src)
        mse = _weighted_mean(_per_sample(F.mse_loss(pred, y, reduction="none")), w_src)
        return mse, None, pred, y

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int, total_epochs: int, warmup: bool) -> dict:
        self.model.train()

        total_mse = total_var = total_da = total_loss_sum = 0.0
        src_p, src_y, tgt_p, tgt_y = [], [], [], []
        last_sigma = 0.0
        total_potential = 0.0

        loader, n_batches = paired_batches(self.source_loader, self.target_loader)

        for (x_src, y_src), tgt_batch in tqdm(
            loader, total=n_batches,
            desc=f"Epoch {epoch}/{total_epochs}", leave=False
        ):
            x_src, y_src, _ = unpack_batch(x_src, y_src, self.device)
            x_tgt, y_tgt = unpack_target_batch(tgt_batch, self.device)

            # One joint forward pass after warmup, as in the published code.
            z_src, z_tgt = encode_domains(self.model, x_src, None if warmup else x_tgt)
            out_src = self.model.regress(z_src)

            if warmup:
                sup, extra, pred_src, y_src = self._supervised(out_src, y_src, None)
                loss = sup if extra is None else sup + extra
                da_val = 0.0
            else:
                da = self._sinkhorn_da(z_src, z_tgt)
                last_sigma = da["sigma"]
                if da["mean_potential"] is not None:
                    total_potential += da["mean_potential"]
                sup, extra, pred_src, y_src = self._supervised(out_src, y_src, da["w_src"])
                da_val = da["da_loss"].item()
                # η-weighting is applied only to the non-negative terms
                loss = self._eta_weighted(sup, da["da_loss"])
                if extra is not None:
                    loss = loss + extra

            _optimizer_step(self, loss, warmup)

            total_mse      += sup.item()
            total_var      += 0.0 if extra is None else extra.item()
            total_da       += da_val
            total_loss_sum += loss.item()
            src_p.append(pred_src.detach().cpu()); src_y.append(y_src.detach().cpu())
            # target monitoring: eval mode, after the step
            self._monitor_target(x_tgt, y_tgt, tgt_p, tgt_y)

        src, tgt = _metrics_or_none(src_p, src_y), _metrics_or_none(tgt_p, tgt_y)
        stats = {
            "epoch":          epoch,
            "mse_loss":       total_mse       / n_batches,
            "da_loss":        total_da        / n_batches,
            "total_loss":     total_loss_sum  / n_batches,
            "src_rmse":       src["rmse"],
            "src_r2":         src["r2"],
            "tgt_rmse":       tgt["rmse"],
            "tgt_r2":         tgt["r2"],
            **self._eta_stats(),
            "sigma":          last_sigma,
            "mean_potential": total_potential / n_batches,
        }
        if self.gaussian:
            stats["var_loss"] = total_var / n_batches
        return stats


class SIDDAGaussianRegressionTrainer(_GaussianMixin, SIDDARegressionTrainer):
    __doc__ = """
    SIDDA regression trainer with a heteroscedastic (mean, variance) head.

    The model must be built with ``predict_var=True``: column 0 is the mean,
    column 1 a raw value mapped to ``var = softplus(raw) + var_eps``.

    Total loss (after warmup)
    -------------------------
    ℒ = (1/2η₁²)·MSE(mean_src, y_src) + (1/2η₂²)·S_σ(z_src, z_tgt)
      + log(|η₁|·|η₂|) + ℒ_var

    SIDDA's inverse-η² weighting has a finite optimum only for non-negative
    terms (η² = ℒ).  A Gaussian NLL can be negative, which would drive η₁ to
    its clamp and scale the term by up to 5e5.  So only the non-negative MSE of
    the mean is η-weighted; the variance term is added unweighted and uses the
    *detached* mean, so it trains the variance head (and the shared encoder)
    without changing the mean objective:

    * ``loss="nll"`` (default): ℒ_var = Gaussian NLL of y under N(mean.detach(), var)
      (``torch.nn.GaussianNLLLoss`` semantics, ``full=False``).
    * ``loss="moment"``: ℒ_var = J1 = mean(((y − mean.detach())² − var)²), the
      Moment-Network variance objective (Jeffrey & Wandelt 2020, Eq. 5); with
      the η-weighted J0 = MSE this is the non-logarithmic Moment-Network
      objective.  The logarithmic form of Villanueva-Domingo et al. (2022) is
      not used.

    Parameters
    ----------
    model                : network with .encode() and a 2-column .regress()
    source_loader        : labelled source DataLoader with scalar float targets
    target_loader        : target DataLoader (labels only for monitoring)
    var_eps              : variance floor added after the softplus (default 1e-6)
    loss                 : "nll" (default) or "moment"
    """ + _SIDDA_PARAM_DOC + """
    ``evaluate`` reports RMSE/MAE/R² of the mean plus ``nll``, ``cov68``,
    ``cov95`` and ``mean_sd``.  These assume a Gaussian predictive form and
    say nothing about calibration on unseen shifts.

    History keys
    ------------
    epoch, mse_loss, var_loss, da_loss, total_loss, src_rmse, src_r2,
    tgt_rmse, tgt_r2, eta1, eta2, sigma, mean_potential
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
        ot_ema_momentum: Optional[float] = None,
        var_eps: float = VAR_EPS,
        scaling: float = 0.5,
        eta_parametrization: str = "log",
        eta_lr: Optional[float] = None,
        device: Optional[str] = None,
        loss: str = "nll",
    ):
        super().__init__(
            model, source_loader, target_loader, lr=lr, weight_decay=weight_decay,
            warmup_epochs=warmup_epochs, sigma_scale=sigma_scale,
            sigma_floor=sigma_floor, grad_clip=grad_clip,
            use_potentials=use_potentials, potential_temperature=potential_temperature,
            weight_ot=weight_ot, ot_ema_momentum=ot_ema_momentum, scaling=scaling,
            eta_parametrization=eta_parametrization, eta_lr=eta_lr, device=device,
        )
        self.var_eps = var_eps
        self.loss = _check_gaussian_loss(loss)

    def _supervised(self, out, y_src, w_src):
        mean, y, var = self._head(out, y_src)
        mse = _weighted_mean((y - mean) ** 2, w_src)
        if self.loss == "nll":
            var_term = gaussian_nll(mean.detach(), var, y, w_src)
        else:
            var_term = _weighted_mean(((y - mean.detach()) ** 2 - var) ** 2, w_src)
        return mse, var_term, mean, y
