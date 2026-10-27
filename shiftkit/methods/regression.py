"""
Regression variants of domain adaptation trainers.

SourceOnlyRegressionTrainer
    MSE baseline — trains on source labels only, no alignment.

MMDRegressionTrainer
    MSE + λ·MMD²  — minimises regression loss on source while aligning latent
    distributions across domains via Maximum Mean Discrepancy.

SourceOnlyGaussianRegressionTrainer / MMDGaussianRegressionTrainer
    Heteroscedastic (mean, variance) variants for models built with
    ``predict_var=True`` (see "Two-column head contract" below).

All trainers expect a model with ``encode(x)`` and ``regress(z)`` methods
(e.g. ``MLPRegressor``).  Target labels, if the target loader yields them, are
used only for monitoring — never in the loss.  Target loaders may yield ``x``,
``(x,)`` or ``(x, y)``; without labels the target metrics are ``None``.

Metric convention
-----------------
History and ``evaluate`` both use :func:`shiftkit.methods.metrics.regression_metrics`:
MSE/RMSE/MAE average over all output elements, R² is computed per output
column and averaged.  Source metrics in the history are training-mode
predictions on each batch *before* its optimizer step; target metrics are
computed in evaluation mode (no dropout, no BatchNorm update) *after* it.

An epoch covers the longer of the two loaders; the shorter one restarts.
Tensor source and target batches are encoded in one joint forward pass during
alignment epochs, so BatchNorm (if any) sees both domains together.

Two-column head contract
------------------------
Column 0 of a 2-column head is the predicted mean ``mu``; column 1 is an
unconstrained value ``r`` mapped to the predicted variance
``var = softplus(r) + var_eps`` (:func:`shiftkit.models.networks.mean_var_from_head`).

Gaussian training objectives (``loss=``)
---------------------------------------
``"nll"`` (default) — per-sample Gaussian negative log-likelihood with
``torch.nn.GaussianNLLLoss(full=False)`` semantics,
``0.5·(log var + (y − mu)² / var)``, averaged over the batch.  It assumes a
Gaussian conditional distribution; a good NLL does not prove normality,
calibration under shift, or epistemic uncertainty.

``"moment"`` — the Moment-Network objective of Jeffrey & Wandelt (2020,
arXiv:2011.05991, Eqs. 4–5): ``J0 = mean (y − mu)²`` for the mean and
``J1 = mean (((y − mu)² − var)²)`` for the variance, where the paper trains the
variance network with an already-trained, *fixed* mean.  Here both are trained
simultaneously; the mean is detached inside ``J1`` so the variance term never
moves the mean head (a shared encoder still receives gradients from both).
``J0`` has units of y² and ``J1`` units of y⁴, so standardise targets.  The
logarithmic variant ``log ΣJ0 + log ΣJ1`` used by Villanueva-Domingo et al.
(2022, arXiv:2111.08683, Eqs. 6–7) is *not* implemented: it can be negative,
which is unsafe under SIDDA's inverse-η² weighting.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.networks import VAR_EPS, mean_var_from_head
from .base import BaseTrainer
from .metrics import gaussian_metrics, regression_metrics
from .mmd import MMDLoss, _auto_device
from .node_batch import (
    encode_domains,
    eval_mode,
    is_node_graph_batch,
    node_latent_vectors,
    paired_batches,
    unpack_batch,
    unpack_target_batch,
)


# ─── shared helpers ───────────────────────────────────────────────────────────

_GAUSSIAN_LOSSES = ("nll", "moment")


def _check_gaussian_loss(loss: str) -> str:
    if loss not in _GAUSSIAN_LOSSES:
        raise ValueError(f"loss must be one of {_GAUSSIAN_LOSSES}, got {loss!r}")
    return loss


def _weighted_mean(per_sample: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    """Batch mean, or ``Σ w_i·l_i`` when *weights* (summing to 1) are given."""
    if weights is None:
        return per_sample.mean()
    return (weights.reshape(-1) * per_sample.reshape(-1)).sum()


def _per_sample(x: torch.Tensor) -> torch.Tensor:
    """Average a per-element tensor to one value per sample (row)."""
    return x.reshape(x.shape[0], -1).mean(dim=1)


def gaussian_nll(mean: torch.Tensor, var: torch.Tensor, y: torch.Tensor,
                 weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Gaussian NLL, ``torch.nn.GaussianNLLLoss(full=False)`` semantics.

    ``0.5·(log var + (y − mean)² / var)`` per sample, averaged (or weighted by
    *weights*, which must sum to 1).  Can be negative.
    """
    nll = 0.5 * (torch.log(var) + (y - mean) ** 2 / var)
    return _weighted_mean(nll, weights)


def moment_losses(mean: torch.Tensor, var: torch.Tensor, y: torch.Tensor,
                  weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Moment-Network losses ``(J0, J1)`` (Jeffrey & Wandelt 2020, Eqs. 4–5).

    ``J0 = mean (y − mean)²``; ``J1 = mean (((y − mean.detach())² − var)²)``.
    Both are non-negative.  See the module docstring for the deviation from the
    paper's sequential (fixed-mean) training.
    """
    j0 = _weighted_mean((y - mean) ** 2, weights)
    j1 = _weighted_mean(((y - mean.detach()) ** 2 - var) ** 2, weights)
    return j0, j1


def _point_preds(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reshape head output to the target shape, with a clear error on mismatch."""
    if out.numel() != y.numel():
        raise ValueError(
            f"model output shape {tuple(out.shape)} does not match target shape "
            f"{tuple(y.shape)}; 2-column (mean, variance) heads need a Gaussian trainer")
    return out.reshape(y.shape)


def _mean_var(out: torch.Tensor, y: torch.Tensor, eps: float):
    """``(mean, var, y)`` flattened to one value per sample (scalar targets)."""
    if out.dim() == 1 or out.size(-1) != 2:
        raise ValueError(
            f"Gaussian trainers need a 2-column (mean, raw-variance) head; got output "
            f"shape {tuple(out.shape)}. Build the model with predict_var=True.")
    mean, var = mean_var_from_head(out, eps)
    mean, var = mean.reshape(-1), var.reshape(-1)
    y = y.reshape(-1).float()
    if y.numel() != mean.numel():
        raise ValueError(
            f"Gaussian trainers support scalar targets only: got {y.numel()} target values "
            f"for {mean.numel()} predictions")
    return mean, var, y


def _check_finite(loss: torch.Tensor, epoch: int, step: int) -> None:
    if not torch.isfinite(loss).all():
        raise FloatingPointError(
            f"non-finite training loss ({loss.item()}) at epoch {epoch}, batch {step}; "
            "try a lower learning rate, standardised targets, or a larger var_eps")


def _latents(model: nn.Module, x) -> torch.Tensor:
    if is_node_graph_batch(x):
        return node_latent_vectors(model, x)
    return model.encode(x)


def _cat_or_none(chunks: List[torch.Tensor]) -> Optional[torch.Tensor]:
    return torch.cat(chunks) if chunks else None


def _metrics_or_none(preds: List[torch.Tensor], ys: List[torch.Tensor]) -> Dict[str, Optional[float]]:
    if not preds:
        return {"rmse": None, "r2": None}
    m = regression_metrics(torch.cat(preds), torch.cat(ys))
    return {"rmse": m["rmse"], "r2": m["r2"]}


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.4f}"


class _RegressionTrainerBase(BaseTrainer):
    """Shared device/loader setup, monitoring and evaluation for regression trainers."""

    gaussian: bool = False

    def _setup(self, model, source_loader, target_loader, device, var_eps=VAR_EPS):
        self.device = torch.device(device) if device else _auto_device()
        self.model = model.to(self.device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.var_eps = var_eps
        self.history: List[dict] = []

    # -- forward helpers ---------------------------------------------------
    def _head(self, out: torch.Tensor, y: torch.Tensor):
        """Return ``(point_prediction, y, var_or_None)`` for one batch."""
        if self.gaussian:
            mean, var, y = _mean_var(out, y, self.var_eps)
            return mean, y, var
        y = y.float()
        return _point_preds(out, y), y, None

    @torch.no_grad()
    def _monitor_target(self, x_tgt, y_tgt, preds: list, ys: list) -> None:
        """Eval-mode target predictions after the step (labels only for monitoring)."""
        if y_tgt is None:
            return
        with eval_mode(self.model):
            pred, y, _ = self._head(self.model.regress(_latents(self.model, x_tgt)), y_tgt)
        preds.append(pred.detach().cpu())
        ys.append(y.detach().cpu())

    # -- evaluation --------------------------------------------------------
    @torch.no_grad()
    def _predict_raw(self, loader: DataLoader):
        """Eval-mode ``(point_preds, vars_or_None, ys)`` over a labelled loader."""
        preds, variances, ys = [], [], []
        with eval_mode(self.model):
            for batch in loader:
                x, y = unpack_target_batch(batch, self.device)
                if y is None:
                    raise ValueError("evaluate()/predict() need a labelled loader yielding (x, y)")
                pred, y, var = self._head(self.model.regress(_latents(self.model, x)), y)
                preds.append(pred)
                ys.append(y)
                if var is not None:
                    variances.append(var)
        if not preds:
            raise ValueError("cannot evaluate on an empty loader")
        return torch.cat(preds), _cat_or_none(variances), torch.cat(ys)

    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """MSE, RMSE, MAE and per-column-averaged R² on a labelled loader.

        Gaussian trainers additionally report ``nll``, ``cov68``, ``cov95`` and
        ``mean_sd`` (see :func:`shiftkit.methods.metrics.gaussian_metrics`).
        """
        preds, variances, ys = self._predict_raw(loader)
        result = {"domain": domain, **regression_metrics(preds, ys)}
        if variances is not None:
            result.update(gaussian_metrics(preds, variances, ys))
        result["n_samples"] = int(ys.shape[0])
        return result


# ─── Source-Only Regression Baseline ─────────────────────────────────────────

class SourceOnlyRegressionTrainer(_RegressionTrainerBase):
    """
    Regression baseline: MSE on source data only, no domain adaptation.

    Parameters
    ----------
    model         : network with encode(x) and regress(z) (e.g. MLPRegressor)
    source_loader : labelled source DataLoader  (x, y) with float targets
    target_loader : target DataLoader yielding ``x`` or ``(x, y)``; labels are
                    used only for ``tgt_rmse`` / ``tgt_r2`` monitoring
    lr            : learning rate
    device        : 'cuda', 'mps', or 'cpu' (auto-detected if None)

    History keys
    ------------
    epoch, mse_loss, mmd_loss (always 0.0), total_loss, src_rmse, src_r2,
    tgt_rmse, tgt_r2 (``None`` for an unlabelled target loader)
    """

    def __init__(
        self,
        model: nn.Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        self._setup(model, source_loader, target_loader, device)
        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def fit(self, epochs: int = 10) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        for epoch in range(1, epochs + 1):
            stats = self._train_epoch(epoch, epochs)
            self.history.append(stats)
            print(
                f"[{epoch:>3}/{epochs}]  "
                f"MSE={stats['mse_loss']:.4f}  "
                f"SrcRMSE={_fmt(stats['src_rmse'])}  "
                f"TgtRMSE={_fmt(stats['tgt_rmse'])}"
            )
        return self.history

    def _train_epoch(self, epoch: int, total_epochs: int) -> dict:
        self.model.train()
        total_mse = 0.0
        src_p, src_y, tgt_p, tgt_y = [], [], [], []
        loader, n_batches = paired_batches(self.source_loader, self.target_loader)

        for step, ((x_src, y_src), tgt_batch) in enumerate(tqdm(
            loader, total=n_batches, desc=f"Epoch {epoch}/{total_epochs}", leave=False
        )):
            x_src, y_src, _ = unpack_batch(x_src, y_src, self.device)
            x_tgt, y_tgt = unpack_target_batch(tgt_batch, self.device)

            pred_src, y_src, _ = self._head(self.model.regress(_latents(self.model, x_src)), y_src)
            loss = self.mse_loss(pred_src, y_src)
            _check_finite(loss, epoch, step)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_mse += loss.item()
            src_p.append(pred_src.detach().cpu()); src_y.append(y_src.detach().cpu())
            self._monitor_target(x_tgt, y_tgt, tgt_p, tgt_y)

        src, tgt = _metrics_or_none(src_p, src_y), _metrics_or_none(tgt_p, tgt_y)
        return {
            "epoch":      epoch,
            "mse_loss":   total_mse / n_batches,
            "mmd_loss":   0.0,
            "total_loss": total_mse / n_batches,
            "src_rmse":   src["rmse"],
            "src_r2":     src["r2"],
            "tgt_rmse":   tgt["rmse"],
            "tgt_r2":     tgt["r2"],
        }


# ─── MMD Regression Trainer ───────────────────────────────────────────────────

class MMDRegressionTrainer(_RegressionTrainerBase):
    """
    Regression trainer using Maximum Mean Discrepancy alignment.

    total_loss = MSE(source) + mmd_weight * MMD²(z_src, z_tgt)

    Parameters
    ----------
    model          : network with encode(x) and regress(z) (e.g. MLPRegressor)
    source_loader  : labelled source DataLoader  (x, y) with float targets
    target_loader  : target DataLoader yielding ``x`` or ``(x, y)``; labels are
                     used only for ``tgt_rmse`` / ``tgt_r2`` monitoring
    mmd_weight     : λ weighting the MMD alignment term
    lr             : learning rate
    warmup_epochs  : epochs of source-only MSE pre-training before MMD begins
    device         : 'cuda', 'mps', or 'cpu' (auto-detected if None)
    mmd_sigmas     : RBF kernel bandwidths (passed to MMDLoss)

    History keys
    ------------
    epoch, mse_loss, mmd_loss, total_loss, src_rmse, src_r2, tgt_rmse, tgt_r2
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
        mmd_sigmas: Optional[List[float]] = None,
    ):
        self._setup(model, source_loader, target_loader, device)
        self.mmd_weight = mmd_weight
        self.warmup_epochs = warmup_epochs
        self.mse_loss = nn.MSELoss()
        self.mmd_loss = MMDLoss(sigmas=mmd_sigmas)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def fit(self, epochs: int = 10) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        for epoch in range(1, epochs + 1):
            is_warmup = epoch <= self.warmup_epochs
            stats = self._train_epoch(epoch, epochs, is_warmup)
            self.history.append(stats)
            phase   = "warmup" if is_warmup else "MMD  "
            mmd_str = f"  MMD={stats['mmd_loss']:.4f}" if not is_warmup else ""
            print(
                f"[{epoch:>3}/{epochs}] [{phase}]  "
                f"MSE={stats['mse_loss']:.4f}{mmd_str}  "
                f"Total={stats['total_loss']:.4f}  "
                f"SrcRMSE={_fmt(stats['src_rmse'])}  "
                f"TgtRMSE={_fmt(stats['tgt_rmse'])}"
            )
        return self.history

    def _train_epoch(self, epoch: int, total_epochs: int, warmup: bool = False) -> dict:
        self.model.train()
        total_mse = total_mmd = total_loss_sum = 0.0
        src_p, src_y, tgt_p, tgt_y = [], [], [], []
        loader, n_batches = paired_batches(self.source_loader, self.target_loader)

        for step, ((x_src, y_src), tgt_batch) in enumerate(tqdm(
            loader, total=n_batches, desc=f"Epoch {epoch}/{total_epochs}", leave=False
        )):
            x_src, y_src, _ = unpack_batch(x_src, y_src, self.device)
            x_tgt, y_tgt = unpack_target_batch(tgt_batch, self.device)

            z_src, z_tgt = encode_domains(self.model, x_src, None if warmup else x_tgt)
            pred_src, y_src, _ = self._head(self.model.regress(z_src), y_src)
            mse = self.mse_loss(pred_src, y_src)

            if warmup:
                loss    = mse
                mmd_val = 0.0
            else:
                mmd     = self.mmd_loss(z_src, z_tgt)
                loss    = mse + self.mmd_weight * mmd
                mmd_val = mmd.item()
            _check_finite(loss, epoch, step)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_mse      += mse.item()
            total_mmd      += mmd_val
            total_loss_sum += loss.item()
            src_p.append(pred_src.detach().cpu()); src_y.append(y_src.detach().cpu())
            self._monitor_target(x_tgt, y_tgt, tgt_p, tgt_y)

        src, tgt = _metrics_or_none(src_p, src_y), _metrics_or_none(tgt_p, tgt_y)
        return {
            "epoch":      epoch,
            "mse_loss":   total_mse      / n_batches,
            "mmd_loss":   total_mmd      / n_batches,
            "total_loss": total_loss_sum / n_batches,
            "src_rmse":   src["rmse"],
            "src_r2":     src["r2"],
            "tgt_rmse":   tgt["rmse"],
            "tgt_r2":     tgt["r2"],
        }


# ─── Heteroscedastic (mean, variance) regression ─────────────────────────────

class _GaussianMixin:
    """Prediction helpers shared by every Gaussian (2-column head) trainer."""

    gaussian = True

    def _regression_objective(self, mean, var, y, weights=None):
        """Return ``(objective, mse_of_mean)`` for ``self.loss``."""
        if self.loss == "nll":
            obj = gaussian_nll(mean, var, y, weights)
            return obj, _weighted_mean((y - mean) ** 2, weights)
        j0, j1 = moment_losses(mean, var, y, weights)
        return j0 + j1, j0

    @torch.no_grad()
    def predict_mean_var(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return predicted ``(mean, var)`` tensors (eval mode) for a labelled loader."""
        means, variances, _ = self._predict_raw(loader)
        return means, variances

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        """Return ``(y_true, pred_mean, pred_std)`` numpy arrays for *loader*.

        ``pred_std = sqrt(var)`` with ``var = softplus(raw) + var_eps``.  Use
        :meth:`predict_mean_var` for the variance directly.
        """
        means, variances, ys = self._predict_raw(loader)
        return ys.cpu().numpy(), means.cpu().numpy(), variances.sqrt().cpu().numpy()


class SourceOnlyGaussianRegressionTrainer(_GaussianMixin, _RegressionTrainerBase):
    """
    Source-only baseline with a heteroscedastic (mean, variance) head.

    The model must be built with ``predict_var=True`` (2-column head: mean, raw
    value mapped to ``var = softplus(raw) + var_eps``).  Trained with the
    Gaussian NLL (``loss="nll"``, default) or the Moment-Network objective
    ``J0 + J1`` (``loss="moment"``); see the module docstring.

    RMSE/R² in history and ``evaluate`` use the predicted mean, so they are
    comparable with ``SourceOnlyRegressionTrainer``; ``evaluate`` also reports
    ``nll``, ``cov68``, ``cov95`` and ``mean_sd``.

    History keys
    ------------
    epoch, reg_loss, mse_loss, total_loss, src_rmse, src_r2, tgt_rmse, tgt_r2
    """

    def __init__(
        self,
        model: nn.Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        lr: float = 1e-3,
        device: Optional[str] = None,
        var_eps: float = VAR_EPS,
        loss: str = "nll",
    ):
        self._setup(model, source_loader, target_loader, device, var_eps)
        self.loss = _check_gaussian_loss(loss)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def fit(self, epochs: int = 10) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        for epoch in range(1, epochs + 1):
            stats = self._train_epoch(epoch, epochs)
            self.history.append(stats)
            print(
                f"[{epoch:>3}/{epochs}]  "
                f"{self.loss.upper()}={stats['reg_loss']:.4f}  MSE={stats['mse_loss']:.4f}  "
                f"SrcRMSE={_fmt(stats['src_rmse'])}  "
                f"TgtRMSE={_fmt(stats['tgt_rmse'])}"
            )
        return self.history

    def _train_epoch(self, epoch: int, total_epochs: int) -> dict:
        self.model.train()
        total_reg = total_mse = 0.0
        src_p, src_y, tgt_p, tgt_y = [], [], [], []
        loader, n_batches = paired_batches(self.source_loader, self.target_loader)

        for step, ((x_src, y_src), tgt_batch) in enumerate(tqdm(
            loader, total=n_batches, desc=f"Epoch {epoch}/{total_epochs}", leave=False
        )):
            x_src, y_src, _ = unpack_batch(x_src, y_src, self.device)
            x_tgt, y_tgt = unpack_target_batch(tgt_batch, self.device)

            mean_src, y_src, var_src = self._head(
                self.model.regress(_latents(self.model, x_src)), y_src)
            loss, mse = self._regression_objective(mean_src, var_src, y_src)
            _check_finite(loss, epoch, step)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_reg += loss.item()
            total_mse += mse.item()
            src_p.append(mean_src.detach().cpu()); src_y.append(y_src.detach().cpu())
            self._monitor_target(x_tgt, y_tgt, tgt_p, tgt_y)

        src, tgt = _metrics_or_none(src_p, src_y), _metrics_or_none(tgt_p, tgt_y)
        return {
            "epoch":      epoch,
            "reg_loss":   total_reg / n_batches,
            "mse_loss":   total_mse / n_batches,
            "total_loss": total_reg / n_batches,
            "src_rmse":   src["rmse"],
            "src_r2":     src["r2"],
            "tgt_rmse":   tgt["rmse"],
            "tgt_r2":     tgt["r2"],
        }


class MMDGaussianRegressionTrainer(_GaussianMixin, _RegressionTrainerBase):
    """
    MMD-aligned regression trainer with a heteroscedastic (mean, variance) head.

    total_loss = reg_loss(mean, var, y_source) + mmd_weight · MMD²(z_src, z_tgt)

    ``reg_loss`` is the Gaussian NLL (``loss="nll"``, default) or the
    Moment-Network objective ``J0 + J1`` (``loss="moment"``).  The model must be
    built with ``predict_var=True``.  The MMD term operates on encoder latents.

    History keys
    ------------
    epoch, reg_loss, mse_loss, mmd_loss, total_loss, src_rmse, src_r2,
    tgt_rmse, tgt_r2
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
        mmd_sigmas: Optional[List[float]] = None,
        var_eps: float = VAR_EPS,
        loss: str = "nll",
    ):
        self._setup(model, source_loader, target_loader, device, var_eps)
        self.loss = _check_gaussian_loss(loss)
        self.mmd_weight = mmd_weight
        self.warmup_epochs = warmup_epochs
        self.mmd_loss = MMDLoss(sigmas=mmd_sigmas)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def fit(self, epochs: int = 10) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        for epoch in range(1, epochs + 1):
            is_warmup = epoch <= self.warmup_epochs
            stats = self._train_epoch(epoch, epochs, is_warmup)
            self.history.append(stats)
            phase   = "warmup" if is_warmup else "MMD  "
            mmd_str = f"  MMD={stats['mmd_loss']:.4f}" if not is_warmup else ""
            print(
                f"[{epoch:>3}/{epochs}] [{phase}]  "
                f"{self.loss.upper()}={stats['reg_loss']:.4f}{mmd_str}  "
                f"Total={stats['total_loss']:.4f}  "
                f"SrcRMSE={_fmt(stats['src_rmse'])}  "
                f"TgtRMSE={_fmt(stats['tgt_rmse'])}"
            )
        return self.history

    def _train_epoch(self, epoch: int, total_epochs: int, warmup: bool = False) -> dict:
        self.model.train()
        total_reg = total_mse = total_mmd = total_loss_sum = 0.0
        src_p, src_y, tgt_p, tgt_y = [], [], [], []
        loader, n_batches = paired_batches(self.source_loader, self.target_loader)

        for step, ((x_src, y_src), tgt_batch) in enumerate(tqdm(
            loader, total=n_batches, desc=f"Epoch {epoch}/{total_epochs}", leave=False
        )):
            x_src, y_src, _ = unpack_batch(x_src, y_src, self.device)
            x_tgt, y_tgt = unpack_target_batch(tgt_batch, self.device)

            z_src, z_tgt = encode_domains(self.model, x_src, None if warmup else x_tgt)
            mean_src, y_src, var_src = self._head(self.model.regress(z_src), y_src)
            reg_loss, mse = self._regression_objective(mean_src, var_src, y_src)

            if warmup:
                loss    = reg_loss
                mmd_val = 0.0
            else:
                mmd     = self.mmd_loss(z_src, z_tgt)
                loss    = reg_loss + self.mmd_weight * mmd
                mmd_val = mmd.item()
            _check_finite(loss, epoch, step)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_reg      += reg_loss.item()
            total_mse      += mse.item()
            total_mmd      += mmd_val
            total_loss_sum += loss.item()
            src_p.append(mean_src.detach().cpu()); src_y.append(y_src.detach().cpu())
            self._monitor_target(x_tgt, y_tgt, tgt_p, tgt_y)

        src, tgt = _metrics_or_none(src_p, src_y), _metrics_or_none(tgt_p, tgt_y)
        return {
            "epoch":      epoch,
            "reg_loss":   total_reg      / n_batches,
            "mse_loss":   total_mse      / n_batches,
            "mmd_loss":   total_mmd      / n_batches,
            "total_loss": total_loss_sum / n_batches,
            "src_rmse":   src["rmse"],
            "src_r2":     src["r2"],
            "tgt_rmse":   tgt["rmse"],
            "tgt_r2":     tgt["r2"],
        }
