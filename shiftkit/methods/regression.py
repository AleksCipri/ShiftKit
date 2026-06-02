"""
Regression variants of domain adaptation trainers.

SourceOnlyRegressionTrainer
    MSE baseline — trains on source labels only, no alignment.

MMDRegressionTrainer
    MSE + λ·MMD²  — minimises regression loss on source while aligning latent
    distributions across domains via Maximum Mean Discrepancy.

Both trainers expect a model with ``encode(x)`` and ``regress(z)`` methods
(e.g. ``MLPRegressor``).  Target labels are used only for tracking ``tgt_rmse``
per epoch — they are never used in the loss.

History keys
------------
epoch, mse_loss, mmd_loss, total_loss, src_rmse, tgt_rmse
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List

from .mmd import MMDLoss, _auto_device
from .node_batch import (
    is_node_graph_batch,
    move_node_graph_batch,
    unpack_batch,
    node_latent_vectors,
    node_regress_preds,
)


# ─── Source-Only Regression Baseline ─────────────────────────────────────────

class SourceOnlyRegressionTrainer:
    """
    Regression baseline: MSE on source data only, no domain adaptation.

    Parameters
    ----------
    model         : network with encode(x) and regress(z) (e.g. MLPRegressor)
    source_loader : labelled source DataLoader  (x, y) with float targets
    target_loader : target DataLoader — labels used only for tgt_rmse tracking
    lr            : learning rate
    device        : 'cuda', 'mps', or 'cpu' (auto-detected if None)

    History keys
    ------------
    epoch, mse_loss, mmd_loss (always 0.0), total_loss, src_rmse, tgt_rmse
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
        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.history: List[dict] = []

    def fit(self, epochs: int = 10) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        for epoch in range(1, epochs + 1):
            stats = self._train_epoch(epoch, epochs)
            self.history.append(stats)
            print(
                f"[{epoch:>3}/{epochs}]  "
                f"MSE={stats['mse_loss']:.4f}  "
                f"SrcRMSE={stats['src_rmse']:.4f}  "
                f"TgtRMSE={stats['tgt_rmse']:.4f}"
            )
        return self.history

    def _train_epoch(self, epoch: int, total_epochs: int) -> dict:
        self.model.train()
        total_mse = src_se = tgt_se = 0.0
        n_src = n_tgt = 0
        n_batches = min(len(self.source_loader), len(self.target_loader))

        for (x_src, y_src), (x_tgt, y_tgt) in tqdm(
            zip(self.source_loader, self.target_loader),
            total=n_batches, desc=f"Epoch {epoch}/{total_epochs}", leave=False
        ):
            x_src, y_src, node_batch = unpack_batch(x_src, y_src, self.device)
            x_tgt, y_tgt, _ = unpack_batch(x_tgt, y_tgt, self.device)

            if node_batch:
                pred_src = node_regress_preds(self.model, x_src)
            else:
                pred_src = self.model(x_src).view_as(y_src)
            loss = self.mse_loss(pred_src, y_src)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_mse += loss.item()
            src_se    += ((pred_src.detach() - y_src) ** 2).sum().item()
            n_src     += y_src.size(0)

            with torch.no_grad():
                if is_node_graph_batch(x_tgt):
                    pred_tgt = node_regress_preds(self.model, x_tgt)
                else:
                    pred_tgt = self.model(x_tgt).view_as(y_tgt)
                tgt_se  += ((pred_tgt - y_tgt) ** 2).sum().item()
                n_tgt   += y_tgt.size(0)

        return {
            "epoch":      epoch,
            "mse_loss":   total_mse / n_batches,
            "mmd_loss":   0.0,
            "total_loss": total_mse / n_batches,
            "src_rmse":   (src_se / n_src) ** 0.5,
            "tgt_rmse":   (tgt_se / n_tgt) ** 0.5,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """Compute MSE, RMSE, and R² on a labelled DataLoader."""
        self.model.eval()
        ys, preds = [], []
        for x, y in loader:
            if is_node_graph_batch(x):
                x = move_node_graph_batch(x, self.device)
                preds.append(node_regress_preds(self.model, x))
                ys.append(x.y)
            else:
                x, y = x.to(self.device), y.to(self.device)
                preds.append(self.model(x).view_as(y))
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


# ─── MMD Regression Trainer ───────────────────────────────────────────────────

class MMDRegressionTrainer:
    """
    Regression trainer using Maximum Mean Discrepancy alignment.

    total_loss = MSE(source) + mmd_weight * MMD²(z_src, z_tgt)

    Parameters
    ----------
    model          : network with encode(x) and regress(z) (e.g. MLPRegressor)
    source_loader  : labelled source DataLoader  (x, y) with float targets
    target_loader  : target DataLoader — labels used only for tgt_rmse tracking
    mmd_weight     : λ weighting the MMD alignment term
    lr             : learning rate
    warmup_epochs  : epochs of source-only MSE pre-training before MMD begins
    device         : 'cuda', 'mps', or 'cpu' (auto-detected if None)
    mmd_sigmas     : RBF kernel bandwidths (passed to MMDLoss)

    History keys
    ------------
    epoch, mse_loss, mmd_loss, total_loss, src_rmse, tgt_rmse
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
        self.device = torch.device(device) if device else _auto_device()
        self.model = model.to(self.device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.mmd_weight = mmd_weight
        self.warmup_epochs = warmup_epochs
        self.mse_loss = nn.MSELoss()
        self.mmd_loss = MMDLoss(sigmas=mmd_sigmas)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.history: List[dict] = []

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
                f"SrcRMSE={stats['src_rmse']:.4f}  "
                f"TgtRMSE={stats['tgt_rmse']:.4f}"
            )
        return self.history

    def _train_epoch(self, epoch: int, total_epochs: int, warmup: bool = False) -> dict:
        self.model.train()
        total_mse = total_mmd = total_loss_sum = 0.0
        src_se = tgt_se = 0.0
        n_src = n_tgt = 0
        n_batches = min(len(self.source_loader), len(self.target_loader))

        for (x_src, y_src), (x_tgt, y_tgt) in tqdm(
            zip(self.source_loader, self.target_loader),
            total=n_batches, desc=f"Epoch {epoch}/{total_epochs}", leave=False
        ):
            x_src, y_src, node_batch = unpack_batch(x_src, y_src, self.device)
            x_tgt, y_tgt, _ = unpack_batch(x_tgt, y_tgt, self.device)

            if node_batch:
                z_src = node_latent_vectors(self.model, x_src)
                pred_src = self.model.regress(z_src).view_as(y_src)
            else:
                z_src = self.model.encode(x_src)
                pred_src = self.model.regress(z_src).view_as(y_src)
            mse = self.mse_loss(pred_src, y_src)

            if warmup:
                loss    = mse
                mmd_val = 0.0
            else:
                if node_batch:
                    z_tgt = node_latent_vectors(self.model, x_tgt)
                else:
                    z_tgt = self.model.encode(x_tgt)
                mmd     = self.mmd_loss(z_src, z_tgt)
                loss    = mse + self.mmd_weight * mmd
                mmd_val = mmd.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_mse      += mse.item()
            total_mmd      += mmd_val
            total_loss_sum += loss.item()
            src_se += ((pred_src.detach() - y_src) ** 2).sum().item()
            n_src  += y_src.size(0)

            with torch.no_grad():
                if is_node_graph_batch(x_tgt):
                    pred_tgt = node_regress_preds(self.model, x_tgt)
                else:
                    pred_tgt = self.model.regress(self.model.encode(x_tgt)).view_as(y_tgt)
                tgt_se  += ((pred_tgt - y_tgt) ** 2).sum().item()
                n_tgt   += y_tgt.size(0)

        return {
            "epoch":      epoch,
            "mse_loss":   total_mse      / n_batches,
            "mmd_loss":   total_mmd      / n_batches,
            "total_loss": total_loss_sum / n_batches,
            "src_rmse":   (src_se / n_src) ** 0.5,
            "tgt_rmse":   (tgt_se / n_tgt) ** 0.5,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """Compute MSE, RMSE, and R² on a labelled DataLoader."""
        self.model.eval()
        ys, preds = [], []
        for x, y in loader:
            if is_node_graph_batch(x):
                x = move_node_graph_batch(x, self.device)
                preds.append(node_regress_preds(self.model, x))
                ys.append(x.y)
            else:
                x, y = x.to(self.device), y.to(self.device)
                preds.append(self.model(x).view_as(y))
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
