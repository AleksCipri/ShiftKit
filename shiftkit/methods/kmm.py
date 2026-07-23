"""
KMM — Kernel Mean Matching.

An instance-based domain adaptation method that estimates importance weights
w(x) = p_target(x) / p_source(x) by minimising the MMD between the reweighted
source distribution and the target distribution in a reproducing kernel Hilbert
space (RKHS).  The weights are found by solving a convex quadratic programme.

References
----------
Huang, J., Smola, A. J., Gretton, A., Borgwardt, K. M., & Schölkopf, B. (2007).
Correcting sample selection bias by unlabeled data.
Advances in Neural Information Processing Systems, 19.
https://proceedings.neurips.cc/paper_files/paper/2006/file/a2186aa7c086b46ad4e8bf81e2a3a19b-Paper.pdf

KMMWeightEstimator
------------------
Standalone NumPy estimator.  Solves the QP:

    min_w  (1/2) w^T K_ss w - κ^T w

    s.t.   w_i ≥ 0
           |Σ w_i / n_src - 1| ≤ ε

where K_ss[i,j] = k(x_src_i, x_src_j) and κ_i = (n_src/n_tgt) Σ_j k(x_src_i, x_tgt_j).

Uses scipy.optimize.minimize with the SLSQP solver.

KMMTrainer
----------
Wraps a standard PyTorch model and trains it with importance-weighted
cross-entropy loss:

    loss = Σ_i w(x_i) · CE(f(x_i), y_i) / Σ_i w(x_i)

Weights are estimated once at initialisation and held fixed during training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional, List

from .mmd import _auto_device
from .kliep import _WeightedDataset


# ─── KMM weight estimator ─────────────────────────────────────────────────────

class KMMWeightEstimator:
    """
    Estimates importance weights w(x) ≈ p_target(x) / p_source(x) via KMM.

    Solves a convex QP to minimise the MMD between the reweighted source and
    the target in an RBF RKHS, subject to non-negativity and a normalisation
    constraint.

    Parameters
    ----------
    sigma       : RBF kernel bandwidth σ.  If None, uses the median heuristic.
    B           : upper bound on each weight w_i (default 1000).
    epsilon     : tolerance on the normalisation constraint
                  |Σ w_i / n_src - 1| ≤ ε.  Defaults to (√n_src - 1) / √n_src.
    weight_clip : if set, clips weights to [0, weight_clip] after solving.
    """

    def __init__(
        self,
        sigma: Optional[float] = None,
        B: float = 1000.0,
        epsilon: Optional[float] = None,
        weight_clip: Optional[float] = None,
    ):
        self.sigma = sigma
        self.B = B
        self.epsilon = epsilon
        self.weight_clip = weight_clip

    @staticmethod
    def _rbf_matrix(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
        """K[i, j] = exp(-||X[i] - Y[j]||² / (2σ²))."""
        diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]   # (n, m, d)
        return np.exp(-(diff ** 2).sum(-1) / (2.0 * sigma ** 2))

    @staticmethod
    def _median_bandwidth(X: np.ndarray, Y: np.ndarray, subsample: int = 500) -> float:
        rng = np.random.RandomState(42)
        idx_x = rng.choice(len(X), min(subsample, len(X)), replace=False)
        idx_y = rng.choice(len(Y), min(subsample, len(Y)), replace=False)
        Z = np.vstack([X[idx_x], Y[idx_y]])
        dists = np.concatenate([((Z[i] - Z) ** 2).sum(1) for i in range(len(Z))])
        return float(np.sqrt(max(np.median(dists[dists > 0]), 1e-8)))

    def fit(self, X_src: np.ndarray, X_tgt: np.ndarray) -> "KMMWeightEstimator":
        """
        Estimate importance weights from source and target feature arrays.

        Parameters
        ----------
        X_src : (n_src, d) float32/64 array
        X_tgt : (n_tgt, d) float32/64 array

        Returns
        -------
        self
        """
        try:
            from scipy.optimize import minimize
        except ImportError as e:
            raise ImportError(
                "KMMWeightEstimator requires scipy. "
                "Install it with: pip install scipy"
            ) from e

        X_src = X_src.astype(np.float64)
        X_tgt = X_tgt.astype(np.float64)
        n_src = len(X_src)
        n_tgt = len(X_tgt)

        sigma = self.sigma if self.sigma is not None else self._median_bandwidth(X_src, X_tgt)
        self.sigma_ = sigma

        # Build kernel matrices
        K_ss = self._rbf_matrix(X_src, X_src, sigma)           # (n_src, n_src)
        K_st = self._rbf_matrix(X_src, X_tgt, sigma)           # (n_src, n_tgt)
        kappa = (n_src / n_tgt) * K_st.sum(axis=1)             # (n_src,)

        epsilon = self.epsilon if self.epsilon is not None else (np.sqrt(n_src) - 1.0) / np.sqrt(n_src)

        # QP: min (1/2) w^T K_ss w - κ^T w
        # s.t. 0 ≤ w_i ≤ B,  |mean(w) - 1| ≤ ε
        def objective(w):
            return 0.5 * w @ K_ss @ w - kappa @ w

        def gradient(w):
            return K_ss @ w - kappa

        w0 = np.ones(n_src, dtype=np.float64)
        bounds = [(0.0, self.B)] * n_src
        constraints = [
            {"type": "ineq", "fun": lambda w:  w.mean() - 1.0 + epsilon},
            {"type": "ineq", "fun": lambda w: -w.mean() + 1.0 + epsilon},
        ]

        result = minimize(
            objective,
            w0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-6},
        )

        weights = np.maximum(result.x, 0.0).astype(np.float32)
        if self.weight_clip is not None:
            weights = np.minimum(weights, self.weight_clip)
        self.weights_ = weights
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the estimated importance weights.

        KMM computes weights for the source training set directly — this method
        returns ``self.weights_`` and ignores ``X`` (included for API consistency
        with KLIEPWeightEstimator).

        Returns
        -------
        weights : (n_src,) float32 array
        """
        return self.weights_


# ─── KMM Trainer ─────────────────────────────────────────────────────────────

class KMMTrainer:
    """
    Instance-based domain adaptation via Kernel Mean Matching.

    Estimates importance weights w(x) = p_target(x) / p_source(x) once at
    initialisation by solving a convex QP, then trains the model with an
    importance-weighted cross-entropy loss.

    Because correction happens at the sample level, KMM does not require the
    model to expose ``encode()`` / ``classify()`` — any standard ``nn.Module``
    with ``forward()`` works.

    Parameters
    ----------
    model          : PyTorch model with standard forward() interface
    source_loader  : labelled source DataLoader
    target_loader  : target DataLoader (labels used only for tgt_acc tracking)
    sigma          : RBF bandwidth for KMM (None → median heuristic)
    B              : upper bound on each importance weight (default 1000)
    epsilon        : normalisation constraint tolerance (None → (√n-1)/√n)
    weight_clip    : clip weights to [0, weight_clip] after solving
    lr             : model optimiser learning rate
    device         : 'cuda', 'mps', or 'cpu' (auto-detected if None)

    History keys
    ------------
    epoch, ce_loss (importance-weighted), mmd_loss (0.0), total_loss,
    src_acc, tgt_acc, mean_weight, max_weight
    """

    def __init__(
        self,
        model: nn.Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        sigma: Optional[float] = None,
        B: float = 1000.0,
        epsilon: Optional[float] = None,
        weight_clip: Optional[float] = None,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device) if device else _auto_device()
        self.model = model.to(self.device)
        self.target_loader = target_loader
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.history: List[dict] = []

        self._estimator = KMMWeightEstimator(
            sigma=sigma, B=B, epsilon=epsilon, weight_clip=weight_clip
        )
        self._weighted_source_loader = self._build_weighted_loader(
            source_loader, target_loader
        )

    def _collect_X(self, loader: DataLoader) -> np.ndarray:
        batches = []
        for batch in loader:
            x = batch[0]
            batches.append(x.reshape(x.size(0), -1).numpy())
        return np.concatenate(batches, axis=0)

    def _build_weighted_loader(
        self, source_loader: DataLoader, target_loader: DataLoader
    ) -> DataLoader:
        print("Estimating KMM importance weights…")
        X_src = self._collect_X(source_loader)
        X_tgt = self._collect_X(target_loader)

        self._estimator.fit(X_src, X_tgt)
        weights = self._estimator.predict(X_src)

        print(
            f"  weights — mean={weights.mean():.3f}  "
            f"std={weights.std():.3f}  "
            f"max={weights.max():.3f}"
        )

        weighted_ds = _WeightedDataset(source_loader.dataset, weights)
        return DataLoader(
            weighted_ds,
            batch_size=source_loader.batch_size,
            shuffle=True,
            num_workers=source_loader.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

    def fit(self, epochs: int = 10) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        for epoch in range(1, epochs + 1):
            stats = self._train_epoch(epoch, epochs)
            self.history.append(stats)
            print(
                f"[{epoch:>3}/{epochs}]  "
                f"CE={stats['ce_loss']:.4f}  "
                f"Total={stats['total_loss']:.4f}  "
                f"Src={stats['src_acc']*100:.1f}%  "
                f"Tgt={stats['tgt_acc']*100:.1f}%  "
                f"w̄={stats['mean_weight']:.3f}"
            )
        return self.history

    def _train_epoch(self, epoch: int, total_epochs: int) -> dict:
        self.model.train()
        total_ce = 0.0
        src_correct = tgt_correct = n_src = n_tgt = 0
        weight_sum = weight_max = 0.0

        n_batches = min(len(self._weighted_source_loader), len(self.target_loader))

        for (x_src, y_src, w_src), (x_tgt, y_tgt) in tqdm(
            zip(self._weighted_source_loader, self.target_loader),
            total=n_batches, desc=f"Epoch {epoch}/{total_epochs}", leave=False
        ):
            x_src = x_src.to(self.device)
            y_src = y_src.to(self.device)
            w_src = w_src.to(self.device)
            x_tgt = x_tgt.to(self.device)
            y_tgt = y_tgt.to(self.device)

            logits = self.model(x_src)

            ce_per_sample = F.cross_entropy(logits, y_src, reduction="none")
            w_norm = w_src / (w_src.sum() + 1e-8)
            loss = (ce_per_sample * w_norm).sum() * w_src.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_ce    += loss.item()
            src_correct += (logits.argmax(1) == y_src).sum().item()
            n_src       += y_src.size(0)
            weight_sum  += w_src.mean().item()
            weight_max   = max(weight_max, w_src.max().item())

            with torch.no_grad():
                tgt_logits   = self.model(x_tgt)
                tgt_correct += (tgt_logits.argmax(1) == y_tgt).sum().item()
                n_tgt       += y_tgt.size(0)

        return {
            "epoch":       epoch,
            "ce_loss":     total_ce    / n_batches,
            "mmd_loss":    0.0,
            "total_loss":  total_ce    / n_batches,
            "src_acc":     src_correct / n_src,
            "tgt_acc":     tgt_correct / n_tgt,
            "mean_weight": weight_sum  / n_batches,
            "max_weight":  weight_max,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """Compute accuracy on a labelled DataLoader."""
        self.model.eval()
        correct = total = 0
        for batch in loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            correct += (self.model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
        return {"domain": domain, "accuracy": correct / total, "n_samples": total}
