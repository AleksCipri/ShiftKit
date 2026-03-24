"""
KLIEP — Kullback–Leibler Importance Estimation Procedure.

An instance-based domain adaptation method that estimates the density ratio
w(x) = p_target(x) / p_source(x) and uses it to reweight source samples
during training, so that the weighted empirical source distribution matches
the target distribution.

References
----------
Sugiyama, M., Nakajima, S., Kashima, H., Bünau, P. V., & Kawanabe, M. (2008).
Direct importance estimation with model selection and its application to
covariate shift adaptation.
*Advances in Neural Information Processing Systems*, 20.

KLIEPWeightEstimator
--------------------
Standalone NumPy estimator.  Runs gradient ascent to maximise:

    L(θ) = (1 / n_tgt) Σ_{x ∈ tgt} log w(x; θ)

subject to the normalisation constraint:

    (1 / n_src) Σ_{x ∈ src} w(x; θ) = 1,   θ_l ≥ 0

The importance model is a non-negative kernel density ratio:

    w(x; θ) = Σ_{l=1}^{m} θ_l · K_σ(x, c_l)

where c_l are RBF centres sampled from the target and K_σ is the RBF kernel.

KLIEPTrainer
------------
Wraps a standard PyTorch model and trains it with importance-weighted
cross-entropy loss:

    loss = Σ_i w(x_i) · CE(f(x_i), y_i) / Σ_i w(x_i)

Importance weights are estimated once at initialisation on raw input features,
then held fixed during model training.  This makes KLIEP particularly suited
to tabular and low-dimensional data where input-space density estimation is
tractable.
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


# ─── KLIEP weight estimator ───────────────────────────────────────────────────

class KLIEPWeightEstimator:
    """
    Estimates importance weights w(x) ≈ p_target(x) / p_source(x).

    Uses an RBF kernel basis with centres drawn from the target domain.
    Gradient ascent with non-negativity projection and normalisation after
    each step ensures the constraint E_{src}[w(x)] = 1 is satisfied.

    Parameters
    ----------
    sigma      : RBF kernel bandwidth σ.  If None, uses the median heuristic
                 computed from the combined source + target features.
    n_centers  : number of RBF basis functions (centres sampled from target).
    lr         : gradient-ascent step size for θ.
    n_iter     : number of gradient-ascent iterations.
    weight_clip: if set, clips estimated weights to [0, weight_clip] to prevent
                 extreme values from destabilising training.
    seed       : random seed for centre selection.
    """

    def __init__(
        self,
        sigma: Optional[float] = None,
        n_centers: int = 100,
        lr: float = 0.01,
        n_iter: int = 500,
        weight_clip: Optional[float] = None,
        seed: int = 0,
    ):
        self.sigma = sigma
        self.n_centers = n_centers
        self.lr = lr
        self.n_iter = n_iter
        self.weight_clip = weight_clip
        self.seed = seed

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _rbf(X: np.ndarray, C: np.ndarray, sigma: float) -> np.ndarray:
        """RBF kernel: K[i, l] = exp(-||X[i] - C[l]||² / (2σ²))."""
        diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]   # (n, m, d)
        return np.exp(-(diff ** 2).sum(-1) / (2.0 * sigma ** 2))

    @staticmethod
    def _median_bandwidth(X: np.ndarray, Y: np.ndarray, subsample: int = 500) -> float:
        """Median heuristic: σ = median pairwise distance over a subsample."""
        rng = np.random.RandomState(42)
        idx_x = rng.choice(len(X), min(subsample, len(X)), replace=False)
        idx_y = rng.choice(len(Y), min(subsample, len(Y)), replace=False)
        Z = np.vstack([X[idx_x], Y[idx_y]])
        dists = []
        for i in range(len(Z)):
            diff = Z[i] - Z
            dists.append((diff ** 2).sum(1))
        dists = np.concatenate(dists)
        median_sq = np.median(dists[dists > 0])
        return float(np.sqrt(max(median_sq, 1e-8)))

    # ── public API ─────────────────────────────────────────────────────────

    def fit(self, X_src: np.ndarray, X_tgt: np.ndarray) -> "KLIEPWeightEstimator":
        """
        Estimate importance weights from source and target feature arrays.

        Parameters
        ----------
        X_src : (n_src, d) float32 array of source features
        X_tgt : (n_tgt, d) float32 array of target features

        Returns
        -------
        self
        """
        X_src = X_src.astype(np.float64)
        X_tgt = X_tgt.astype(np.float64)

        sigma = self.sigma if self.sigma is not None else self._median_bandwidth(X_src, X_tgt)
        self.sigma_ = sigma

        # Select RBF centres from target
        rng = np.random.RandomState(self.seed)
        n_centers = min(self.n_centers, len(X_tgt))
        idx = rng.choice(len(X_tgt), n_centers, replace=False)
        self.centers_ = X_tgt[idx]                          # (m, d)

        # Kernel matrices
        Phi_src = self._rbf(X_src, self.centers_, sigma)    # (n_src, m)
        Phi_tgt = self._rbf(X_tgt, self.centers_, sigma)    # (n_tgt, m)

        # Initialise θ and normalise
        theta = np.ones(n_centers, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            Z = (Phi_src @ theta).mean()
        theta = theta / max(float(Z) if np.isfinite(Z) else 1.0, 1e-10)

        # Gradient ascent
        for _ in range(self.n_iter):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                w_tgt = Phi_tgt @ theta                     # (n_tgt,)
            w_tgt = np.where(np.isfinite(w_tgt), w_tgt, 1e-10)
            w_tgt = np.maximum(w_tgt, 1e-10)

            # ∂L/∂θ_l = (1/n_tgt) Σ_j Φ(x_tgt_j, c_l) / w(x_tgt_j)
            grad = (Phi_tgt / w_tgt[:, np.newaxis]).mean(0)
            grad = np.where(np.isfinite(grad), grad, 0.0)

            theta += self.lr * grad
            theta = np.maximum(theta, 0.0)                  # project θ ≥ 0
            theta = np.minimum(theta, 1e8)                  # prevent overflow

            # Re-normalise: E_{src}[w(x)] = 1
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                Z = (Phi_src @ theta).mean()
            if not np.isfinite(Z) or Z < 1e-10:
                break
            theta /= Z

        self.theta_ = theta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return importance weights for samples X.

        Parameters
        ----------
        X : (n, d) array of features

        Returns
        -------
        weights : (n,) float32 array, values ≥ 0
        """
        X = X.astype(np.float64)
        Phi = self._rbf(X, self.centers_, self.sigma_)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            raw = Phi @ self.theta_
        w = np.where(np.isfinite(raw), raw, 0.0)
        w = np.maximum(w, 0.0).astype(np.float32)
        if self.weight_clip is not None:
            w = np.minimum(w, self.weight_clip)
        return w


# ─── Weighted dataset wrapper ─────────────────────────────────────────────────

class _WeightedDataset(Dataset):
    """Wraps a dataset to append a scalar importance weight per sample."""

    def __init__(self, base: Dataset, weights: np.ndarray):
        self.base = base
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, self.weights[idx]


# ─── KLIEP Trainer ────────────────────────────────────────────────────────────

class KLIEPTrainer:
    """
    Instance-based domain adaptation via KLIEP importance weighting.

    Estimates the density ratio w(x) = p_target(x) / p_source(x) once at
    initialisation, then trains the model with an importance-weighted loss:

        loss = Σ_i w(xᵢ) · CE(f(xᵢ), yᵢ) / Σ_i w(xᵢ)

    Because the correction happens at the sample level rather than the feature
    level, KLIEP does not require the model to expose ``encode()`` /
    ``classify()`` — any standard ``nn.Module`` with ``forward()`` works.

    Parameters
    ----------
    model          : PyTorch model with standard forward() interface
    source_loader  : labelled source DataLoader
    target_loader  : target DataLoader (labels used only for tgt_acc tracking)
    sigma          : RBF bandwidth for KLIEP (None → median heuristic)
    n_centers      : number of RBF basis centres sampled from the target
    kliep_lr       : gradient-ascent learning rate inside KLIEP
    kliep_iter     : number of KLIEP gradient-ascent steps
    weight_clip    : clip importance weights to [0, weight_clip] (None = no clip)
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
        n_centers: int = 100,
        kliep_lr: float = 0.01,
        kliep_iter: int = 500,
        weight_clip: Optional[float] = None,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device) if device else _auto_device()
        self.model = model.to(self.device)
        self.target_loader = target_loader
        self.lr = lr
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.history: List[dict] = []

        self._estimator = KLIEPWeightEstimator(
            sigma=sigma, n_centers=n_centers,
            lr=kliep_lr, n_iter=kliep_iter,
            weight_clip=weight_clip,
        )
        self._weighted_source_loader = self._build_weighted_loader(
            source_loader, target_loader
        )

    # ── weight estimation ──────────────────────────────────────────────────

    def _collect_X(self, loader: DataLoader) -> np.ndarray:
        """Collect all input tensors from a loader into a flat numpy array."""
        batches = []
        for batch in loader:
            x = batch[0]                                    # (B, ...)
            batches.append(x.reshape(x.size(0), -1).numpy())
        return np.concatenate(batches, axis=0)

    def _build_weighted_loader(
        self, source_loader: DataLoader, target_loader: DataLoader
    ) -> DataLoader:
        """Run KLIEP and return a weighted DataLoader for the source domain."""
        print("Estimating KLIEP importance weights…")
        X_src = self._collect_X(source_loader)
        X_tgt = self._collect_X(target_loader)

        self._estimator.fit(X_src, X_tgt)
        weights = self._estimator.predict(X_src)            # (n_src,)

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

    # ── training ───────────────────────────────────────────────────────────

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

            # Importance-weighted cross-entropy
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
            "ce_loss":     total_ce     / n_batches,
            "mmd_loss":    0.0,
            "total_loss":  total_ce     / n_batches,
            "src_acc":     src_correct  / n_src,
            "tgt_acc":     tgt_correct  / n_tgt,
            "mean_weight": weight_sum   / n_batches,
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
