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
*Advances in Neural Information Processing Systems 19* (NIPS 2006 conference;
proceedings published 2007).
https://proceedings.neurips.cc/paper_files/paper/2006/file/a2186aa7c086b46ad4e8bf81e2a3a19b-Paper.pdf

KMMWeightEstimator
------------------
Standalone NumPy estimator.  Solves the QP:

    min_w  (1/2) w^T K_ss w - κ^T w

    s.t.   0 ≤ w_i ≤ B
           |Σ w_i / n_src - 1| ≤ ε

where K_ss[i,j] = k(x_src_i, x_src_j) and κ_i = (n_src/n_tgt) Σ_j k(x_src_i, x_tgt_j).

Uses scipy.optimize.minimize with the SLSQP solver (n_src variables).  The
dense n_src × n_src kernel needs 8·n_src² bytes and SLSQP's dense workspace is
of the same order, so the estimator refuses inputs whose kernel would exceed
``max_kernel_bytes``; subsample the source set for larger data.  The solver
result is checked (success flag, finiteness, bounds and normalisation).

KMMTrainer
----------
Wraps a standard PyTorch model and trains it with importance-weighted
cross-entropy loss (per batch):

    loss = Σ_i w(x_i) · CE(f(x_i), y_i) / Σ_i w(x_i)

Weights are estimated once at initialisation and held fixed during training.
"""

import hashlib
import warnings

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from .kliep import (
    _ImportanceWeightedTrainer,
    _check_features,
    _check_sigma,
    _clip_scale,
    _median_bandwidth,
    _rbf_kernel,
    _rbf_row_sums,
)


DEFAULT_MAX_KERNEL_BYTES = 2.0e8   # ≈ 5 000 source samples (float64 n²)


def _kernel_bytes(n: int) -> int:
    return 8 * int(n) * int(n)


def _check_kernel_size(n_src: int, max_kernel_bytes: Optional[float]) -> None:
    if max_kernel_bytes is None:
        return
    need = _kernel_bytes(n_src)
    if need > max_kernel_bytes:
        max_n = int(np.sqrt(max_kernel_bytes / 8))
        raise MemoryError(
            f"KMM needs a dense {n_src} × {n_src} float64 kernel matrix "
            f"({need / 1e9:.2f} GB), plus an SLSQP workspace of similar order, which "
            f"exceeds max_kernel_bytes={max_kernel_bytes:.3g} (≈ {max_n} samples). "
            "Subsample the source set (e.g. DataLoader(Subset(dataset, indices), ...) with a "
            "few thousand indices), reduce the data, or raise max_kernel_bytes / pass "
            "max_kernel_bytes=None if you have the memory and time."
        )


def _array_fingerprint(X: np.ndarray) -> str:
    X = np.ascontiguousarray(X, dtype=np.float64)
    return hashlib.sha1(repr(X.shape).encode() + X.tobytes()).hexdigest()


# ─── KMM weight estimator ─────────────────────────────────────────────────────

class KMMWeightEstimator:
    """
    Estimates importance weights w(x) ≈ p_target(x) / p_source(x) via KMM.

    Solves a convex QP to minimise the MMD between the reweighted source and
    the target in an RBF RKHS, subject to non-negativity and a normalisation
    constraint.  Weights exist only for the fitted source sample; there is no
    out-of-sample ``predict``.

    The bandwidth is either given or chosen by the median heuristic.

    Parameters
    ----------
    sigma       : RBF kernel bandwidth σ.  If None, uses the median heuristic
                  (raises ValueError when undefined, e.g. constant features).
    B           : upper bound on each weight w_i (default 1000).
    epsilon     : tolerance on the normalisation constraint
                  |Σ w_i / n_src - 1| ≤ ε.  Defaults to (√n_src - 1) / √n_src.
    weight_clip : if set, caps weights at ``weight_clip`` after solving and
                  rescales the unclipped weights by one constant so the mean
                  weight is unchanged.  If impossible (clip too small) a
                  warning is issued and the achieved mean is in ``mean_weight_``.
    max_iter    : SLSQP iteration limit.
    max_kernel_bytes : refuse inputs whose n_src × n_src float64 kernel would
                  exceed this many bytes (None disables the guard).
    constraint_tol : tolerance for verifying bounds/normalisation of the
                  solver output.
    """

    def __init__(
        self,
        sigma: Optional[float] = None,
        B: float = 1000.0,
        epsilon: Optional[float] = None,
        weight_clip: Optional[float] = None,
        max_iter: int = 1000,
        max_kernel_bytes: Optional[float] = DEFAULT_MAX_KERNEL_BYTES,
        constraint_tol: float = 1e-6,
    ):
        self.sigma = sigma
        self.B = B
        self.epsilon = epsilon
        self.weight_clip = weight_clip
        self.max_iter = max_iter
        self.max_kernel_bytes = max_kernel_bytes
        self.constraint_tol = constraint_tol

    @staticmethod
    def _rbf_matrix(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
        """K[i, j] = exp(-||X[i] - Y[j]||² / (2σ²)), without an (n, m, d) temporary."""
        return _rbf_kernel(X, Y, sigma)

    @staticmethod
    def _median_bandwidth(X: np.ndarray, Y: np.ndarray, subsample: int = 500) -> float:
        return _median_bandwidth(X, Y, subsample)

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

        Raises
        ------
        MemoryError  if the source kernel would exceed ``max_kernel_bytes``.
        ValueError   for empty/non-finite inputs or an undefined bandwidth.
        RuntimeError if SLSQP fails or returns infeasible/non-finite weights.
        """
        try:
            from scipy.optimize import minimize
        except ImportError as e:
            raise ImportError(
                "KMMWeightEstimator requires scipy. "
                "Install it with: pip install scipy"
            ) from e

        X_src = _check_features(X_src, "X_src")
        X_tgt = _check_features(X_tgt, "X_tgt")
        n_src = len(X_src)
        n_tgt = len(X_tgt)
        _check_kernel_size(n_src, self.max_kernel_bytes)

        sigma = self.sigma if self.sigma is not None else _median_bandwidth(X_src, X_tgt)
        sigma = _check_sigma(sigma)
        self.sigma_ = sigma

        # Build kernel quantities (K_st is reduced row-wise, never stored)
        K_ss = self._rbf_matrix(X_src, X_src, sigma)           # (n_src, n_src)
        kappa = (n_src / n_tgt) * _rbf_row_sums(X_src, X_tgt, sigma)   # (n_src,)

        epsilon = self.epsilon if self.epsilon is not None else (np.sqrt(n_src) - 1.0) / np.sqrt(n_src)
        self.epsilon_ = float(epsilon)

        # QP: min (1/2) w^T K_ss w - κ^T w
        # s.t. 0 ≤ w_i ≤ B,  |mean(w) - 1| ≤ ε
        def objective(w):
            return 0.5 * w @ K_ss @ w - kappa @ w

        def gradient(w):
            return K_ss @ w - kappa

        mean_grad = np.full(n_src, 1.0 / n_src)
        w0 = np.ones(n_src, dtype=np.float64)
        bounds = [(0.0, self.B)] * n_src
        constraints = [
            {"type": "ineq", "fun": lambda w:  w.mean() - 1.0 + epsilon, "jac": lambda w: mean_grad},
            {"type": "ineq", "fun": lambda w: -w.mean() + 1.0 + epsilon, "jac": lambda w: -mean_grad},
        ]

        result = minimize(
            objective,
            w0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": 1e-6},
        )
        self.result_ = result

        w = np.asarray(result.x, dtype=np.float64)
        tol = self.constraint_tol
        problems = []
        if not result.success:
            problems.append(f"SLSQP did not converge ({result.message!r}, nit={result.nit})")
        if not np.isfinite(w).all():
            problems.append("solution contains non-finite values")
        else:
            if w.min() < -tol or w.max() > self.B + tol:
                problems.append(f"bounds violated (min={w.min():.3g}, max={w.max():.3g}, B={self.B})")
            if abs(w.mean() - 1.0) > epsilon + tol:
                problems.append(f"normalisation violated (mean={w.mean():.6f}, ε={epsilon:.4g})")
            if w.sum() <= 0:
                problems.append("all weights are zero")
        if problems:
            raise RuntimeError(
                "KMM weight estimation failed: " + "; ".join(problems)
                + ". Try a different sigma, a larger max_iter, or fewer source samples."
            )

        weights = np.clip(w, 0.0, self.B)
        self.clip_scale_ = 1.0
        if self.weight_clip is not None:
            if not self.weight_clip > 0:
                raise ValueError("weight_clip must be positive.")
            self.clip_scale_, feasible = _clip_scale(weights, self.weight_clip, float(weights.mean()))
            if not feasible:
                warnings.warn(
                    f"weight_clip={self.weight_clip} is too small to preserve the solved mean "
                    f"weight {weights.mean():.4f}; weights are clipped without renormalisation "
                    f"(achieved mean {np.minimum(weights, self.weight_clip).mean():.4f}).",
                    RuntimeWarning, stacklevel=2,
                )
            weights = np.minimum(weights * self.clip_scale_, self.weight_clip)
        self.weights_ = weights.astype(np.float32)
        self.mean_weight_ = float(self.weights_.mean())
        self._source_fingerprint = _array_fingerprint(X_src)
        return self

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the estimated importance weights of the fitted source sample.

        KMM has no out-of-sample extension: weights exist only for the source
        rows passed to ``fit``.  ``X`` may be omitted or must be exactly that
        source array (same rows, same order); anything else raises ValueError.

        Returns
        -------
        weights : (n_src,) float32 array
        """
        if not hasattr(self, "weights_"):
            raise RuntimeError("Call fit() before predict().")
        if X is not None:
            X = np.asarray(X)
            if (X.shape[0] != len(self.weights_)
                    or _array_fingerprint(X) != self._source_fingerprint):
                raise ValueError(
                    "KMM weights are defined only for the fitted source samples; predict(X) "
                    "requires X to be the exact source array passed to fit() (or None). "
                    "Use KLIEPWeightEstimator for out-of-sample weights."
                )
        return self.weights_.copy()


# ─── KMM Trainer ─────────────────────────────────────────────────────────────

class KMMTrainer(_ImportanceWeightedTrainer):
    """
    Instance-based domain adaptation via Kernel Mean Matching.

    Estimates importance weights w(x) = p_target(x) / p_source(x) once at
    initialisation by solving a convex QP, then trains the model with an
    importance-weighted cross-entropy loss (per batch, Σ w ℓ / Σ w).  Batches
    whose weights are all zero are skipped (no update) and counted in the
    history.  An epoch covers the longer of the two loaders.

    Because correction happens at the sample level, KMM does not require the
    model to expose ``encode()`` / ``classify()`` — any standard ``nn.Module``
    with ``forward()`` works.

    Parameters
    ----------
    model          : PyTorch model with standard forward() interface
    source_loader  : labelled source DataLoader (fixed sampler membership);
                     KMM cost is O(n_src²) memory — subsample large sources
    target_loader  : target DataLoader yielding x, (x,) or (x, y); labels are
                     used only for tgt_acc tracking
    sigma          : RBF bandwidth for KMM (None → median heuristic)
    B              : upper bound on each importance weight (default 1000)
    epsilon        : normalisation constraint tolerance (None → (√n-1)/√n)
    weight_clip    : cap weights after solving, preserving their mean when feasible
    lr             : model optimiser learning rate
    device         : 'cuda', 'mps', or 'cpu' (auto-detected if None)
    max_iter       : SLSQP iteration limit
    max_kernel_bytes : memory guard for the n_src × n_src kernel (None disables)

    History keys
    ------------
    epoch, ce_loss (importance-weighted, Σwℓ/Σw over the epoch), mmd_loss (0.0),
    total_loss, src_acc, tgt_acc (None without target labels), mean_weight
    (per source sample seen), max_weight, skipped_zero_weight_batches
    """

    _method_name = "KMM"

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
        max_iter: int = 1000,
        max_kernel_bytes: Optional[float] = DEFAULT_MAX_KERNEL_BYTES,
    ):
        self._estimator = KMMWeightEstimator(
            sigma=sigma, B=B, epsilon=epsilon, weight_clip=weight_clip,
            max_iter=max_iter, max_kernel_bytes=max_kernel_bytes,
        )
        # Fail before reading any data if the source is too large.
        sampler = getattr(source_loader, "sampler", None)
        if sampler is not None:
            try:
                n_selected = len(sampler)
            except TypeError:
                n_selected = None
            if n_selected is not None:
                _check_kernel_size(n_selected, max_kernel_bytes)
        self._setup(model, source_loader, target_loader, lr, device)
