"""
KLIEP — Kullback–Leibler Importance Estimation Procedure.

An instance-based domain adaptation method that estimates the density ratio
w(x) = p_target(x) / p_source(x) and uses it to reweight source samples
during training, so that the weighted empirical source distribution matches
the target distribution.

References
----------
Sugiyama, M., Nakajima, S., Kashima, H., von Bünau, P., & Kawanabe, M. (2008).
Direct importance estimation with model selection and its application to
covariate shift adaptation.  *Advances in Neural Information Processing
Systems 20* (NIPS 2007 conference; proceedings published 2008).

ShiftKit chooses σ by the median heuristic (or a user-given value); the
paper's likelihood cross-validation model selection is not implemented.

KLIEPWeightEstimator
--------------------
Standalone NumPy estimator.  Runs projected gradient ascent to maximise the
target log-likelihood (equivalently, to minimise KL(p_target ‖ ŵ·p_source)):

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

    loss = Σ_i w(x_i) · CE(f(x_i), y_i) / Σ_i w(x_i)      (per batch)

Importance weights are estimated once at initialisation on raw input features,
then held fixed during model training.  This makes KLIEP particularly suited
to tabular and low-dimensional data where input-space density estimation is
tractable.
"""

import warnings
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (
    Dataset, DataLoader, IterableDataset, RandomSampler, Subset, default_collate,
)
from tqdm import tqdm
from typing import Optional, List, Tuple

from .base import BaseTrainer
from .metrics import split_target_batch
from .mmd import _auto_device
from .node_batch import batch_accuracy, eval_mode, paired_batches


# ─── Shared kernel helpers (also used by KMM) ─────────────────────────────────

# Rows per chunk are chosen so that each temporary block holds at most this many
# float64 entries (~64 MB).  No (n, m, d) difference tensor is ever formed.
_CHUNK_ELEMENTS = 2 ** 23


def _check_features(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array (n, d); got shape {X.shape}.")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains non-finite values.")
    return X


def _check_sigma(sigma: float) -> float:
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"RBF bandwidth sigma must be finite and positive; got {sigma}.")
    return sigma


def _chunk_rows(n_cols: int) -> int:
    return max(1, _CHUNK_ELEMENTS // max(n_cols, 1))


def _sq_dists_block(X: np.ndarray, Y: np.ndarray, Y_sq: np.ndarray) -> np.ndarray:
    """||x||² + ||y||² − 2 x·y for one row block, clamped at zero."""
    D = (X ** 2).sum(1)[:, None] + Y_sq[None, :] - 2.0 * (X @ Y.T)
    np.maximum(D, 0.0, out=D)
    return D


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """K[i, j] = exp(-||X[i] - Y[j]||² / (2σ²)), computed in row chunks.

    Memory is O(n·m) for the result plus one O(chunk·m) temporary; there is no
    (n, m, d) intermediate.
    """
    Y_sq = (Y ** 2).sum(1)
    K = np.empty((len(X), len(Y)), dtype=np.float64)
    step = _chunk_rows(len(Y))
    scale = -1.0 / (2.0 * sigma ** 2)
    for start in range(0, len(X), step):
        block = _sq_dists_block(X[start:start + step], Y, Y_sq)
        block *= scale
        np.exp(block, out=K[start:start + step])
    return K


def _rbf_row_sums(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Σ_j k(X[i], Y[j]) without materialising the full (n, m) kernel."""
    Y_sq = (Y ** 2).sum(1)
    out = np.zeros(len(X), dtype=np.float64)
    step = _chunk_rows(len(Y))
    scale = -1.0 / (2.0 * sigma ** 2)
    for start in range(0, len(X), step):
        block = _sq_dists_block(X[start:start + step], Y, Y_sq)
        out[start:start + step] = np.exp(block * scale).sum(1)
    return out


def _median_bandwidth(X: np.ndarray, Y: np.ndarray, subsample: int = 500) -> float:
    """Median heuristic: σ = sqrt(median positive pairwise squared distance).

    Computed on a fixed-seed subsample of at most *subsample* points from each
    domain.  Raises if every sampled pair coincides (e.g. constant features),
    because no positive bandwidth can then be inferred from the data.
    """
    rng = np.random.RandomState(42)
    idx_x = rng.choice(len(X), min(subsample, len(X)), replace=False)
    idx_y = rng.choice(len(Y), min(subsample, len(Y)), replace=False)
    Z = np.vstack([X[idx_x], Y[idx_y]])
    D = _sq_dists_block(Z, Z, (Z ** 2).sum(1))
    positive = D[np.triu_indices(len(Z), k=1)]
    positive = positive[positive > 0]
    if positive.size == 0:
        raise ValueError(
            "Median-heuristic bandwidth is undefined: all sampled pairwise distances "
            "are zero (constant or duplicated features). Pass sigma explicitly."
        )
    sigma = float(np.sqrt(np.median(positive)))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(
            f"Median-heuristic bandwidth is not usable (sigma={sigma}). Pass sigma explicitly."
        )
    return sigma


def _clip_scale(weights: np.ndarray, clip: float, target_mean: float) -> Tuple[float, bool]:
    """Return ``(c, feasible)`` with mean(min(c·w, clip)) = target_mean.

    Clipping alone lowers the mean; rescaling the unclipped weights restores it
    while every weight stays ≤ clip.  Infeasible when clip · (#positive) / n is
    below the requested mean (e.g. clip < 1 for KLIEP); then ``(1.0, False)``.
    """
    w = np.asarray(weights, dtype=np.float64)
    n = len(w)
    if np.max(w) <= clip:
        return 1.0, True
    w_desc = np.sort(w)[::-1]
    suffix = np.cumsum(w_desc[::-1])[::-1]                # Σ_{j ≥ k} w_desc[j]
    for k in range(n):                                   # k = number of clipped weights
        if suffix[k] <= 0:
            break
        c = (n * target_mean - k * clip) / suffix[k]
        if c <= 0:
            break
        if c * w_desc[k] <= clip * (1 + 1e-12) and (k == 0 or c * w_desc[k - 1] >= clip * (1 - 1e-12)):
            return float(c), True
    return 1.0, False


# ─── KLIEP weight estimator ───────────────────────────────────────────────────

class KLIEPWeightEstimator:
    """
    Estimates importance weights w(x) ≈ p_target(x) / p_source(x).

    Uses an RBF kernel basis with centres drawn from the target domain.
    Projected gradient ascent enforces non-negativity and the empirical
    normalisation constraint E_{src}[w(x)] = 1 jointly after each step.

    The bandwidth is either given or set by the median heuristic; this class
    does **not** implement the likelihood cross-validation model selection of
    Sugiyama et al.

    Parameters
    ----------
    sigma      : RBF kernel bandwidth σ.  If None, uses the median heuristic
                 computed from the combined source + target features.  Raises
                 ValueError when that is undefined (e.g. constant features).
    n_centers  : number of RBF basis functions (centres sampled from target).
    lr         : gradient-ascent step size for θ.
    n_iter     : number of gradient-ascent iterations.
    weight_clip: if set, weights are capped at ``weight_clip`` and the
                 unclipped weights rescaled by one constant (fitted on the
                 source sample) so the source mean weight stays 1.  If that is
                 impossible (weight_clip too small) a warning is issued, weights
                 are only clipped, and the achieved mean is stored in
                 ``source_mean_weight_``.
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
        """RBF kernel: K[i, l] = exp(-||X[i] - C[l]||² / (2σ²)), chunked."""
        return _rbf_kernel(X, C, sigma)

    @staticmethod
    def _median_bandwidth(X: np.ndarray, Y: np.ndarray, subsample: int = 500) -> float:
        """Median heuristic: σ = median pairwise distance over a subsample."""
        return _median_bandwidth(X, Y, subsample)

    @staticmethod
    def _project_coefficients(theta: np.ndarray, source_mean: np.ndarray) -> np.ndarray:
        """Euclidean projection onto non-negative coefficients with source mean one."""
        positive = source_mean > 0
        basis_means = source_mean[positive]
        proposal = theta[positive]
        order = np.argsort(proposal / basis_means)[::-1]
        sorted_means = basis_means[order]
        sorted_proposal = proposal[order]
        thresholds = (
            (np.cumsum(sorted_means * sorted_proposal) - 1.0)
            / np.cumsum(sorted_means ** 2)
        )
        active = sorted_proposal > thresholds * sorted_means
        threshold = thresholds[np.flatnonzero(active)[-1]]
        projected = np.maximum(theta, 0.0)
        projected[positive] = np.maximum(proposal - threshold * basis_means, 0.0)
        return projected

    def _raw_weights(self, X: np.ndarray) -> np.ndarray:
        Phi = self._rbf(X, self.centers_, self.sigma_)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            raw = Phi @ self.theta_
        return np.maximum(np.where(np.isfinite(raw), raw, 0.0), 0.0)

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
        X_src = _check_features(X_src, "X_src")
        X_tgt = _check_features(X_tgt, "X_tgt")

        sigma = self.sigma if self.sigma is not None else _median_bandwidth(X_src, X_tgt)
        sigma = _check_sigma(sigma)
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
        source_mean = Phi_src.mean(axis=0)
        normalization = source_mean.sum()
        if not np.isfinite(normalization) or normalization <= 0:
            raise ValueError("KLIEP cannot normalize the source kernels; check sigma and domain overlap.")
        theta = np.ones(n_centers, dtype=np.float64) / normalization

        # Gradient ascent
        for _ in range(self.n_iter):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                w_tgt = Phi_tgt @ theta                     # (n_tgt,)
            w_tgt = np.where(np.isfinite(w_tgt), w_tgt, 1e-10)
            w_tgt = np.maximum(w_tgt, 1e-10)

            # ∂L/∂θ_l = (1/n_tgt) Σ_j Φ(x_tgt_j, c_l) / w(x_tgt_j)
            grad = (Phi_tgt / w_tgt[:, np.newaxis]).mean(0)
            grad = np.where(np.isfinite(grad), grad, 0.0)

            theta = self._project_coefficients(theta + self.lr * grad, source_mean)

        self.theta_ = theta

        raw_src = np.maximum(Phi_src @ theta, 0.0)
        if not np.isfinite(raw_src).all() or raw_src.sum() <= 0:
            raise ValueError(
                "KLIEP produced all-zero or non-finite source weights; check sigma "
                "(too small a bandwidth underflows the kernel) and domain overlap."
            )
        self.clip_scale_ = 1.0
        if self.weight_clip is not None:
            if not self.weight_clip > 0:
                raise ValueError("weight_clip must be positive.")
            self.clip_scale_, feasible = _clip_scale(raw_src, self.weight_clip, float(raw_src.mean()))
            if not feasible:
                warnings.warn(
                    f"weight_clip={self.weight_clip} is too small to keep the source mean weight "
                    "at 1; weights are clipped without renormalisation "
                    f"(achieved source mean {np.minimum(raw_src, self.weight_clip).mean():.4f}).",
                    RuntimeWarning, stacklevel=2,
                )
        self.source_mean_weight_ = float(self._finalise(raw_src).mean())
        return self

    def _finalise(self, raw: np.ndarray) -> np.ndarray:
        w = raw * self.clip_scale_
        if self.weight_clip is not None:
            w = np.minimum(w, self.weight_clip)
        return w

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return importance weights for samples X.

        With ``weight_clip`` set, the same fitted rescaling constant is applied
        to every input before clipping, so ``predict(X_src)`` has mean 1 when
        clipping is feasible.

        Parameters
        ----------
        X : (n, d) array of features

        Returns
        -------
        weights : (n,) float32 array, values ≥ 0
        """
        X = _check_features(X, "X")
        return self._finalise(self._raw_weights(X)).astype(np.float32)


# ─── Weighted dataset wrapper ─────────────────────────────────────────────────

class _WeightedDataset(Dataset):
    """Wraps a dataset to append a scalar importance weight per sample."""

    def __init__(self, base: Dataset, weights: np.ndarray):
        if weights.ndim != 1 or len(weights) != len(base):
            raise ValueError("Importance weights must contain one entry per dataset index.")
        self.base = base
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if not torch.isfinite(self.weights[idx]):
            raise ValueError(
                f"Source index {idx} was not included in weight estimation. "
                "Keep the source sampler membership fixed across epochs."
            )
        x, y = self.base[idx]
        return x, y, self.weights[idx]


def _loader_worker_options(loader: DataLoader) -> dict:
    return {
        name: getattr(loader, name)
        for name in (
            "num_workers", "pin_memory", "timeout", "worker_init_fn",
            "multiprocessing_context", "generator", "prefetch_factor",
            "persistent_workers",
        )
    }


def _loader_generators(*loaders) -> list:
    """Every distinct torch.Generator a loader or its samplers may draw from."""
    found = {}
    for loader in loaders:
        batch_sampler = getattr(loader, "batch_sampler", None)
        for owner in (loader, getattr(loader, "sampler", None),
                      batch_sampler, getattr(batch_sampler, "sampler", None)):
            gen = getattr(owner, "generator", None)
            if isinstance(gen, torch.Generator):
                found[id(gen)] = gen
    return list(found.values())


@contextmanager
def _preserved_rng_state(*loaders):
    """Restore loader/sampler generators and the global torch RNG on exit.

    Weight estimation iterates the user's samplers and loaders; without this,
    the first training epoch would not follow the order a seeded generator
    would otherwise produce.
    """
    generators = _loader_generators(*loaders)
    states = [gen.get_state() for gen in generators]
    global_state = torch.get_rng_state()
    try:
        yield
    finally:
        for gen, state in zip(generators, states):
            gen.set_state(state)
        torch.set_rng_state(global_state)


def _build_source_estimation_loader(source_loader: DataLoader):
    """Collect the sampler's source indices once, including its final short batch.

    Call inside :func:`_preserved_rng_state` so the user's generators are not
    advanced.
    """
    if isinstance(source_loader.dataset, IterableDataset) or source_loader.batch_size is None:
        raise ValueError("Importance weighting requires a map-style source dataset and explicit batch_size.")
    sampler = source_loader.sampler
    if getattr(sampler, "replacement", False):
        raise ValueError("Importance weighting does not support source sampling with replacement.")
    if isinstance(sampler, RandomSampler) and sampler.num_samples != len(sampler.data_source):
        raise ValueError(
            "Source sampler membership must stay fixed across epochs, but "
            f"RandomSampler(num_samples={sampler.num_samples}) draws a different subset of the "
            f"{len(sampler.data_source)} samples each epoch. Fix the subset instead, e.g. "
            "DataLoader(Subset(dataset, indices), shuffle=True, ...)."
        )
    indices = np.asarray(list(sampler), dtype=np.int64)
    if indices.size == 0:
        raise ValueError("The source sampler must select at least one sample.")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Importance weighting does not support duplicate source sampler indices.")
    if (indices < 0).any() or (indices >= len(source_loader.dataset)).any():
        raise ValueError("Source sampler indices must lie within the source dataset.")
    if set(np.asarray(list(sampler), dtype=np.int64).tolist()) != set(indices.tolist()):
        raise ValueError(
            "Source sampler membership must stay fixed across epochs (a second pass selected "
            "different samples). Fix the subset instead, e.g. Subset(dataset, indices)."
        )
    loader = DataLoader(
        Subset(source_loader.dataset, indices.tolist()),
        batch_size=source_loader.batch_size,
        collate_fn=source_loader.collate_fn,
        **_loader_worker_options(source_loader),
    )
    return loader, indices


class _WeightedCollate:
    """Apply the original, order-preserving collation to samples and append weights."""

    def __init__(self, collate_fn):
        self.collate_fn = collate_fn

    def __call__(self, samples):
        batch = self.collate_fn([(inputs, labels) for inputs, labels, _ in samples])
        weights = default_collate([weight for _, _, weight in samples])
        return batch[0], batch[1], weights


def _build_indexed_weighted_loader(source_loader: DataLoader, indices, weights: np.ndarray):
    """Associate fitted weights with dataset indices while preserving source batching."""
    if weights.shape != (len(indices),) or not np.isfinite(weights).all():
        raise ValueError("The estimator must return one finite weight per selected source index.")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Importance weights must be non-negative and not all zero.")
    indexed_weights = np.full(len(source_loader.dataset), np.nan, dtype=np.float32)
    indexed_weights[indices] = weights
    return DataLoader(
        _WeightedDataset(source_loader.dataset, indexed_weights),
        batch_sampler=source_loader.batch_sampler,
        collate_fn=_WeightedCollate(source_loader.collate_fn),
        **_loader_worker_options(source_loader),
    )


def _collect_inputs(loader: DataLoader) -> np.ndarray:
    """Collect all input tensors (labels optional) into a flat (n, d) array."""
    batches = []
    for batch in loader:
        x, _ = split_target_batch(batch)
        batches.append(x.reshape(x.size(0), -1).cpu().numpy())
    if not batches:
        raise ValueError("Cannot estimate importance weights from an empty loader.")
    return np.concatenate(batches, axis=0)


# ─── Shared importance-weighted training loop ─────────────────────────────────

class _ImportanceWeightedTrainer(BaseTrainer):
    """Training/evaluation shared by KLIEPTrainer and KMMTrainer.

    Subclasses set ``self._estimator`` and ``self._method_name`` and call
    ``self._setup(...)``.
    """

    _method_name = "importance"

    def _setup(self, model, source_loader, target_loader, lr, device):
        self.device = torch.device(device) if device else _auto_device()
        self.model = model.to(self.device)
        self.target_loader = target_loader
        self.lr = lr
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.history: List[dict] = []
        if len(target_loader) == 0:
            raise ValueError("target_loader must yield at least one batch.")
        self._weighted_source_loader = self._build_weighted_loader(source_loader, target_loader)

    def _build_weighted_loader(
        self, source_loader: DataLoader, target_loader: DataLoader
    ) -> DataLoader:
        """Estimate weights and return a weighted DataLoader for the source domain."""
        print(f"Estimating {self._method_name} importance weights…")
        with _preserved_rng_state(source_loader, target_loader):
            estimation_loader, source_indices = _build_source_estimation_loader(source_loader)
            X_src = _collect_inputs(estimation_loader)
            X_tgt = _collect_inputs(target_loader)

        self._estimator.fit(X_src, X_tgt)
        weights = self._estimator.predict(X_src)            # (n_src,)

        print(
            f"  weights — mean={weights.mean():.3f}  "
            f"std={weights.std():.3f}  "
            f"max={weights.max():.3f}"
        )

        return _build_indexed_weighted_loader(source_loader, source_indices, weights)

    # Kept for backward compatibility with code that called the method.
    def _collect_X(self, loader: DataLoader) -> np.ndarray:
        return _collect_inputs(loader)

    # ── training ───────────────────────────────────────────────────────────

    def fit(self, epochs: int = 10) -> List[dict]:
        """Train for *epochs* epochs. Returns per-epoch history list."""
        for epoch in range(1, epochs + 1):
            stats = self._train_epoch(epoch, epochs)
            self.history.append(stats)
            tgt = "n/a" if stats["tgt_acc"] is None else f"{stats['tgt_acc']*100:.1f}%"
            print(
                f"[{epoch:>3}/{epochs}]  "
                f"CE={stats['ce_loss']:.4f}  "
                f"Total={stats['total_loss']:.4f}  "
                f"Src={stats['src_acc']*100:.1f}%  "
                f"Tgt={tgt}  "
                f"w̄={stats['mean_weight']:.3f}"
            )
        return self.history

    def _train_epoch(self, epoch: int, total_epochs: int) -> dict:
        self.model.train()
        weighted_ce = weight_total = 0.0
        src_correct = tgt_correct = n_src = n_tgt = 0
        weight_max = 0.0
        skipped = 0

        pairs, n_batches = paired_batches(self._weighted_source_loader, self.target_loader)

        for (x_src, y_src, w_src), tgt_batch in tqdm(
            pairs, total=n_batches, desc=f"Epoch {epoch}/{total_epochs}", leave=False
        ):
            x_src = x_src.to(self.device)
            y_src = y_src.to(self.device)
            w_src = w_src.to(self.device)

            batch_weight = w_src.sum()
            n_src       += y_src.size(0)
            weight_total += batch_weight.item()
            weight_max   = max(weight_max, w_src.max().item())

            if batch_weight.item() > 0:
                logits = self.model(x_src)
                src_correct += (logits.argmax(1) == y_src).sum().item()
                # Normalised importance-weighted cross-entropy: Σ w ℓ / Σ w
                ce_per_sample = F.cross_entropy(logits, y_src, reduction="none")
                loss = (ce_per_sample * w_src).sum() / batch_weight

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                weighted_ce += loss.item() * batch_weight.item()
            else:
                # Nothing to learn from this batch: no forward in train mode,
                # no update; source accuracy is monitored in eval mode.
                skipped += 1
                src_correct += batch_accuracy(self.model, x_src, y_src)

            x_tgt, y_tgt = split_target_batch(tgt_batch)
            if y_tgt is not None:
                tgt_correct += batch_accuracy(self.model, x_tgt.to(self.device), y_tgt.to(self.device))
                n_tgt       += y_tgt.size(0)

        if skipped:
            warnings.warn(
                f"Epoch {epoch}: skipped {skipped} source batch(es) whose importance weights "
                "were all zero (no parameter update).", RuntimeWarning, stacklevel=3,
            )
        ce = weighted_ce / weight_total if weight_total > 0 else float("nan")
        return {
            "epoch":       epoch,
            "ce_loss":     ce,
            "mmd_loss":    0.0,
            "total_loss":  ce,
            "src_acc":     src_correct / n_src,
            "tgt_acc":     tgt_correct / n_tgt if n_tgt else None,
            "mean_weight": weight_total / n_src,
            "max_weight":  weight_max,
            "skipped_zero_weight_batches": skipped,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, domain: str = "source") -> dict:
        """Compute accuracy on a labelled DataLoader."""
        correct = total = 0
        with eval_mode(self.model):
            for batch in loader:
                x, y = split_target_batch(batch)
                if y is None:
                    raise ValueError("evaluate() requires a labelled loader yielding (x, y).")
                x, y = x.to(self.device), y.to(self.device)
                correct += (self.model(x).argmax(1) == y).sum().item()
                total   += y.size(0)
        if total == 0:
            raise ValueError("Cannot evaluate on an empty loader.")
        return {"domain": domain, "accuracy": correct / total, "n_samples": total}


# ─── KLIEP Trainer ────────────────────────────────────────────────────────────

class KLIEPTrainer(_ImportanceWeightedTrainer):
    """
    Instance-based domain adaptation via KLIEP importance weighting.

    Estimates the density ratio w(x) = p_target(x) / p_source(x) once at
    initialisation, then trains the model with an importance-weighted loss
    (per batch):

        loss = Σ_i w(xᵢ) · CE(f(xᵢ), yᵢ) / Σ_i w(xᵢ)

    Batches whose weights are all zero are skipped (no update) and counted in
    the history.  An epoch covers the longer of the two loaders.

    Because the correction happens at the sample level rather than the feature
    level, KLIEP does not require the model to expose ``encode()`` /
    ``classify()`` — any standard ``nn.Module`` with ``forward()`` works.

    Parameters
    ----------
    model          : PyTorch model with standard forward() interface
    source_loader  : labelled source DataLoader (fixed sampler membership)
    target_loader  : target DataLoader yielding x, (x,) or (x, y); labels are
                     used only for tgt_acc tracking
    sigma          : RBF bandwidth for KLIEP (None → median heuristic)
    n_centers      : number of RBF basis centres sampled from the target
    kliep_lr       : gradient-ascent learning rate inside KLIEP
    kliep_iter     : number of KLIEP gradient-ascent steps
    weight_clip    : cap importance weights at weight_clip, renormalising the
                     source mean to 1 when feasible (None = no clip)
    lr             : model optimiser learning rate
    device         : 'cuda', 'mps', or 'cpu' (auto-detected if None)

    History keys
    ------------
    epoch, ce_loss (importance-weighted, Σwℓ/Σw over the epoch), mmd_loss (0.0),
    total_loss, src_acc, tgt_acc (None without target labels), mean_weight
    (per source sample seen), max_weight, skipped_zero_weight_batches
    """

    _method_name = "KLIEP"

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
        self._estimator = KLIEPWeightEstimator(
            sigma=sigma, n_centers=n_centers,
            lr=kliep_lr, n_iter=kliep_iter,
            weight_clip=weight_clip,
        )
        self._setup(model, source_loader, target_loader, lr, device)
