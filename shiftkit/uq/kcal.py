"""
Kernel-based calibration of classifier probabilities (KCal).

A trained classifier maps an input to an embedding ``z = f(x)`` and then to class
probabilities with a linear layer and softmax.  Those probabilities are often
over-confident.  KCal discards the last layer and predicts with a kernel density
classifier on a held-out *calibration* set instead:

    p_k(x) = Σ_{i: y_i = k} K(z, z_i)  /  Σ_i K(z, z_i),        K(z, z') = exp(-‖Π z − Π z'‖² / b²)

Π is a learned low-dimensional projection (a two-layer MLP with a skip connection) and
``b`` a bandwidth tuned on the calibration set.  Π is trained on the data the network was
trained on, by minimising the log loss of the same kernel classifier, scoring each
point against a class-stratified background sample that excludes the point itself.

This follows Algorithm 1 and Eq. (3), (6) and (12) of

    Lin, Trivedi & Sun (2022). Taking a Step Back with KCal: Multi-Class Kernel-Based
    Calibration for Deep Neural Networks. arXiv:2202.07679.

and is a compact re-implementation, not the authors' code.  The paper's calibration
guarantee is asymptotic and assumes the calibration data are drawn from the distribution
being predicted; it does not hold under domain shift.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..methods.metrics import split_target_batch
from ..methods.node_batch import eval_mode


# ─── embeddings ───────────────────────────────────────────────────────────────

@torch.no_grad()
def embed(model, loader: DataLoader, device=None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return ``(embeddings, logits, labels_or_None)`` for a loader, in eval mode.

    The loader must not shuffle if the result is to be aligned with other arrays.
    ``model`` needs ``.encode(x)`` and ``.classify(z)``.
    """
    device = torch.device(device) if device else next(model.parameters()).device
    zs, logits, ys = [], [], []
    with eval_mode(model):
        for batch in loader:
            x, y = split_target_batch(batch)
            z = model.encode(x.to(device))
            zs.append(z.cpu()); logits.append(model.classify(z).cpu())
            if y is not None:
                ys.append(y.cpu())
    if not zs:
        raise ValueError("loader is empty")
    cat = lambda c: torch.cat(c).numpy() if c else None
    return cat(zs), cat(logits), cat(ys)


# ─── metrics ──────────────────────────────────────────────────────────────────

def _check_probs(probs, labels):
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels)
    if probs.ndim != 2 or len(probs) != len(labels):
        raise ValueError("probs must be (n, K) and match the labels")
    if (probs < -1e-9).any() or not np.allclose(probs.sum(1), 1.0, atol=1e-4):
        raise ValueError("probs must be non-negative and sum to 1 along axis 1")
    return probs, labels.astype(int)


def brier_score(probs, labels) -> float:
    """Multi-class Brier score ``mean_i Σ_k (p_ik − 1[y_i = k])²`` (lower is better)."""
    probs, labels = _check_probs(probs, labels)
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def log_loss(probs, labels, eps: float = 1e-12) -> float:
    """Negative log-likelihood of the true class (lower is better)."""
    probs, labels = _check_probs(probs, labels)
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0))))


def _binned_gap(conf, correct, bins):
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total, n = 0.0, len(conf)
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / n * abs(conf[m].mean() - correct[m].mean())
    return total


def confidence_ece(probs, labels, bins: int = 15) -> float:
    """Expected calibration error of the top-class confidence (equal-width bins)."""
    probs, labels = _check_probs(probs, labels)
    pred = probs.argmax(1)
    return float(_binned_gap(probs.max(1), (pred == labels).astype(float), bins))


def classwise_ece(probs, labels, bins: int = 15) -> float:
    """Mean over classes of the binned gap between p_ik and the frequency of class k.

    A class-wise proxy for full calibration; the paper's own estimator differs.
    """
    probs, labels = _check_probs(probs, labels)
    K = probs.shape[1]
    return float(np.mean([_binned_gap(probs[:, k], (labels == k).astype(float), bins) for k in range(K)]))


def calibration_report(probs, labels, bins: int = 15) -> Dict[str, float]:
    """Accuracy, Brier score, log loss, confidence ECE and class-wise ECE."""
    probs, labels = _check_probs(probs, labels)
    return {"accuracy": float(np.mean(probs.argmax(1) == labels)),
            "brier": brier_score(probs, labels), "log_loss": log_loss(probs, labels),
            "ece_conf": confidence_ece(probs, labels, bins), "ece_classwise": classwise_ece(probs, labels, bins)}


def softmax_np(logits) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def reliability_curve(probs, labels, bins: int = 10) -> Dict[str, np.ndarray]:
    """Binned top-class confidence against accuracy (the data behind a reliability diagram).

    Returns ``confidence``, ``accuracy`` and ``count`` per equal-width bin; empty bins have
    ``count == 0`` and ``nan`` for the other two.  A calibrated model lies on the diagonal.
    """
    probs, labels = _check_probs(probs, labels)
    conf, correct = probs.max(1), (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    out = {"confidence": np.full(bins, np.nan), "accuracy": np.full(bins, np.nan), "count": np.zeros(bins, int)}
    for b in range(bins):
        m = idx == b
        out["count"][b] = int(m.sum())
        if m.any():
            out["confidence"][b], out["accuracy"][b] = conf[m].mean(), correct[m].mean()
    return out


# ─── temperature scaling (baseline) ───────────────────────────────────────────

def fit_temperature(logits, labels, bounds: Tuple[float, float] = (0.05, 50.0)) -> float:
    """Temperature T minimising the log loss of ``softmax(logits / T)`` (Guo et al., 2017).

    The standard one-parameter baseline for calibration.  It changes confidence but never the
    predicted class.  Fit it on held-out data, not on the data the network was trained on.
    """
    from scipy.optimize import minimize_scalar
    logits, labels = np.asarray(logits, dtype=float), np.asarray(labels).astype(int)
    if logits.ndim != 2 or len(logits) != len(labels):
        raise ValueError("logits must be (n, K) and match the labels")
    res = minimize_scalar(lambda T: log_loss(softmax_np(logits / T), labels), bounds=bounds, method="bounded")
    return float(res.x)


# ─── the kernel ───────────────────────────────────────────────────────────────

class _Projection(nn.Module):
    """Two-layer MLP with a skip connection: ``Π(z) = W_s z + W_2 relu(W_1 z)``."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, z):
        return self.skip(z) + self.mlp(z)


def _kernel(a: torch.Tensor, b: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
    """exp(-‖a_i − b_j‖² / bandwidth²) for projected rows."""
    d2 = (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(-1)
    return torch.exp(-d2 / (bandwidth ** 2))


class KCalibrator:
    """Calibrated class probabilities from a kernel density classifier on embeddings.

    Parameters
    ----------
    n_classes  : number of classes K
    proj_dim   : output size d of Π (the paper uses min(embedding size, 32))
    hidden     : hidden width of Π
    n_iters    : maximum training iterations for Π
    batch_size : prediction points per iteration (paper: 64)
    per_class  : background points per class per iteration (paper: m = 20)
    lr         : Adam learning rate
    patience   : stop when the loss has not improved for this many iterations
    seed       : seeds the sampler and the initialisation

    Usage
    -----
    >>> kc = KCalibrator(n_classes=6).fit(z_train, y_train)      # learn Π on the training embeddings
    >>> kc.calibrate(z_cal, y_cal)                               # tune the bandwidth on held-out data
    >>> probs = kc.predict_proba(z_test)
    """

    def __init__(self, n_classes: int, proj_dim: int = 32, hidden: int = 128, n_iters: int = 500,
                 batch_size: int = 64, per_class: int = 20, lr: float = 1e-3, patience: int = 50,
                 seed: int = 0, device: Optional[str] = None):
        self.K, self.proj_dim, self.hidden = n_classes, proj_dim, hidden
        self.n_iters, self.batch_size, self.per_class = n_iters, batch_size, per_class
        self.lr, self.patience, self.seed = lr, patience, seed
        self.device = torch.device(device) if device else torch.device("cpu")
        self.projection: Optional[_Projection] = None
        self.bandwidth_ = 1.0
        self.loss_history_: list = []

    # -- training of Π --------------------------------------------------------
    def fit(self, z_train, y_train) -> "KCalibrator":
        z = torch.as_tensor(np.asarray(z_train), dtype=torch.float32, device=self.device)
        y = np.asarray(y_train).astype(int)
        if z.ndim != 2 or len(z) != len(y):
            raise ValueError("z_train must be (n, h) and match y_train")
        counts = np.bincount(y, minlength=self.K)
        if (counts == 0).any():
            raise ValueError(f"every class needs at least one training example; counts {counts.tolist()}")
        self.mean_, self.std_ = z.mean(0), z.std(0).clamp_min(1e-6)         # standardise the embedding
        zn = (z - self.mean_) / self.std_
        rng = np.random.default_rng(self.seed); torch.manual_seed(self.seed)
        self.projection = _Projection(zn.shape[1], min(self.proj_dim, zn.shape[1]), self.hidden).to(self.device)
        opt = torch.optim.Adam(self.projection.parameters(), lr=self.lr)
        by_class = [np.flatnonzero(y == k) for k in range(self.K)]
        best, best_state, since = np.inf, None, 0
        for it in range(self.n_iters):
            batch = rng.choice(len(y), min(self.batch_size, len(y)), replace=False)
            in_batch = np.zeros(len(y), bool); in_batch[batch] = True
            bg_idx, bg_cls, scale = [], [], []
            for k in range(self.K):
                pool = by_class[k][~in_batch[by_class[k]]]                 # background excludes the prediction batch
                if len(pool) == 0:
                    continue
                m = min(self.per_class, len(pool))
                bg_idx.append(rng.choice(pool, m, replace=False)); bg_cls += [k] * m
                scale += [len(by_class[k]) / m] * m                        # |S_k| / m  (Eq. 12)
            bg_idx = np.concatenate(bg_idx)
            proj = self.projection(zn)
            kern = _kernel(proj[batch], proj[bg_idx]) * torch.as_tensor(scale, dtype=torch.float32, device=self.device)
            onehot = F.one_hot(torch.as_tensor(bg_cls, device=self.device), self.K).float()
            num = kern @ onehot                                            # (B, K)
            p = (num / num.sum(1, keepdim=True).clamp_min(1e-30)).clamp_min(1e-12)
            loss = -torch.log(p[torch.arange(len(batch)), torch.as_tensor(y[batch], device=self.device)]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            self.loss_history_.append(float(loss))
            if loss.item() < best - 1e-4:
                best, since = loss.item(), 0
                best_state = {k: v.detach().clone() for k, v in self.projection.state_dict().items()}
            else:
                since += 1
                if since >= self.patience:
                    break
        if best_state is not None:
            self.projection.load_state_dict(best_state)
        self.projection.eval()
        for p_ in self.projection.parameters():
            p_.requires_grad_(False)
        return self

    # -- inference ------------------------------------------------------------
    def _project(self, z) -> torch.Tensor:
        if self.projection is None:
            raise RuntimeError("call fit() first")
        t = torch.as_tensor(np.asarray(z), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.projection((t - self.mean_) / self.std_)

    def _proba(self, proj_q: torch.Tensor, bandwidth: float, chunk: int = 512) -> np.ndarray:
        out, mass = [], []
        with torch.no_grad():
            for s in range(0, len(proj_q), chunk):
                kern = _kernel(proj_q[s:s + chunk], self._proj_cal, bandwidth)         # (c, N)
                num = kern @ self._onehot_cal
                total = num.sum(1, keepdim=True)
                mass.append(total.squeeze(1).cpu().numpy())
                out.append((num / total.clamp_min(1e-30)).cpu().numpy())
        p, mass = np.concatenate(out), np.concatenate(mass)
        # A query far from every calibration point has (numerically) zero kernel mass.  The test must use the
        # raw mass: after dividing by a clamped total the row no longer sums to 1 but may still look plausible.
        dead = ~np.isfinite(p).all(1) | ~(mass > 1e-30)
        if dead.any():
            p[dead] = self.class_freq_                                                  # fall back to the class frequencies
        return p

    def calibrate(self, z_cal, y_cal, bandwidths: Optional[np.ndarray] = None) -> "KCalibrator":
        """Store the calibration set and choose the bandwidth that minimises its log loss.

        The log loss is evaluated leave-one-out (a calibration point never votes for itself).
        """
        y = np.asarray(y_cal).astype(int)
        if len(y) != len(z_cal):
            raise ValueError("z_cal and y_cal must have the same length")
        if (np.bincount(y, minlength=self.K) == 0).any():
            raise ValueError("every class must appear in the calibration set")
        self._proj_cal = self._project(z_cal)
        self._onehot_cal = F.one_hot(torch.as_tensor(y, device=self.device), self.K).float()
        self.class_freq_ = np.bincount(y, minlength=self.K) / len(y)
        yt = torch.as_tensor(y, device=self.device)
        eye = torch.eye(len(y), device=self.device, dtype=torch.bool)

        def loo_loss(b: float) -> float:
            tot = 0.0
            with torch.no_grad():
                for s in range(0, len(y), 512):
                    kern = _kernel(self._proj_cal[s:s + 512], self._proj_cal, b).masked_fill(eye[s:s + 512], 0.0)
                    num = kern @ self._onehot_cal
                    p = (num / num.sum(1, keepdim=True).clamp_min(1e-30)).clamp_min(1e-12)
                    tot += -torch.log(p[torch.arange(len(p)), yt[s:s + 512]]).sum().item()
            return tot / len(y)

        grid = np.asarray(bandwidths) if bandwidths is not None else np.geomspace(0.05, 20.0, 25)
        losses = np.array([loo_loss(float(b)) for b in grid])
        self.bandwidth_ = float(grid[int(np.argmin(losses))])
        self.bandwidth_grid_, self.bandwidth_losses_ = grid, losses
        return self

    def predict_proba(self, z) -> np.ndarray:
        """Calibrated class probabilities ``(n, K)`` for embeddings *z*."""
        if not hasattr(self, "_proj_cal"):
            raise RuntimeError("call calibrate() first")
        return self._proba(self._project(z), self.bandwidth_)
