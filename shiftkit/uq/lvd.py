"""Localized conformal intervals following Lin, Trivedi & Sun (LVD).

Ported from ``/home/karla/LVD/corrected_lvd/lvd_exact.py`` (same algorithm).

This module follows the authors' reference implementation in
``models/regmodel.py::KernelMLKR`` and
``models/conformal.py::{LocalConditional,LocalConditionalMAD}``:

* a normalized, learned rank-10 Gaussian kernel;
* the published MLKR training defaults (1,000 iterations, batch size 100,
  at most 3,000 kernel neighbours, learning rate 1e-2, patience 50, and the
  leave-one-out-mean self response used by the smoothness experiment); and
* the test-point kernel mass as an atom at ``+inf`` when taking the localized
  weighted conformal quantile.

The training loop is vectorized across the sampled query rows.  It computes
the same summed per-query objective and one Adam update as the reference loop;
the vectorization only removes Python/GPU launch overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class LVDConfig:
    """Defaults used by the official LVD experiment driver."""

    d: int = 10
    n_iters: int = 1000
    max_n: int = 3000
    batch_size: int = 100
    lr: float = 1e-2
    stop_iters: int = 50
    norm: bool = True
    ybar_bias: bool = True


class LVDKernel(nn.Module):
    """Learned Gaussian kernel matching LVD's ``KernelMLKR``."""

    def __init__(self, input_dim: int, d: int = 10, projection_input_dim=None):
        super().__init__()
        if projection_input_dim is None:
            projection_input_dim = input_dim
        self.projection = nn.Linear(projection_input_dim, d, bias=False)
        self.register_buffer("x_mean", torch.empty(input_dim))
        self.register_buffer("x_std", torch.empty(input_dim))
        self.register_buffer("valid", torch.ones(input_dim, dtype=torch.bool))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values[..., self.valid] - self.x_mean[self.valid]) / self.x_std[self.valid]

    def transform(self, values, *, device=None, batch_size=8192) -> np.ndarray:
        """Return A(x-normalized) for reusable exact-LVD inference."""
        if device is None:
            device = self.projection.weight.device
        array = np.asarray(values)
        chunks = []
        self.eval()
        with torch.no_grad():
            for start in range(0, len(array), batch_size):
                tensor = torch.as_tensor(
                    array[start:start + batch_size], dtype=torch.float32, device=device)
                chunks.append(self.projection(self.normalize(tensor)).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def export_numpy(self, prefix: str = "lvd") -> Dict[str, np.ndarray]:
        return {
            f"{prefix}_weight": self.projection.weight.detach().cpu().numpy(),
            f"{prefix}_mean": self.x_mean.detach().cpu().numpy(),
            f"{prefix}_std": self.x_std.detach().cpu().numpy(),
            f"{prefix}_valid": self.valid.detach().cpu().numpy(),
        }


def fit_lvd_kernel(
    features,
    response,
    device,
    *,
    seed: int = 7,
    config: LVDConfig = LVDConfig(),
    quiet: bool = False,
) -> LVDKernel:
    """Fit the official LVD MLKR kernel, with its published defaults."""
    x_numpy = np.asarray(features, dtype=np.float32)
    y_numpy = np.asarray(response, dtype=np.float32).reshape(-1)
    if x_numpy.ndim != 2 or len(x_numpy) != len(y_numpy) or len(y_numpy) < 2:
        raise ValueError("features must be 2D and match a response with at least two rows")
    if not np.isfinite(x_numpy).all() or not np.isfinite(y_numpy).all():
        raise ValueError("LVD kernel fitting inputs must be finite")

    # This ordering matches utils.set_all_seeds in the official repository.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    mean = x_numpy.mean(axis=0)
    std = x_numpy.std(axis=0)
    valid = std > 1e-3 if config.norm else np.ones(x_numpy.shape[1], dtype=bool)
    if not valid.any():
        raise ValueError("LVD normalization removed every feature")
    x_processed = ((x_numpy[:, valid] - mean[valid]) / std[valid]
                   if config.norm else x_numpy[:, valid])
    x = torch.as_tensor(x_processed, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_numpy, dtype=torch.float32, device=device)

    kernel = LVDKernel(
        x_numpy.shape[1], config.d, projection_input_dim=int(valid.sum())).to(device)
    kernel.x_mean.copy_(torch.as_tensor(mean, dtype=torch.float32, device=device))
    safe_std = std.copy()
    safe_std[~valid] = 1.0
    kernel.x_std.copy_(torch.as_tensor(safe_std, dtype=torch.float32, device=device))
    kernel.valid.copy_(torch.as_tensor(valid, dtype=torch.bool, device=device))

    optimizer = torch.optim.Adam(kernel.projection.parameters(), lr=config.lr)
    n_rows = len(y_numpy)
    all_indices = np.arange(n_rows)
    ybar = (y.sum() - y) / (n_rows - 1)
    max_n = min(config.max_n, n_rows - 1) if config.max_n is not None else None
    best_loss = np.inf
    best_iteration = -1
    best_weight = kernel.projection.weight.detach().cpu().clone()
    loss_history = []

    kernel.train()
    grad_was_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    for iteration in range(config.n_iters):
        if config.batch_size is not None and config.batch_size < n_rows:
            selected_numpy = np.random.choice(
                all_indices, config.batch_size, replace=False)
        else:
            selected_numpy = all_indices
        selected = torch.as_tensor(selected_numpy, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        projected_all = kernel.projection(x)
        difference = projected_all[selected, None, :] - projected_all[None, :, :]
        weights = torch.exp(-difference.square().mean(dim=-1))

        if config.ybar_bias:
            candidate_y = y.unsqueeze(0).expand(len(selected_numpy), -1).clone()
            candidate_y[torch.arange(len(selected), device=device), selected] = ybar[selected]
        else:
            candidate_y = y.unsqueeze(0).expand(len(selected_numpy), -1)
            weights[torch.arange(len(selected), device=device), selected] = 0.0

        if max_n is not None:
            top_weights, top_indices = torch.topk(
                weights, max_n, dim=1, largest=True, sorted=False)
            top_y = torch.gather(candidate_y, 1, top_indices)
            estimate = (top_weights * top_y).sum(dim=1) / top_weights.sum(dim=1).clamp_min(1e-7)
        else:
            estimate = (weights * candidate_y).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-7)

        per_query = (estimate - y[selected]).square()
        # The reference calls backward once per query and updates Adam once.
        per_query.sum().backward()
        if kernel.projection.weight.grad is None or not torch.isfinite(
                kernel.projection.weight.grad).all():
            raise FloatingPointError("invalid gradient while fitting the LVD kernel")
        optimizer.step()

        average_loss = float(per_query.detach().mean().cpu())
        loss_history.append(average_loss)
        if average_loss < best_loss:
            best_loss = average_loss
            best_iteration = iteration
            best_weight = kernel.projection.weight.detach().cpu().clone()
        elif iteration > best_iteration + config.stop_iters:
            break
        if not quiet and (iteration == 0 or (iteration + 1) % 100 == 0):
            print(f"    LVD kernel iter {iteration + 1}: loss={average_loss:.6g}", flush=True)

    kernel.projection.weight.data.copy_(best_weight.to(device))
    kernel.eval()
    kernel.loss_history = loss_history
    kernel.best_iteration = best_iteration
    kernel.best_loss = best_loss
    kernel.config = config
    # Restore the caller's autograd state. (The original script left autograd
    # globally disabled, which silently stops any model trained afterwards.)
    torch.set_grad_enabled(grad_was_enabled)
    return kernel


def lvd_weighted_quantiles(
    calibration_transformed,
    test_transformed,
    scores,
    *,
    alpha: float = 0.1,
    device=None,
    batch_size: int = 256,
) -> np.ndarray:
    """Exact LVD quantiles, including the test point as a ``+inf`` atom.

    Inputs are already transformed by the learned rank-``d`` projection.  The
    Gaussian kernel is ``exp(-mean((Ax-Ax')**2))`` and ``K(x,x)=1``.  Returning
    ``inf`` is intentional when the test-point atom is reached; that is part of
    the finite-sample construction, not a numerical failure.
    """
    calibration = np.asarray(calibration_transformed, dtype=np.float32)
    testing = np.asarray(test_transformed, dtype=np.float32)
    values = np.asarray(scores, dtype=float).reshape(-1)
    if calibration.ndim != 2 or testing.ndim != 2 or calibration.shape[1] != testing.shape[1]:
        raise ValueError("transformed calibration/test arrays must be compatible 2D arrays")
    if len(calibration) != len(values) or len(values) == 0:
        raise ValueError("scores must match the non-empty calibration array")
    if not np.isfinite(calibration).all() or not np.isfinite(testing).all():
        raise ValueError("transformed features must be finite")
    if not np.isfinite(values).all():
        raise ValueError("calibration scores must be finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    order = np.argsort(values, kind="mergesort")
    sorted_scores = values[order]
    cal = torch.as_tensor(calibration[order], dtype=torch.float32, device=device)
    result = np.empty(len(testing), dtype=float)
    threshold_fraction = 1.0 - alpha
    with torch.no_grad():
        for start in range(0, len(testing), batch_size):
            test = torch.as_tensor(
                testing[start:start + batch_size], dtype=torch.float32, device=device)
            weights = torch.exp(-(test[:, None, :] - cal[None, :, :]).square().mean(dim=-1))
            # Normalized cumulative mass reaches 1-alpha before +inf iff the
            # finite calibration mass crosses (1-alpha)*(sum K_i + K_self).
            target_mass = threshold_fraction * (weights.sum(dim=1) + 1.0)
            cumulative = torch.cumsum(weights, dim=1)
            reached = cumulative >= target_mass[:, None]
            has_finite = reached.any(dim=1)
            positions = reached.to(torch.int8).argmax(dim=1).cpu().numpy()
            block = np.full(len(test), np.inf, dtype=float)
            finite_numpy = has_finite.cpu().numpy()
            block[finite_numpy] = sorted_scores[positions[finite_numpy]]
            result[start:start + len(test)] = block
    return result


def lvd_interval(
    calibration_transformed,
    test_transformed,
    calibration_scores,
    test_prediction,
    *,
    alpha: float = 0.1,
    test_scale=None,
    device=None,
):
    """Construct exact LVD or LVD-MAD symmetric prediction intervals."""
    prediction = np.asarray(test_prediction, dtype=float).reshape(-1)
    quantile = lvd_weighted_quantiles(
        calibration_transformed, test_transformed, calibration_scores,
        alpha=alpha, device=device)
    scale = (np.ones(len(prediction), dtype=float) if test_scale is None
             else np.abs(np.asarray(test_scale, dtype=float).reshape(-1)) + 1e-5)
    if len(scale) != len(prediction) or len(quantile) != len(prediction):
        raise ValueError("test predictions, scales, and transformed rows must match")
    half_width = quantile * scale
    return prediction - half_width, prediction + half_width, quantile
