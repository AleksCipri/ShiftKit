"""
Shared metric conventions for ShiftKit trainers.

Regression
----------
``regression_metrics`` is the single definition used by every trainer, for both
per-epoch training history and ``evaluate``:

* MSE / MAE average over **all output elements** (samples × outputs), so
  ``RMSE = sqrt(MSE)`` is a per-element error;
* R² is computed **per output column** and then averaged uniformly (the
  scikit-learn ``multioutput="uniform_average"`` convention).  Pooling every
  element into one variance would let a useless predictor score R² ≈ 1 when
  outputs have different scales.

Gaussian predictive distributions
---------------------------------
``gaussian_metrics`` scores a predicted mean and variance against observed
targets: Gaussian NLL, coverage of the central 68.27 % and 95.45 % intervals
(|z| ≤ 1 and |z| ≤ 2), and mean predicted standard deviation.  These checks
assume the stated Gaussian form; good values do not prove normality or
robustness under shift.
"""

import math
from typing import Dict, Optional, Tuple

import torch


def regression_metrics(preds: torch.Tensor, ys: torch.Tensor) -> Dict[str, float]:
    """Return ``{"mse", "rmse", "mae", "r2"}`` under the shared convention."""
    preds = preds.detach().float()
    ys = ys.detach().float()
    if preds.numel() == 0:
        raise ValueError("cannot compute regression metrics on an empty set")
    if preds.shape != ys.shape:
        preds = preds.reshape(ys.shape)
    p2 = preds.reshape(ys.shape[0], -1)
    y2 = ys.reshape(ys.shape[0], -1)
    err = p2 - y2
    mse = err.pow(2).mean().item()
    mae = err.abs().mean().item()
    ss_res = err.pow(2).sum(0)
    ss_tot = (y2 - y2.mean(0, keepdim=True)).pow(2).sum(0)
    valid = ss_tot > 0
    r2 = (1.0 - ss_res[valid] / ss_tot[valid]).mean().item() if valid.any() else float("nan")
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": mae, "r2": r2}


def gaussian_metrics(mean: torch.Tensor, var: torch.Tensor, ys: torch.Tensor) -> Dict[str, float]:
    """Gaussian NLL and central-interval coverage of predicted ``N(mean, var)``."""
    mean = mean.detach().float().reshape(-1)
    var = var.detach().float().reshape(-1)
    ys = ys.detach().float().reshape(-1)
    if not (torch.isfinite(mean).all() and torch.isfinite(var).all()):
        raise FloatingPointError("predicted mean/variance contains non-finite values")
    if (var <= 0).any():
        raise ValueError("predicted variances must be positive")
    z = (ys - mean) / var.sqrt()
    nll = 0.5 * (torch.log(2 * math.pi * var) + z.pow(2))
    return {
        "nll": nll.mean().item(),
        "cov68": (z.abs() <= 1.0).float().mean().item(),
        "cov95": (z.abs() <= 2.0).float().mean().item(),
        "mean_sd": var.sqrt().mean().item(),
    }


def split_target_batch(batch) -> Tuple[object, Optional[torch.Tensor]]:
    """Return ``(x, y_or_None)`` for a target batch.

    Target loaders may yield ``x``, ``(x,)`` or ``(x, y)``; labels are only
    used for optional monitoring, never for gradients.
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 1:
            return batch[0], None
        return batch[0], batch[1]
    return batch, None


__all__ = ["regression_metrics", "gaussian_metrics", "split_target_batch"]
