"""Glue between trained ShiftKit models and the conformal tools."""

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..methods.metrics import split_target_batch
from ..methods.node_batch import eval_mode
from ..models.networks import VAR_EPS, mean_var_from_head


@torch.no_grad()
def predict_arrays(model, loader: DataLoader, device=None, var_eps: float = VAR_EPS
                   ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Run *model* in eval mode and return ``(mean, var_or_None, y_or_None)``.

    Works for plain regressors (1 output column → ``var`` is None) and for
    ``predict_var=True`` models (2 columns → softplus variance).  The loader
    may yield ``x`` or ``(x, y)``; labels are returned only if present.
    """
    device = torch.device(device) if device else next(model.parameters()).device
    means, variances, ys = [], [], []
    with eval_mode(model):
        for batch in loader:
            x, y = split_target_batch(batch)
            out = model(x.to(device))
            if out.dim() == 2 and out.size(1) == 2:
                m, v = mean_var_from_head(out, var_eps)
                variances.append(v.reshape(-1).cpu())
            else:
                m = out.reshape(-1)
            means.append(m.reshape(-1).cpu())
            if y is not None:
                ys.append(y.reshape(-1).float().cpu())
    if not means:
        raise ValueError("loader is empty")
    cat = lambda c: torch.cat(c).numpy() if c else None
    return cat(means), cat(variances), cat(ys)


def _inputs(loader: DataLoader) -> np.ndarray:
    xs = []
    for batch in loader:
        x, _ = split_target_batch(batch)
        xs.append(x.reshape(x.size(0), -1).cpu().numpy())
    if not xs:
        raise ValueError("loader is empty")
    return np.concatenate(xs)


def importance_weights(source_loader: DataLoader, target_loader: DataLoader,
                       method: str = "kliep", **kwargs) -> Tuple[np.ndarray, object]:
    """Estimate w(x) = p_target(x) / p_source(x) for every source sample in *source_loader*.

    Returns ``(weights, estimator)``.  Use ``estimator.predict(X)`` for new
    inputs (e.g. test points).  ``method`` is ``"kliep"`` or ``"kmm"``; extra
    keyword arguments go to the estimator.  The source loader should be the
    **calibration** set (order is preserved, so weights line up with its
    samples), and must not be shuffled.
    """
    from ..methods.kliep import KLIEPWeightEstimator
    from ..methods.kmm import KMMWeightEstimator
    est = {"kliep": KLIEPWeightEstimator, "kmm": KMMWeightEstimator}.get(method)
    if est is None:
        raise ValueError("method must be 'kliep' or 'kmm'")
    est = est(**kwargs)
    X_src, X_tgt = _inputs(source_loader), _inputs(target_loader)
    est.fit(X_src, X_tgt)
    return est.predict(X_src), est
