"""
Conformal prediction intervals for regression.

Conformal prediction wraps any trained regressor and turns its point
predictions into intervals ``[ŷ − q·s, ŷ + q·s]``.  The width ``q`` is the
⌈(1−α)(n+1)⌉-th smallest *score* on a held-out **calibration** set, where the
score is ``|y − ŷ| / s`` (``s = 1`` for plain split conformal, or a per-sample
scale such as a predicted standard deviation).

Guarantee (and its limits)
--------------------------
If calibration and test points are exchangeable, the interval covers the true
label with probability ≥ 1 − α **on average over test points**.  This is a
marginal statement: it does not hold for every x separately, and it does not
hold under domain shift unless the shift is corrected for.

Importance-weighted conformal prediction (Tibshirani et al., 2019) restores
the guarantee under **covariate shift** — p(x) differs but p(y|x) is the same —
by weighting each calibration score with w(x) = p_target(x)/p_source(x).  It
needs overlapping support and degrades when the weights are extreme.  With a
KLIEP estimator the guarantee is only approximate (the weights are estimated).

References
----------
Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World.*
Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018). Distribution-free
predictive inference for regression. JASA 113(523):1094–1111.
Tibshirani, Barber, Candès & Ramdas (2019). Conformal prediction under
covariate shift. arXiv:1904.06019.
"""

from typing import Optional, Tuple

import numpy as np

_EPS = 1e-12


def _as_1d(a, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D (scalar targets only), got shape {a.shape}")
    return a


def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")


def conformal_scores(y, mean, scale=None) -> np.ndarray:
    """Return ``|y − mean| / scale`` (``scale=None`` means 1)."""
    y, mean = _as_1d(y, "y"), _as_1d(mean, "mean")
    if y.shape != mean.shape:
        raise ValueError("y and mean must have the same length")
    if scale is None:
        return np.abs(y - mean)
    scale = np.abs(_as_1d(scale, "scale")) + _EPS
    if scale.shape != y.shape:
        raise ValueError("scale must have the same length as y")
    return np.abs(y - mean) / scale


def split_quantile(scores, alpha: float = 0.1) -> float:
    """The ⌈(1−α)(n+1)⌉-th smallest score; ``inf`` if n is too small for α."""
    _check_alpha(alpha)
    s = np.sort(_as_1d(scores, "scores"))
    if len(s) == 0:
        raise ValueError("need at least one calibration score")
    k = int(np.ceil((1.0 - alpha) * (len(s) + 1)))
    return float(s[k - 1]) if k <= len(s) else float("inf")


def weighted_quantile(scores, weights, test_weights, alpha: float = 0.1) -> np.ndarray:
    """Covariate-shift quantile, one value per test weight.

    Calibration point *i* gets mass ``w_i / (Σw + w_test)`` and the test point
    itself keeps ``w_test / (Σw + w_test)`` at ``+inf``.  The result is the
    smallest score whose cumulative mass reaches 1 − α (``inf`` if none does).
    """
    _check_alpha(alpha)
    s, w = _as_1d(scores, "scores"), _as_1d(weights, "weights")
    wt = np.atleast_1d(np.asarray(test_weights, dtype=float))
    if s.shape != w.shape or len(s) == 0:
        raise ValueError("scores and weights must be non-empty and the same length")
    if (w < 0).any() or (wt < 0).any() or not np.isfinite(w).all() or not np.isfinite(wt).all():
        raise ValueError("weights must be finite and non-negative")
    order = np.argsort(s, kind="mergesort")
    s_sorted, cum = s[order], np.cumsum(w[order])
    target = (1.0 - alpha) * (cum[-1] + wt)
    idx = np.searchsorted(cum, target * (1.0 - 1e-12), side="left")
    out = np.full(len(wt), np.inf)
    ok = idx < len(s_sorted)
    out[ok] = s_sorted[idx[ok]]
    return out


class ConformalRegressor:
    """Calibrate once, then turn point predictions into intervals.

    Parameters
    ----------
    alpha : miscoverage level; intervals target 1 − α coverage (default 0.1 → 90 %)

    Example
    -------
    >>> cr = ConformalRegressor(alpha=0.1).calibrate(y_cal, mean_cal)
    >>> lo, hi = cr.predict_interval(mean_test)
    """

    def __init__(self, alpha: float = 0.1):
        _check_alpha(alpha)
        self.alpha = alpha
        self.scores_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None

    # -- calibration -------------------------------------------------------
    def calibrate(self, y, mean, scale=None, weights=None) -> "ConformalRegressor":
        """Store calibration scores.

        ``scale`` (e.g. a predicted standard deviation) gives *normalised*
        conformal intervals whose width follows the model's own uncertainty.
        ``weights`` (importance weights of the calibration points) switches to
        the covariate-shift version; pass ``test_weights`` at prediction time.
        """
        self.scores_ = conformal_scores(y, mean, scale)
        self.normalised_ = scale is not None
        if weights is None:
            self.weights_ = None
        else:
            w = _as_1d(weights, "weights")
            if w.shape != self.scores_.shape:
                raise ValueError("weights must match the calibration set")
            self.weights_ = w
        return self

    def _check(self):
        if self.scores_ is None:
            raise RuntimeError("call calibrate() first")

    # -- prediction --------------------------------------------------------
    def quantile(self, test_weights=None, n: Optional[int] = None) -> np.ndarray:
        """Score quantile(s): a scalar array for split, one per test point if weighted."""
        self._check()
        if self.weights_ is None:
            if test_weights is not None:
                raise ValueError("calibrate() was called without weights; "
                                 "pass weights= there to use test_weights")
            return np.array([split_quantile(self.scores_, self.alpha)])
        if test_weights is None:
            raise ValueError("weighted calibration needs test_weights at prediction time")
        return weighted_quantile(self.scores_, self.weights_, test_weights, self.alpha)

    def predict_interval(self, mean, scale=None, test_weights=None) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)``; infinite where the data cannot support 1 − α."""
        self._check()
        mean = _as_1d(mean, "mean")
        if self.normalised_ and scale is None:
            raise ValueError("calibrated with a scale; pass scale= for the test points")
        if not self.normalised_ and scale is not None:
            raise ValueError("calibrated without a scale; do not pass scale=")
        q = self.quantile(test_weights)
        s = 1.0 if scale is None else np.abs(_as_1d(scale, "scale")) + _EPS
        half = q * s
        return mean - half, mean + half


def interval_metrics(y, lower, upper) -> dict:
    """Empirical coverage, mean width and the share of infinite intervals.

    Width is averaged over finite intervals only; check ``frac_infinite``.
    """
    y, lo, hi = _as_1d(y, "y"), _as_1d(lower, "lower"), _as_1d(upper, "upper")
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError("y, lower and upper must have the same length")
    fin = np.isfinite(lo) & np.isfinite(hi)
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean((hi - lo)[fin])) if fin.any() else float("inf"),
        "frac_infinite": float(1.0 - fin.mean()),
        "n": int(len(y)),
    }


def gaussian_interval(mean, var, alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Interval ``mean ± z·sd`` implied by a predicted Gaussian (no calibration)."""
    from statistics import NormalDist
    _check_alpha(alpha)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    mean, var = _as_1d(mean, "mean"), _as_1d(var, "var")
    sd = np.sqrt(np.maximum(var, 0.0))
    return mean - z * sd, mean + z * sd
