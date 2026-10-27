"""Post-hoc uncertainty quantification: regression intervals (conformal, LVD) and classifier calibration (KCal)."""

from .conformal import (
    ConformalRegressor, conformal_scores, split_quantile, weighted_quantile,
    interval_metrics, gaussian_interval,
)
from .shift import importance_weights, predict_arrays
from .kcal import (KCalibrator, embed, brier_score, log_loss, confidence_ece, classwise_ece,
                   calibration_report, softmax_np, fit_temperature, reliability_curve)
from .lvd import LVDConfig, fit_lvd_kernel, lvd_interval, lvd_weighted_quantiles

__all__ = [
    "ConformalRegressor", "conformal_scores", "split_quantile", "weighted_quantile",
    "interval_metrics", "gaussian_interval", "importance_weights", "predict_arrays",
    "KCalibrator", "embed", "brier_score", "log_loss", "confidence_ece", "classwise_ece",
    "calibration_report", "softmax_np", "fit_temperature", "reliability_curve", "LVDConfig", "fit_lvd_kernel", "lvd_interval", "lvd_weighted_quantiles",
]
