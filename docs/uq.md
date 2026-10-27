# Uncertainty (`shiftkit.uq`)

`shiftkit.uq` adds uncertainty to a **trained model** without changing how it
was trained, so it works with every trainer in [Methods](methods/index.md).
For **regression** it builds prediction intervals around the point predictions.
For **classification** it recalibrates the predicted class probabilities
([KCal](#classification-calibrated-probabilities)).

```python
from shiftkit.uq import predict_arrays, ConformalRegressor, interval_metrics

mean_cal, var_cal, y_cal   = predict_arrays(model, calib_loader)   # held-out labelled data
mean_test, var_test, y_test = predict_arrays(model, test_loader)

cr = ConformalRegressor(alpha=0.1).calibrate(y_cal, mean_cal)      # 90 % intervals
lower, upper = cr.predict_interval(mean_test)
print(interval_metrics(y_test, lower, upper))
```

!!! warning "Calibration data must be held out"
    Use labelled data the model was **not** trained on. Calibrating on training
    data gives intervals that are too narrow.

!!! tip "Worked example with outputs"
    [`examples/uq_galaxies.ipynb`](https://github.com/AleksCipri/ShiftKit/blob/main/examples/uq_galaxies.ipynb)
    calibrates a galaxy classifier trained on SDSS and evaluates it on DESI, with results saved in the notebook.
    The regression tools on this page are covered by tutorials 9–11; they have no notebook.

---

## Which tool?

| Situation | Use | Tutorial |
|-----------|-----|----------|
| No shift, just want intervals | `ConformalRegressor` | [9](tutorials/uq_intervals.md) |
| Model predicts a variance too | `ConformalRegressor(..., scale=sd)` | [9](tutorials/uq_intervals.md) |
| Target inputs differ from source | `importance_weights` + `weights=` | [10](tutorials/uq_shift.md) |
| Uncertainty varies across inputs | `fit_lvd_kernel` + `lvd_interval` | [11](tutorials/uq_lvd.md) |

---

## Classification: calibrated probabilities

A classifier's softmax output is often over-confident: when it says 0.9, it is
right less than 90 % of the time. A model is **calibrated** when the stated
probability matches the observed frequency. `KCalibrator` replaces the last
layer with a kernel density classifier on the model's embedding, learned on
training data and tuned on a held-out calibration set
([Lin, Trivedi & Sun, 2022](https://arxiv.org/abs/2202.07679)).

```python
from shiftkit.uq import embed, KCalibrator, fit_temperature, calibration_report, softmax_np

z_fit, logits_fit, y_fit = embed(model, fit_loader)      # data the network trained on
z_cal, logits_cal, y_cal = embed(model, cal_loader)      # held out
z_test, logits_test, y_test = embed(model, test_loader)

kc = KCalibrator(n_classes=6).fit(z_fit, y_fit)          # learn the kernel
kc.calibrate(z_cal, y_cal)                               # choose the bandwidth
probs = kc.predict_proba(z_test)

T = fit_temperature(logits_cal, y_cal)                   # the one-parameter baseline
print(calibration_report(probs, y_test))
print(calibration_report(softmax_np(logits_test / T), y_test))
```

The model needs `.encode(x)` and `.classify(z)`, as all ShiftKit models do.
`embed` needs unshuffled loaders. `calibration_report` gives `accuracy`,
`brier`, `log_loss`, `ece_conf` (error of the top-class confidence) and
`ece_classwise` (a class-by-class proxy for full calibration). Lower is better
for all but accuracy.

The calibration guarantee is asymptotic and assumes the calibration data come
from the distribution being predicted. Under domain shift it does not apply
([notebook](https://github.com/AleksCipri/ShiftKit/blob/main/examples/uq_galaxies.ipynb)).

---

## Classes and functions

| Name | Purpose |
|------|---------|
| `predict_arrays(model, loader)` | `(mean, var_or_None, y_or_None)` numpy arrays, eval mode |
| `ConformalRegressor(alpha)` | `.calibrate(y, mean, scale=None, weights=None)`, `.predict_interval(mean, scale=None, test_weights=None)` |
| `importance_weights(src_loader, tgt_loader, method)` | KLIEP/KMM weights for the calibration set; returns `(weights, estimator)` |
| `interval_metrics(y, lower, upper)` | `coverage`, `mean_width`, `frac_infinite`, `n` |
| `gaussian_interval(mean, var, alpha)` | Interval implied by a predicted Gaussian (no calibration) |
| `split_quantile`, `weighted_quantile`, `conformal_scores` | The building blocks |
| `fit_lvd_kernel`, `lvd_interval` | Localised intervals ([LVD](https://arxiv.org/abs/2106.00225)-style) |
| `embed(model, loader)` | `(embeddings, logits, labels)` in eval mode |
| `KCalibrator(n_classes)` | `.fit(z, y)`, `.calibrate(z_cal, y_cal)`, `.predict_proba(z)` |
| `fit_temperature(logits, y)` | Temperature-scaling baseline |
| `calibration_report`, `brier_score`, `log_loss`, `confidence_ece`, `classwise_ece` | Calibration metrics |

`importance_weights` needs an unshuffled calibration loader so weights line up
with samples. `scale` must be passed at both calibration and prediction, or at
neither.

---

## What the guarantee says

With exchangeable calibration and test data, a `1 − α` interval covers the true
value at least `1 − α` of the time **on average over test points**.

- It is **marginal**: some inputs can be covered much less, others more
  ([Tutorial 9](tutorials/uq_intervals.md)).
- It is **averaged over calibration sets**: one calibration set of 1,000 points
  can land a few points either side of `1 − α`.
- It does **not** hold under domain shift unless corrected for
  ([Tutorial 10](tutorials/uq_shift.md)); the correction assumes covariate shift
  (same `p(y|x)`), overlapping inputs, and, with estimated weights, is
  approximate.
- LVD guarantees coverage for its kernel-localised distribution, not at every
  input. An interval can be infinite when the calibration set is too small for
  `α`; `interval_metrics` reports this as `frac_infinite`.
- Classification intervals (prediction *sets*) are not implemented; the classification tools recalibrate probabilities.

??? note "References"
    - Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018). Distribution-free predictive inference for regression. *JASA* 113.
    - Tibshirani, Barber, Candès & Ramdas (2019). [Conformal prediction under covariate shift](https://arxiv.org/abs/1904.06019). arXiv:1904.06019.
    - Lin, Trivedi & Sun (2021). [Locally valid and discriminative prediction intervals for deep learning models](https://arxiv.org/abs/2106.00225). *NeurIPS* 34 (LVD).
