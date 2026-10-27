# Tutorial 10 — Uncertainty under domain shift

Intervals from [Tutorial 9](uq_intervals.md) assume the data you deploy on
looks like the data you calibrated on. Under **domain shift** it does not, and
a "90 %" interval can quietly cover far less.

This tutorial shows the failure and a fix for **covariate shift** — the inputs
`x` change but the rule `y = f(x) + noise` does not.

---

## New classes introduced

| Name | Module | Purpose |
|------|--------|---------|
| `importance_weights` | `shiftkit.uq` | KLIEP/KMM weights for the calibration set |
| `ConformalRegressor(..., weights=)` | `shiftkit.uq` | Weighted calibration |

---

## The idea

Calibration points that look like target data should count more. Each gets a
weight `w(x)` = how much likelier `x` is in the target than in the source. This
is the same weight estimate used by [KLIEP](../methods/kliep.md) and
[KMM](../methods/kmm.md). The intervals are then calibrated as if the
calibration set came from the target.

You need **unlabelled target inputs**, not target labels.

---

## Setup

Source inputs are spread over `(−π, π)`; target inputs sit near `1.5`:

```python
import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from shiftkit import MLPRegressor, SourceOnlyRegressionTrainer
from shiftkit.uq import (predict_arrays, ConformalRegressor,
                         interval_metrics, importance_weights)

torch.manual_seed(0)
rng = np.random.default_rng(0)

def make(n, target):
    x = np.clip(rng.normal(1.5, 0.6, n), -np.pi, np.pi) if target else rng.uniform(-np.pi, np.pi, n)
    y = np.sin(x) + (0.05 + 0.3 * (x + np.pi) / (2 * np.pi)) * rng.standard_normal(n)
    return TensorDataset(torch.tensor(x, dtype=torch.float32)[:, None],
                         torch.tensor(y, dtype=torch.float32)[:, None])

train_s, calib_s = make(3000, False), make(1000, False)
train_t, test_t  = make(1000, True),  make(2000, True)      # train_t: labels never used
L = lambda d, shuffle=False: DataLoader(d, batch_size=256, shuffle=shuffle)

model = MLPRegressor(input_dim=1, latent_dim=32, hidden_dims=(64, 64))
SourceOnlyRegressionTrainer(model, L(train_s, True), L(train_t, True), lr=1e-3).fit(epochs=60)

mean_c, _, y_c = predict_arrays(model, L(calib_s))
mean_t, _, y_t = predict_arrays(model, L(test_t))
```

---

## Plain conformal under shift

```python
cr = ConformalRegressor(alpha=0.1).calibrate(y_c, mean_c)
lower, upper = cr.predict_interval(mean_t)
print(interval_metrics(y_t, lower, upper))
```

## Weighted conformal

```python
w_cal, est = importance_weights(L(calib_s), L(train_t), "kliep",
                                sigma=0.5, n_centers=100, lr=0.1, n_iter=2000)
w_test = est.predict(test_t.tensors[0].numpy())

crw = ConformalRegressor(alpha=0.1).calibrate(y_c, mean_c, weights=w_cal)
lower, upper = crw.predict_interval(mean_t, test_weights=w_test)
print(interval_metrics(y_t, lower, upper))
```

The calibration loader must **not** be shuffled, so weights line up with
samples. `sigma`, `n_centers`, `lr` and `n_iter` are KLIEP settings; see its
page for tuning.

---

## Results over 8 seeds

Same code, seeds 0–7, coverage target 0.9:

| Method | Mean coverage | Range over seeds | Mean width |
|--------|:-------------:|:----------------:|:----------:|
| Plain conformal | 0.817 | 0.782 – 0.845 | 0.721 |
| Weighted conformal | 0.902 | 0.866 – 0.932 | 0.903 |

!!! warning "Not a guarantee"
    Weighted coverage ranged from 0.866 to 0.932 across seeds. The weights are
    estimated, so the guarantee is only approximate, and any single run can
    miss by several points. The intervals are also about 25 % wider — that is
    the price of the correction.

---

## When this can fail

- **The rule itself changes.** Weighting fixes a change in *where* data
  appears, not a change in how `y` depends on `x`. Nothing here can detect
  that without target labels.
- **No overlap.** Where the target has inputs the source never produced, no
  weighting can recover them.
- **Extreme weights.** A few huge weights make the interval depend on a few
  points. Check the effective sample size:

```python
ess = w_cal.sum() ** 2 / (w_cal ** 2).sum()
print(f"effective sample size: {ess:.0f} of {len(w_cal)}")
```

  In these runs it was 303–358 of 1,000. If it is a few dozen, treat the
  intervals with suspicion; they may also come back infinite.
- **Weights from high-dimensional inputs** such as raw images work poorly; see
  the [KLIEP](../methods/kliep.md) notes.

If you do have a few labelled target points, check coverage on them directly
with `interval_metrics`. That is the most reliable test.

See also the [`shiftkit.uq` reference](../uq.md).
