# Tutorial 9 — Prediction intervals in 5 minutes

A regression model answers "what is `y`?" with one number. In practice you also
need to know **how far off that number might be**. A prediction interval gives
that: "`y` is between 1.2 and 1.8, 90 % of the time".

This tutorial adds intervals to a trained model with **conformal prediction**.
It needs no change to training and no knowledge of statistics.

---

## New classes introduced

| Name | Module | Purpose |
|------|--------|---------|
| `predict_arrays` | `shiftkit.uq` | Run a model, return numpy `mean`, `var`, `y` |
| `ConformalRegressor` | `shiftkit.uq` | Calibrate on held-out data, then build intervals |
| `interval_metrics` | `shiftkit.uq` | Coverage and width of intervals |

---

## The idea

1. Train a model as usual.
2. Keep some labelled data **the model never saw** — the *calibration set*.
3. Measure the model's errors on it. Take the error that 90 % of points stay
   below. Intervals are `prediction ± that error`.

Two numbers describe any interval set:

| Metric | Meaning | Want |
|--------|---------|------|
| **coverage** | share of true `y` inside the interval | close to the target (0.9) |
| **width** | interval length | as small as coverage allows |

A huge interval always has good coverage. Look at both.

---

## Data and model

Noise grows with `x`, so some inputs are harder than others:

```python
import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from shiftkit import MLPRegressor, SourceOnlyGaussianRegressionTrainer
from shiftkit.uq import predict_arrays, ConformalRegressor, interval_metrics, gaussian_interval

torch.manual_seed(0)
rng = np.random.default_rng(0)

def make(n):
    x = rng.uniform(-np.pi, np.pi, n)
    y = np.sin(x) + (0.05 + 0.3 * (x + np.pi) / (2 * np.pi)) * rng.standard_normal(n)
    return TensorDataset(torch.tensor(x, dtype=torch.float32)[:, None],
                         torch.tensor(y, dtype=torch.float32)[:, None])

train, calib, test = make(3000), make(1000), make(2000)
loader = lambda d, shuffle=False: DataLoader(d, batch_size=256, shuffle=shuffle)

model = MLPRegressor(input_dim=1, latent_dim=32, hidden_dims=(64, 64), predict_var=True)
trainer = SourceOnlyGaussianRegressionTrainer(
    model, loader(train, True), loader(train, True), lr=1e-3)
trainer.fit(epochs=60)
```

`predict_var=True` makes the model predict a variance as well as a mean. It is
used in the last section; plain conformal needs only the mean.

---

## Intervals

```python
mean_c, var_c, y_c = predict_arrays(model, loader(calib))
mean_t, var_t, y_t = predict_arrays(model, loader(test))

cr = ConformalRegressor(alpha=0.1).calibrate(y_c, mean_c)   # alpha=0.1 → 90 % target
lower, upper = cr.predict_interval(mean_t)
print(interval_metrics(y_t, lower, upper))
```

---

## Results from one run

CPU, seed 0, the code above:

| Method | Coverage | Mean width |
|--------|:--------:|:----------:|
| Conformal | 0.903 | 0.746 |

Coverage is close to the 0.9 target. Now split the test set by difficulty:

| Test inputs | Coverage | Width |
|-------------|:--------:|:-----:|
| `x < 0` (low noise) | 0.985 | 0.746 |
| `x ≥ 0` (high noise) | 0.825 | 0.746 |

!!! warning "Average coverage hides local failures"
    The guarantee is on **average**. Here the easy inputs are over-covered and
    the hard ones are covered only 82 % of the time, because every interval has
    the same width.

---

## Intervals that follow the model's own uncertainty

Divide each error by the model's predicted standard deviation before
calibrating. Intervals then widen where the model is unsure:

```python
sd_c, sd_t = np.sqrt(var_c), np.sqrt(var_t)

crn = ConformalRegressor(alpha=0.1).calibrate(y_c, mean_c, scale=sd_c)
lower, upper = crn.predict_interval(mean_t, scale=sd_t)
```

| Test inputs | Coverage | Width |
|-------------|:--------:|:-----:|
| all | 0.913 | 0.711 |
| `x < 0` | 0.921 | 0.492 |
| `x ≥ 0` | 0.906 | 0.919 |

Coverage is now even across both halves; intervals are narrower where the
model is confident and wider where it is not. `scale` must be given to both `calibrate` and `predict_interval`.

---

## Why not use the predicted variance directly?

```python
lower, upper = gaussian_interval(mean_t, var_t, alpha=0.1)
```

This gave coverage **0.939** and width 0.783 for a 0.9 target. A predicted
variance is only trustworthy if the model is well calibrated, and nothing
checks that. Conformal calibration does.

---

## Things to remember

- Calibrate on data the model **did not train on**.
- Coverage numbers from one run vary by about ±1 point; one calibration set of
  1,000 points can miss the target slightly. This is expected.
- A calibration set that is too small for `alpha` gives **infinite** intervals
  (`frac_infinite` in `interval_metrics`).
- Intervals assume test data looks like calibration data. When it does not, see
  [Tutorial 10](uq_shift.md).

See also the [`shiftkit.uq` reference](../uq.md).
