# Tutorial 11 — Local intervals with LVD

Plain conformal intervals have one width everywhere. If some inputs are noisier
than others, that is too wide for some and too narrow for others. **LVD**
(Lin, Trivedi & Sun, 2021) fixes this by calibrating each test point mainly
against *similar* calibration points.

---

## New classes introduced

| Name | Module | Purpose |
|------|--------|---------|
| `fit_lvd_kernel` | `shiftkit.uq` | Learn which input directions define "similar" |
| `lvd_interval` | `shiftkit.uq` | Intervals from the learned similarity |

---

## The idea

1. Learn a small projection of the inputs (the *kernel*) on a **fitting** set.
2. For each test point, weight calibration errors by how close they are in
   that projection.
3. Take the 90 % weighted error quantile.

You need three separate labelled sets: **fitting**, **calibration**, **test**.

---

## Example

Noise is high when `x[:, 1] > 0` and low otherwise. `y` itself depends on
`x[:, 0]`. We use a fixed predictor so the example is about the intervals:

```python
import numpy as np, torch
from shiftkit.uq import ConformalRegressor, interval_metrics, fit_lvd_kernel, lvd_interval

rng = np.random.default_rng(0)
dev = torch.device("cpu")

def make(n):
    x = rng.uniform(-3, 3, (n, 4))
    noise = 0.1 + 0.5 * (x[:, 1] > 0)
    return x, np.sin(x[:, 0]) + noise * rng.standard_normal(n)

x_fit, y_fit = make(800)
x_cal, y_cal = make(1000)
x_test, y_test = make(3000)
predict = lambda x: np.sin(x[:, 0])          # stand-in for your model's mean

# The response the kernel learns to predict — see the note below.
response = np.abs(y_fit - predict(x_fit))
kernel = fit_lvd_kernel(x_fit, response, dev, quiet=True)

lower, upper, _ = lvd_interval(
    kernel.transform(x_cal), kernel.transform(x_test),
    np.abs(y_cal - predict(x_cal)), predict(x_test), alpha=0.1, device=dev)
print(interval_metrics(y_test, lower, upper))
```

`kernel.transform(x)` applies the learned projection. Fitting takes seconds on
CPU for a few thousand points.

---

## What matters: the kernel's response

The kernel only learns to separate inputs that change its **response**. Which
response you pick decides what "similar" means. Mean over 6 seeds, target 0.9:

| Noise depends on | Kernel fitted on | Quiet: coverage / width | Noisy: coverage / width |
|------------------|------------------|:-----------------------:|:-----------------------:|
| `x0` (same feature as `y`) | `y` | 0.966 / 0.70 | 0.881 / 1.87 |
| `x1` (unrelated to `y`'s mean) | `y` | 1.000 / 1.49 | 0.816 / 1.58 |
| `x1` | `abs(y − prediction)` | 0.984 / 0.87 | 0.873 / 1.83 |

!!! warning "Fit on something that reflects where errors vary"
    Fitted on `y`, the kernel in row 2 cannot see the noise feature and
    intervals stay almost the same width everywhere. Fitting on the absolute
    residual (row 3) makes it adapt. If you do not know where your model's
    errors vary, use the residual.

Plain conformal on the same data (6 seeds): quiet half coverage **1.00**, noisy
half **0.80**, one width of 1.52 everywhere.

---

## Things to remember

- LVD's guarantee is for the kernel-localised distribution, **not** every input
  separately. Coverage in a noisy region can still fall short (0.87 above).
- Intervals can be **infinite** when the calibration set is too small for
  `alpha`; check `frac_infinite`.
- The kernel is fitted with a fixed seed; set `seed=` for repeatable results.
- On real data the gain can be modest, and LVD does not repair a shift to inputs that the
  calibration set does not cover; check coverage on labelled target data where you have it.
- It uses the calibration set at prediction time, so keep it in memory.
- It does not correct for domain shift. For shift, see
  [Tutorial 10](uq_shift.md).

See also the [`shiftkit.uq` reference](../uq.md).
