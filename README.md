# pysmoothspl
Python wrapper around R's lovely `smooth.spline`

## Example

``` python
import matplotlib.pyplot as plt
import numpy as np
from pysmoothspl import SmoothSpline

n = 1000
x = np.arange(n).astype(np.float)
y = 100 + 50 * np.sin(2 * np.pi * x / 365.0) + np.random.normal(0, 25, n)

spl = SmoothSpline(spar=0.2).fit(x, y)
yhat = spl.predict(x)
spl_smoother = SmoothSpline(spar=0.5).fit(x, y)
yhat_smoother = spl_smoother.predict(x)

fig, ax = plt.subplots(1, figsize=(16, 9))
ax.plot(x, y, 'ro', label='Points')
ax.plot(x, yhat, 'g-', label='Less smooth')
ax.plot(x, yhat_smoother, 'b-', label='Smoother')
ax.legend()
fig.show()
```
![Example](./docs/media/example_splines.png)

## Install

You will need `Cython` (unless/until this repo includes the `*.c` files) and
`numpy`.

### Anaconda

There are two Conda `environment.yaml` files to help guide the installation:

1. `environment.yaml` contains just the needed packages for installation
2. `tests/environment.yaml` contains the packages needed to install and test
   this package

## Tests

This package uses `py.test` to run tests, and you will also need `pandas`:

``` bash
py.test tests/
```

If `rpy2` is installed, the tests will actually calculate the expected values
of calculations using the copy of R it is installed against. For users without
`rpy2`, the test data also contains "cached" answers calculated using R code.

## TODO

* [x] Object oriented sklearn-esque estimator
* [ ] For function inputs: typed memory views > NumPy buffers
* [ ] Kill the GIL
* [ ] CI service tests
    * [ ] Code coverage checks with CI
* [x] Check output against literal R output via rpy2
* [ ] Implement more features
    * [ ] Cross validation
    * [ ] Derivatives (hint: check `bvalue` code for commented out)
* [ ] Work back from fresh copy of R code...
