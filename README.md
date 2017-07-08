# PyML

Ongoing attempts to implement machine learning algorithms in Python.

## Linear Regression

Found in `linear/linear_regression.py`. Import the `LinearRegressor` object from this file. One can create and fit the model as:

```
from linear_regression import LinearRegressor

# instantiate the model
model = LinearRegressor()
# assuming (X, y) are a tuple of ndarrays of data examples and output targets
# change optional argument values to best suit your machine learning task
model.fit(X, y, eta=0.0001, standardization=True, delta=1e-16, max_itrs=10000, convex_opt=False)
```
