import numpy as np

from linear_regression import LinearRegressor


X = np.random.random((100, 10))
y = np.random.random(100)

model = LinearRegressor()
model.fit(X, y)

print model.w

print model.predict(X)
