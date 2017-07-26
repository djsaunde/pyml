import numpy as np

from utils import mean_squared_error, mean_squared_error_gradient, standardize
from linear_regression import LinearRegressor

X = np.random.uniform(size=(100, 5))
y = np.dot(X, np.random.randint(low=-10, high=10, size=5)) + np.random.randint(low=-10, high=10, size=1)

X = standardize(X)

print np.shape(X)
print np.shape(y)

model = LinearRegressor()
model.fit(X, y)

print('Model weights:', model.get_params())

print('Difference between model predictions and ground truth:', model.predict(X) - y)

