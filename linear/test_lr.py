import numpy as np

from utils import mean_squared_error, mean_squared_error_gradient
from linear_regression import LinearRegressor

X = np.random.uniform(size=(100, 5))
y = np.dot(X, np.random.randint(low=-10, high=10, size=5)) + np.random.randint(low=-10, high=10, size=1)

model = LinearRegressor()
model.fit(X, y)

print 'Model weights:', model.w

print 'Difference between model predictions and ground truth:', model.predict(X) - y

print 'Mean squared error between predictions and ground truth:', mean_squared_error(model.predict(X), y)
