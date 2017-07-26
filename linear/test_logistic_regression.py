import numpy as np

from utils import log_likelihood, log_likelihood_gradient
from logistic_regression import LogisticRegressor

num_samples = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_samples)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_samples)

X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(num_samples), np.ones(num_samples)))

print '\n'

model = LogisticRegressor()
model.fit(X, y, max_itrs=1000000)

print '\n'

print 'Model weights:', model.get_params()

differences = model.predict(X) - y
print 'Number of incorrect predictions:', np.count_nonzero(differences), '/', np.size(differences)

print 'Model accuracy:', model.get_accuracy(X, y)

print '\n'
