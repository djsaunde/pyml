'''
Helper functions for a variety of machine learning algorithms.
'''

__author__ = 'Dan Saunders'

import numpy as np


def mean_squared_error(y_, y):
	'''
	Returns the mean squared differences between the ndarray of predictions (y_) and the ndarray of
	target values (y). This attains a minimum of 0 if and only if the two tensors are exactly
	equal. The error increases as the Euclidean distance between the predictions and the targets
	increases.

	Inputs:
		- y_: A tensor of predicted values.
		- y: A tensor of the true values being predicted.
	
	Returns:
		- MSE = \frac{1}{m} \sum_i (\mathbf{predictions} - \mathbf{targets})_i^2.
	'''
	return np.divide(np.sum(np.square(np.subtract(y_, y))), y.size)


def mean_squared_error_gradient(X, y, w):
	'''
	Compute the gradient of the mean squared error with respect to the ndarray of parameters w.

	Inputs:
		- X: The ndarray of training data.
		- y: The ndarray of training targets.
		- w: The ndarray of of parameters.
	
	Return:
		- \nabla_{\mathbf{w}} MSE = 2 * X^{\top} * T * w - 2 * X^{\top} * y (known as the __normal equations__)
	'''
	return np.subtract(np.multiply(2, np.dot(np.dot(X.T, X), w)), np.multiply(2, np.dot(X.T, y)))
