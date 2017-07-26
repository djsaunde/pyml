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
		- MSE = \frac{1}{m} \sum_i (\mathbf{y_} - \mathbf{y})_i^2.
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
		- \nabla_{\mathbf{w}} MSE = 2 * X^{\top} * T * w - 2 * X^{\top} * y (known as the __normal equations__).
	'''
	return np.subtract(np.multiply(2, np.dot(np.dot(X.T, X), w)), np.multiply(2, np.dot(X.T, y)))


def standardize(X):
	'''
	Zero-mean (center) the data and normalize it to have unit variance.

	Inputs:
		- X: The ndarray of training data.
	
	Return:
		- \frac{X - \frac{1}{n} \sum_{i=1}^n X_i}{\frac{1}{n - 1} \sum_{i=1}^n (X_i - \frac{1}{n} \sum_{i=1}^n X_i)^2}.
	'''
	return np.divide(np.subtract(X, np.mean(X, axis=0)), np.std(X, axis=0))


def sigmoid(z):
	'''
	Calculates the logistic function of a vector of categorical "scores", transforming them such that
	they're positive and sum to 1, satisfying conditions necessary for a probability distribution.
	'''
	return 1 / (1 + np.exp(-z))


def log_likelihood(X, y, w):
	'''
	Calculates the log-likelihood of the logistic regression model over the dataset.

	- Inputs:
		- X: An ndarray of regressor variable values; i.e., features; i.e., inputs.
		- y: An ndarray of dependent variable values; i.e., tragets; i.e., labels; i.e., outputs.
		- w: The ndarray of model parameter values.

	- Returns:
		- LL = \sum_{i=1}^n y_i w^{\top} X_i - \text{log} (1 + e^{w^{\top} X_i }).
	'''
	# Calculate per-class "scores".
	z = np.dot(X, w)

	# Return the log-likelihood of the model over the dataset.
	return np.sum(y * z - np.log(1 + np.exp(z)))


def log_likelihood_gradient(X, y, w):
	'''
	Calculates the gradient of the log-likelihood with respect to the weight parameters.

	- Inputs:
		- X: An ndarray of regressor variable values; i.e., features; i.e., inputs.
		- y: An ndarray of dependent variable values; i.e., tragets; i.e., labels; i.e., outputs.
		- w: The ndarray of model parameter values.
	
	- Returns:
		- \nabla_{\mathbf{w}} LL = \mathbf{X}^{\top} (\mathbf{y} - \hat \mathbf{y})
	'''
	return -np.dot(X.T, np.subtract(y, sigmoid(np.dot(X, w))))
