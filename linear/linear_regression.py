'''
Linear regression from scratch using NumPy.
'''

__author__ = 'Dan Saunders'

import numpy as np

from utils import mean_squared_error, mean_squared_error_gradient


class LinearRegressor(object):
	'''
	Given a set of m sample pairs (x_1^{(i)}, ..., x_n^{(i)}, y_i)_{i=1}^m, we assume that the relationship between the dependent variable y and
	the vector of regressors x is linear (really, we assume the relationship is _affine_). Then, the model takes the form:
	
	y_i = w_0 + w_1 x_1^{(i)} + x_2^{(i)} + ... + w_n x_n^{(i)} = w_0 + \mathbf{w}^\top \mathbf{x}^{(i)}, i = 1, ..., n,

	where \mathbf{w}^\top \mathbf{x}^{(i)} denotes the inner product between vectors \mathbf{w} and \mathbf{x}.

	Often, these n equations are stacked together and written in vector form as \mathbf{y} = \mathbf{Xw} + \mathbf{b}, where \mathbf{y} \in
	\mathbb{R}^m is the vector of dependent variables, \mathbf{x} \in \mathbb{R}^{m \times n} is the matrix of stacked vectors of regressors,
	\mathbf{W} \in \mathbf{R}^n is the vector of parameters, and b = w_0 the bias parameter (offsetting the predicting from the origin).
	'''

	def __init__(self):
		'''
		Linear regression constructor. Does nothing.
		'''

		self.w = None
	

	def fit(self, X, y, eta=0.001, standardization=True, delta=1e-16, max_itrs=10000, convex_opt=False):
		'''
		Fit the linear regression model to the data (X, y).

		- Inputs:
			- X: An ndarray of regressor variable values; i.e., features; i.e., inputs.
			- y: An ndarray of dependent variable values; i.e., targets; i.e., labels; i.e., targets.
			- eta: A real-valued parameter representing the learning rate; i.e., gradient descent step size.
			- standardization: Whether to apply the standardization pre-processing step.
			- delta: Tolerance parameter for early-stopping the gradient descent algorithm.
			- max_itrs: The maximum number of iterations to run the gradient descent algorithm before quitting.
			- convex_opt: Whether or not to use convex optimization (i.e., gradient descent) instead of fitting
				the model with the normal equations in a single step.
		'''
		# Data standardization.
		if standardization:
			X = np.divide(np.subtract(X, np.mean(X, axis=0)), np.std(X, axis=0))

		# Add bias dimension to original data.
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

		# Initialize parameters arbitarily.
		self.w = np.random.random(X.shape[1])

		if convex_opt:
			# Run regression until MSE is less than 'eps' (epsilon) or we've exceeded 'max_iters'.
			for itr in xrange(max_itrs):
				# Calculate mean squared error (MSE)
				# If the current MSE is less than the specified tolerance (delta), break out of gradient descent
				if self.get_mean_squared_error(X, y) < delta:
					break

				# Apply gradient descent step to parameter ndarray.
				self.w -= np.multiply(eta, get_mean_squared_error_gradient(X, y))
		
		else:
			# Use normal equations to fit linear regression model in one step.
			self.w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))


	def predict(self, X, standardization=True):
		'''
		Predict the dependent \mathbf{y} ndarray given the associated values of regressors ndarray \mathbf{X}.

		- Inputs:
			- X: An ndarray of regressor variable values; i.e., features; i.e., inputs.
			- standardization: Whether to zero-mean and normalize the data to have unit variance.

		- Returns:
			\mathbf{X}^{\top} \dot \mathbf{w} + \mathbf{b}
		'''

		# Data standardization.
		if standardization:
			X = np.divide(np.subtract(X, np.mean(X, axis=0)), np.std(X, axis=0))

		# As before, add bias dimension to original data.
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

		if X.shape[1] != self.w.size:
			raise Exception('Input data incorrectly shaped. Be sure to use the same number of features as was trained on.')

		# Compute predictions via a dot product between the regressors and the parameter + bias vector.
		try:
			return np.dot(X, self.w)
		except NameError:
			raise Exception('Model has not yet been fitted to data.')
	
	
	def get_params(self):
		'''
		Returns the parameters \mathbf{w} and b of the linear regression model as a single, concatenated ndarray.
		'''
		
		if self.w is None:
			raise Exception('Model has not yet been fit to data.')

		return self.w
	

	def get_mean_squared_error(self, X, y, standardization=True):
		'''
		Gets the mean squared error of the model evaluated on the dataset (X, y).

		- Inputs:
			- X: An ndarray of regressor variable values; i.e., features; i.e., inputs.
			- y: An ndarray of dependent variable values; i.e., targets; i.e., labels, i.e., outputs.
			- standardization: Whether to zero-mean and normalize the data to have unit variance.

		- Returns:
			- MSE = \frac{1}{m} \sum_i (\mathbf{predictions} - \mathbf{targets})_i^2.
		'''

		# Data standardization.
		if standardization:
			X = np.divide(np.subtract(X, np.mean(X, axis=0)), np.std(X, axis=0))

		# As before, add bias dimension to original data.
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

		return mean_squared_error(np.dot(X, self.w), y)
	
	
	def get_mean_squared_error_gradient(self, X, y):
		'''
		Gets the gradient of the mean squared error of the model evaluated on the dataset (X, y).

		- Inputs:
			- X: An ndarray of regressor variable values; i.e., features; i.e., inputs.
			- y: An ndarray of dependent variable values; i.e., targets; i.e., labels, i.e., outputs.

		- Returns:
			- \nabla_{\mathbf{w}} MSE = 2 * X^{\top} * T * w - 2 * X^{\top} * y (known as the __normal equations__)
		'''
		
		return mean_squared_error_gradient(X, y, self.w)

