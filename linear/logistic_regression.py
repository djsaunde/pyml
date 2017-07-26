'''
Logistic regression from scratch using NumPy.
'''

__author__ = 'Dan Saunders'

import numpy as np

from utils import mean_squared_error, mean_squared_error_gradient, sigmoid, log_likelihood, log_likelihood_gradient

# np.set_printoptions(threshold=np.nan)


class LogisticRegressor(object):
	'''
	To Do.
	'''

	def __init__(self):
		'''
		Logistic regression constructor. Does nothing.
		'''

		self.w = None
	

	def fit(self, X, y, eta=0.001, max_itrs=10000, delta=1e-7):
		'''
		Fit the linear regression model to the data (X, y).

		- Inputs:
			- X: An ndarray of regressor variable values; i.e., features; i.e., inputs.
			- y: An ndarray of dependent variable values; i.e., targets; i.e., labels; i.e., targets.
			- eta: A real-valued parameter representing the learning rate; i.e., gradient descent step size.
			- max_itrs: The maximum number of iterations to run the gradient descent algorithm before quitting.
			- delta: A scalar parameter which determines the reduction in log-likelihood needed to continue
				optimizing the model parameters.
		'''
		# Add bias dimension to original data.
		bias = np.ones((X.shape[0], 1))
		X = np.hstack((X, bias))
		
		# Initialize parameters arbitarily.
		self.set_params(np.random.random(X.shape[1]))

		# Initialize previous log-likelihood
		prev_ll = 0

		# Run regression until we've exceeded 'max_iters'.
		for itr in xrange(max_itrs):
			# Apply gradient descent step to parameter ndarray.
			self.set_params(self.get_params() - np.multiply(eta, self.get_log_likelihood_gradient(X, y)))
			
			# Calculate reduction in log-likelihood from previous iteration
			# Halt if reduction is less than the specified delta parameter
			ll = self.get_log_likelihood(X, y)
			if ll - prev_ll < delta and itr > 10:
				break

			# Print log-liklihood as training progresses
			if itr % 500 == 0:
				print '- Log-likehood (Iteration %d)' % itr, ':', ll

			# Set previous log-likelihood to current log-likelihood before moving on to next iteration
			prev_ll = ll


	def predict(self, X):
		'''
		Predict the dependent \hat\mathbf{y} ndarray given the associated values of regressors ndarray \mathbf{X}.

		- Inputs:
			- X: An ndarray of regressor variable values; i.e., features; i.e., inputs.

		- Returns:
			\mathbf{X}^{\top} \dot \mathbf{w} + \mathbf{b}
		'''
		# As before, add bias dimension to original data.
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

		if X.shape[1] != self.get_params().size:
			raise Exception('Input data incorrectly shaped. Be sure to use the same number of features as was trained on.')

		# Compute predictions via a dot product between the regressors and the parameter + bias vector.
		try:
			return np.round(sigmoid(np.dot(X, self.get_params())))

		except NameError:
			raise Exception('Model has not yet been fitted to data.')

	
	def get_accuracy(self, X, y):
		'''
		Predict the dependent \hat\mathbf{y} ndarray given the associated values of regressors ndarray \mathbf{X} and compare
		them to the ground truth labels \mathbf{y}. Return percentage of correct classifications.

		- Inputs:
			- X: An ndarray of regressor variable values; i.e., features; i.e., inputs.
			- y: An ndarray of ground truth dependent variable values; i.e., targets; i.e., labels.

		- Returns:
			- Percentage of correct classifications.
		'''
		# As before, add bias dimension to original data.
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

		if X.shape[1] != self.get_params().size:
			raise Exception('Input data incorrectly shaped. Be sure to use the same number of features as was trained on.')

		# Compute predictions via a dot product between the regressors and the parameter + bias vector.
		try:
			return np.divide((y == np.round(sigmoid(np.dot(X, self.get_params())))).sum().astype(float), np.size(y))

		except NameError:
			raise Exception('Model has not yet been fitted to data.')

	
	def get_params(self):
		'''
		Returns the parameters \mathbf{w} and b of the linear regression model as a single, concatenated ndarray.
		'''
		
		if self.w is None:
			raise Exception('Model has not yet been fit to data.')

		return self.w

	
	def set_params(self, w):
		'''
		Sets the parameters \mathbf{w} and b of the lienar regresssion model as a single, concatenated ndarray.
		
		- Inputs:
			- w: The value to set self.w to.
		'''

		self.w = w
	

	def get_log_likelihood(self, X, y):
		'''
		Calculates the log-likelihood of the logistic regression model over the dataset.

		- Inputs:
			- X: An ndarray of regressor variable values; i.e., features; i.e., inputs.
			- y: An ndarray of dependent variable values; i.e., tragets; i.e., labels; i.e., outputs.
		
		- Returns:
			- LL = \sum_{i=1}^n y_i w^{\top} X_i - \text{log} (1 + e^{w^{\top} X_i }).
		'''

		return log_likelihood(X, y, self.get_params())

	
	def get_log_likelihood_gradient(self, X, y):
		'''
		Calculates the gradient of the log-likehood of the logistic regression model over the datset.

		- Inputs:
			- X: An ndarray of regressor variable values; i.e., features; i.e., inputs.
			- y: An ndarray of dependent variable values; i.e., tragets; i.e., labels; i.e., outputs.

		- Returns:
			- \nabla_{\mathbf{w}} LL = \mathbf{X}^{\top} (\mathbf{y} - \hat \mathbf{y})
		'''

		return log_likelihood_gradient(X, y, self.get_params())
