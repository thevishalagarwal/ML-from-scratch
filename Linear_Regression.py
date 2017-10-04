import numpy as np
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

class Linear_Regression:

	def __init__(self, num_iter=40000, alpha=0.3):
		self.weights = None
		self.num_iter = num_iter
		self.alpha = alpha


	def fit(self, X, y):
		X = np.insert(X, 0, 1, axis=1)
		m, n = X.shape
		self.weights = np.random.random((n, ))

		for i in xrange(self.num_iter):
			w_grad = X.T.dot(X.dot(self.weights) - y)
			self.weights -= w_grad * (self.alpha/m)

	def predict(self, X):
		X = np.insert(X, 0, 1, axis = 1)
		y_pre = X.dot(self.weights)
		return y_pre


def main():
	
	X, y = datasets.make_regression(n_features=1, n_samples=200, bias=100, noise=5)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

	lReg = Linear_Regression()
	
	lReg.fit(X_train, y_train)
	
	y_predict = lReg.predict(X_test)

	error = np.mean((y_test-y_predict)**2)
	print "Mean Square error : %0.8f"%(error)

	score = r2_score(y_test, y_predict)
	print "R2 Score : %0.8f"%(score)

	plt.scatter(X_test[:, 0], y_test, color='black')
	plt.plot(X_test[:, 0], y_predict, color='blue', linewidth=3)
	plt.title("Linear Regression (%.8f Score)"%score)
	plt.show()



if __name__ == "__main__":
	main()
