import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

class logisticRegression:

	def __init__(self, gradient_descent=True, c=0.01, alpha=1):
		self.weights = None
		self.gradient_descent = gradient_descent
		self.c = c
		self.alpha = alpha

	def costFunction(self, X, y):
		m, n = X.shape
		h = sigmoid(X.dot(self.weights))
		reg = (self.weights)**2
		reg[0] = 0
		J = (y.dot((np.log(h)).T) + (1-y).dot((np.log(1-h)).T) + np.sum(reg)/2)/m
		return J

	def gradCostFunction(self, X, y):
		m, n = X.shape
		reg = self.c*self.weights
		reg[0] = 0
		w_grad = (X.T.dot(sigmoid(X.dot(self.weights)) - y) + reg)/m
		return w_grad


	def fit(self, X, y, num_iter=5000):
		X = np.insert(X, 0, 1, axis=1)
		m, n = X.shape

		self.weights = np.random.random((n, ))

		for i in xrange(num_iter):
			temp = self.gradCostFunction(X,y)
			self.weights -= temp*self.alpha

	def predict(self, X):
		X = np.insert(X, 0, 1, axis = 1)
		dot = X.dot(self.weights)
		y_pre = np.round(sigmoid(dot)).astype(int)
		return y_pre


def main():

	data = datasets.load_iris()
	X = normalize(data.data[data.target != 0])
	y = data.target[data.target != 0]
	y[y == 1] = 0
	y[y == 2] = 1

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

	clf = logisticRegression(gradient_descent=True)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	#print y_pred[1:10]

	accuracy = accuracy_score(y_test, y_pred)

	print ("Accuracy:", accuracy)

	
if __name__ == '__main__':
	main()