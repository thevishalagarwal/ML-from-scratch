import numpy as np
import random

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def grad_sigmoid(z):
	return sigmoid(z)*(1-sigmoid(z))

class NeuralNet:

	def __init__(self, size):
		self.num_layers = len(size)
		self.size = size
		self.weights = [np.random.randn(j,i) for i,j in zip(size[:-1], size[1:])]
		self.bias = [np.random.randn(i, 1) for i in size[1:]]

	def feedforward(self, X):
		for b,w in zip(self.bias, self.weights):
			X = sigmoid(w.dot(X)+b)
		return X

	def SGD(self, X_train, epoch, mini_batch_size, alpha, X_test=None):
		if X_test:
			n_test = len(X_test)
		n = len(X_train)

		for i in xrange(epoch):
			random.shuffle(X_train)
			mini_batches = [X_train[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]

			for mini_batch in mini_batches:
				self.update(mini_batch, alpha)

			if X_test:
				print 'Epoch {0}: {1}/{2}'.format(i, self.evaluate(X_test), n_test)
			else:
				print 'Epoch {0} complete'.format(i)

	def update(self, mini_batch, alpha):
		m = len(mini_batch)
		del_b = [np.zeros(b.shape) for b in self.bias]
		del_w = [np.zeros(w.shape) for w in self.weights]
		for x,y in mini_batch:
			d_del_b, d_del_w = self.backprop(x, y)
			del_b = [nb+dnb for nb, dnb in zip(del_b, d_del_b)]
			del_w = [nw+dnw for nw, dnw in zip(del_w, d_del_w)]

		self.weights = [w-(alpha/m)*nw for w,nw in zip(self.weights, del_w)]
		self.bias = [b-(alpha/m)*nb 
						for b, nb in zip(self.bias, del_b)]

	def backprop(self, X, y):
		del_b = [np.zeros(b.shape) for b in self.bias]
		del_w = [np.zeros(w.shape) for w in self.weights]

		activation = X
		activations = [X]
		zs = []
		for b, w in zip(self.bias, self.weights):
			z = (np.dot(w, activation))+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		delta = (activations[-1] - y)*grad_sigmoid(zs[-1])
		del_b[-1] = delta
		del_w[-1] = delta.dot(activations[-2].T)

		for l in xrange(2,self.num_layers):
			z = zs[-l]
			delta = self.weights[-l+1].T.dot(delta) * grad_sigmoid(z)
			del_b[-l] = delta
			del_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (del_b, del_w)

	def evaluate(self, X_test):
		score = [(np.argmax(self.feedforward(x)),y) for (x, y) in X_test]
		score_1 = sum(int(x==y) for (x, y) in score)
		return score_1