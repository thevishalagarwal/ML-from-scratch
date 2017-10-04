import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import Neural
net = Neural.NeuralNet([784, 100, 100, 100, 10])
net.SGD(training_data, 30, 10, 3.0, X_test=test_data)