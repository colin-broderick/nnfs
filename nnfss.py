import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()


class Layer:
    pass

class Activation:
    pass

class Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def __call__(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

class ReLU(Activation):
    def __call__(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output

class Softmax(Activation):
    def __call__(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

X, y = spiral_data(samples=100, classes=3)

dense1 = Dense(2, 3)
activation1 = ReLU()
dense2 = Dense(3, 3)
activation2 = Softmax()

X = dense1(X)
X = activation1(X)
X = dense2(X)
X = activation2(X)

print(X[:5])
