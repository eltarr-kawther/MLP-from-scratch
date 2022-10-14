import numpy as np

class Perceptron:
    def __init__(self, weights, bias, learningRate) -> None:
        self.weights = weights
        self.bias = bias
        self.learningRate = learningRate
    
    def Activation(self, net):
        """
        This is the Soma method
        """
        return np.where(net>0, 1, 0)
    
    def Propagation(self, inputs):
        """
        This is the Axon method
        """
        net = np.dot(self.weights, inputs) + self.bias
        return self.Activation(net)
    
    def Learning(self, inputs, output):
        """
        This is how a Neuron learns
        """
        initial_output = self.Propagation(inputs)
        self.weights = self.weights + inputs * self.learningRate * (output - initial_output)
        #self.weights = [W + X * self.learningRate * (output - initial_output) for (W, X) in zip(self.weights, inputs)]
        self.bias = self.bias + self.learningRate*(output - initial_output)
        error = np.abs(initial_output - output)
        return error
