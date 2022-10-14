import numpy as np

class Perceptron:
    def __init__(self, weights, bias, learningRate=0.1):
        self.weights = weights
        self.bias = bias
        self.learningRate = learningRate
        self.initial_output = None
    
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
        self.initial_output = self.Propagation(inputs)
        self.weights = [wi + self.learningRate*(output - self.initial_output)*xi 
                        for (wi, xi) in zip(self.weights, inputs)]
        self.bias = self.bias + self.learningRate*(output - self.initial_output)
        error = np.abs(self.initial_output - output)
        return error

