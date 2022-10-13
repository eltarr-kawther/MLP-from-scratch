import numpy as np

class Perceptron:
    def __init__(self, weights, bias, learningRate) -> None:
        self.weights = weights
        self.bias = bias
        self.learningRate = learningRate
    
    def Activation(self, signal):
        """
        This is the Soma method
        """
        return np.where(signal>0, 1, 0)
    
    def Propagation(self, inputs):
        """
        This is the Axon method
        """
        propagated_signal = np.dot(inputs, self.weights) + self.bias
        return self.Activation(propagated_signal)
    
    def Learning(self, inputs, output):
        """
        This is how a Neuron learns
        """
        prev_output = self.Propagation(inputs)
        self.weights = [W + X * self.learningRate * (output - output_prev) for (W, X) in zip(self.weights, inputs)]
        self.bias = self.bias + self.learningRate*(output - prev_output)
        error = np.abs(prev_output - output)
        return error
