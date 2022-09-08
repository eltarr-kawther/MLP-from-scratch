import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self) -> None:
        self.weights = 0
        self.biais = 0
        self.learningRate = 0
    
    def activate(self, signal):
        if signal > 0:
            return 1
        else:
            return 0
    
    def propagation(self, inputs):
        propagate_signal = np.dot(self.weights, inputs) + self.bias
        return self.activate(propagate_signal)
    
    def train(self, inputs, output):
        output_prev = self.propagate(inputs)
        self.weights = [W + X * self.learningRate * (output - output_prev) for (W, X) in zip(self.weights, inputs)]
        self.bias = self.bias + self.learningRate * (output - output_prev)
        error = np.abs(output_prev - output)
        return error