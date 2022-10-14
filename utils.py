import matplotlib.pyplot as plt

def get_threshold(perceptron, x):
    weights_0 = perceptron.weights[0]
    weights_1 = perceptron.weights[1]
    bias = perceptron.bias
    threshold = -weights_0 * x - bias
    threshold = threshold / weights_1
    return threshold

def display_threshold(perceptron, ax, xlim, color="black"):    
    x2 = [get_threshold(perceptron, x1) for x1 in xlim]
    
    ax.plot(xlim, x2, color=color)
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)