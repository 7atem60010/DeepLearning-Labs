import numpy as np


class Sigmoid:
    def __init__(self):
        self.fx = None

    def forward(self, input):
        self.fx = 1 / (1 + np.exp(input))
        return self.fx

    def backward(self):
        return self.fx * (1 - self.fx)