import numpy as np


class TanH:
    def __init__(self):
        self.fx = None
        self.trainable = False

    def forward(self, input):
        self.fx = np.tanh(input)
        return self.fx

    def backward(self, error_tensor):
        return (1 - self.fx * self.fx) * error_tensor