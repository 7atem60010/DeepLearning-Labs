import numpy as np
from Optimization import Optimizers
from Layers.Base import Base


class FullyConnected(Base):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._optimizer = True
        self.trainable = True
        self.weights = np.random.uniform(size=(input_size, output_size))
        self.gradient_weights = None
        self.error = None

    def forward(self, input_tensor):
        output_tensor = np.matmul(input_tensor, self.weights)
        return output_tensor

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def get_optimizer(self):
        return self._optimizer

    def backward(self, error_tensor):
        error_back_tensor = np.matmul(error_tensor, self.weights.T)
        self.error = error_tensor
        return error_back_tensor

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self._optimizer:
            pass


FC = FullyConnected(4, 3)
