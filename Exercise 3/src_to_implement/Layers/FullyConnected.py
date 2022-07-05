import numpy as np
from Optimization import Optimizers
from Layers.Base import Base
from Layers.Initializers import Constant


class FullyConnected(Base):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fan_in = input_size
        self.fan_out = output_size
        self._optimizer = None
        self.trainable = True
        self.bias = np.ones((1, self.fan_out))
        self.weights = np.random.uniform(size=(self.fan_in, self.fan_out))
        # print(self.bias.shape)
        # print(self.weights.shape)
        self.weights = np.concatenate((self.bias, self.weights))
        # print(self.weights.shape)

        self.input_tensor = None
        self.gradient_weights = None
        self.error = None

    def forward(self, input_tensor):
        N = input_tensor.shape[0]
        self.input_tensor = np.concatenate((np.ones((N, 1)), input_tensor), axis=1)
        # print(self.bias.shape, self.input_tensor.shape)
        # self.input_tensor = np.concatenate((np.ones((self.input_tensor.shape[0], 1)), self.input_tensor), axis=1)
        # self.input_tensor = np.concatenate((np.ones((self.input_tensor.shape[0], 1)), self.input_tensor), axis=1)
        self.output_tensor = np.matmul(self.input_tensor, self.weights)
        return self.output_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((2, 1), self.fan_in, self.fan_out)
        self.bias = bias_initializer.initialize(1, self.fan_in, self.fan_out)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, Optimizer):
        self._optimizer = Optimizer

    def gradiant_weights(self):
        return self.gradiant_tensor

    def backward(self, error_tensor):
        # The input is error tensor , error tensor is 

        # Get error tensor for prevouis layer
        error_tensor_prev_layer = np.matmul(error_tensor, self.weights[1:, :].T)

        # Update weights
        if self._optimizer != None:
            # print(self.input_tensor.shape , error_tensor.shape)
            self.gradiant_tensor = np.matmul(self.input_tensor.T, error_tensor)
            self.weights = self._optimizer.calculate_update(self.weights, self.gradiant_tensor)

        return error_tensor_prev_layer

# FC = FullyConnected(4, 3)
