import numpy as np


class Flatten:
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.backward_shape  = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.backward_shape = self.input_tensor.shape
        return self.input_tensor.reshape((self.input_tensor.shape[0] , self.input_tensor.shape[1]*self.input_tensor.shape[2]*self.input_tensor.shape[3]))

    def backward (self , error_tensor):
        return error_tensor.reshape(self.backward_shape)



