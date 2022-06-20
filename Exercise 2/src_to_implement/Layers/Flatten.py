import numpy as np


class Flatten:
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        input_tensor =  np.array(input_tensor)
        return input_tensor.flatten()

    def backward (self , error_tensor):
        error_tensor = np.array(error_tensor)
        return error_tensor.flatten()



