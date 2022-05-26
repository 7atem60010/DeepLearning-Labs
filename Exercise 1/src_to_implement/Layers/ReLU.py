from Layers.Base import Base
import numpy as np


class ReLU(Base):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        return

    def forward(self, input_tensor):
        input_tensor *= (input_tensor > 0)
        self.input_tensor = input_tensor
        return input_tensor

    def backward(self, error_tensor):
        output = error_tensor * self.input_tensor
        return output
