from Layers.Base import Base
import numpy as np


class ReLU(Base):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        return

    def forward(self, input_tensor):
        #maybe there is an error here
        input_tensor *= (input_tensor > 0)
        self.output_tensor = input_tensor
        return input_tensor

    def backward(self, error_tensor):
        # you need to mask here, not multiply
        output = error_tensor * (self.output_tensor > 0)
        return output
