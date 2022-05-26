from Layers.Base import Base
import numpy as np


class ReLU(Base):
    def __init__(self):
        super().__init__()
        return

    def forward(self, input_tensor):
        input_tensor *= (input_tensor > 0)
        print(input_tensor)
        return input_tensor

    #self.bac
