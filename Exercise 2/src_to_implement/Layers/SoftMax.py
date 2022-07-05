from Layers.Base import Base
import numpy as np

class SoftMax(Base):
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        #print(input_tensor)
        max_Xk = np.amax(input_tensor, axis=1).reshape((-1, 1))
        input_tensor -= max_Xk
        input_tensor = np.exp(input_tensor)
        #print(input_tensor)
        #print(input_tensor)
        sum_Xk = np.sum(input_tensor, axis=1).reshape((-1, 1))
        input_tensor /= sum_Xk
        #print(input_tensor)
        self.output_tensor = input_tensor
        return self.output_tensor

    def backward(self, error_tensor):
        # print(error_tensor)
        # print(self.output_tensor)
        # print(error_tensor * self.output_tensor)
        # print("=====================")
        back_error = self.output_tensor * (error_tensor - np.sum(error_tensor * self.output_tensor , axis=1).reshape((-1, 1)))
        return back_error
