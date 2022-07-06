import scipy.sparse as sparse
import numpy as np
from Layers.Base import Base


class Dropout(Base):
    def __init__(self, probability):
        self.probability = probability
        self.trainable = False
        self.testing_phase = False
        self.random_selection = None

    def forward(self, input_tensor):

        output_tensor = input_tensor
        if self.testing_phase == False:
            self.random_selection = sparse.random(output_tensor.shape[0], output_tensor.shape[1], density=self.probability,
                                             data_rvs=np.ones).toarray()
            output_tensor = output_tensor * self.random_selection
            output_tensor = (1 / self.probability) * output_tensor

        return output_tensor

    def backward(self , error_tensor):
        if self.testing_phase == False:
            error_tensor = error_tensor * self.random_selection
            error_tensor_prev = (1 / self.probability) * error_tensor

        return error_tensor_prev

