import scipy.sparse as sparse
import numpy as np
from Layers.Base import Base


class BatchNormalization(Base):
    def __init__(self, channels):
        self.channels = channels
        self.weigths = None
        self.bias = None
        self.trainable = True
        self.testing_phase = False

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones((1, self.channels))
        self.bias = np.zeros((1, self.channels))
        # self.weights = np.vstack((self.weights, self.bias))

    def forward(self, input_tensor):
        # self.mean  = np.zeros(input_tensor.shape)
        # self.var = np.zeros(input_tensor.shape)
        # print(self.mean)
        # print(np.array(np.mean(input_tensor , axis=0))[:, np.newaxis] )
        # self.mean =   self.mean + np.mean(input_tensor , axis=0)[:, np.newaxis]
        # print(self.mean)
        print(input_tensor[:, 0])
        print(input_tensor.shape)
        print(np.mean(input_tensor , axis=0))
        # print(input_tensor - np.mean(input_tensor , axis=0))
        # print(np.sqrt(np.power(np.var(input_tensor , axis=0), 2 )+ np.finfo(float).eps))
        upper = input_tensor.T - np.array([np.mean(input_tensor , axis=0)] * input_tensor.shape[0]).T
        print(upper[0])
        # lower = np.sqrt(np.power(np.var(input_tensor , axis=0), 2 )+ np.finfo(float).eps)
        # print(lower)
        # print(upper/lower)
        # print(input_tensor)
        # print(np.mean(input_tensor , axis=0))
        # print(np.var(input_tensor, axis=0))
        output = 0
        return output



        pass


        #return output_tensor

    def backward(self , error_tensor):
       pass

#        return error_tensor_prev

