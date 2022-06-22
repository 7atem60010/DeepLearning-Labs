import numpy as np
from Optimization import Optimizers
from Layers import Initializers
from Layers.Base import Base
from scipy import signal
from math import ceil


class Pooling(Base):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_layer = np.zeros((input_tensor.shape[0], int((input_tensor.shape[2]-(self.pooling_shape[0]-1)-1)/self.stride_shape[0])+1 , int((input_tensor.shape[3]-(self.pooling_shape[1]-1)-1)/self.stride_shape[1] )+1))
        print(output_layer.shape)
        for b in range(input_tensor.shape[0]):
            image = input_tensor[b, :]
            for c in range(image.shape[0]):
                for y in range(0,output_layer.shape[1]):
                    for x in range(0, output_layer.shape[2]):
                        output_layer[y,x ] = np.max(image[c , y*self.stride_shape[0]:y*(self.stride_shape[0]+1) , x*self.stride_shape[1]:x*(self.stride_shape[1]+1)])

        print(output_layer)
        return output_layer


