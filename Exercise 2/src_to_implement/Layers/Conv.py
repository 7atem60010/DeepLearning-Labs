import numpy as np
from Optimization import Optimizers
from Layers.Base import Base
from scipy import ndimage

class Conv(Base):
    def __init__(self, stride_shape, convolution_shape , num_kernels : int):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.bias = np.random.uniform(size=convolution_shape)
        self.weights = np.random.uniform(size=convolution_shape)


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = ndimage.convolve(self.input_tensor , self.weights, mode='constant')
        self.output_tensor += self.bias
        

        return self.output_tensor
