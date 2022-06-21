import numpy as np
from Optimization import Optimizers
from Layers.Base import Base
from scipy import signal
from math import ceil

class Conv(Base):
    def __init__(self, stride_shape, convolution_shape , num_kernels : int):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels



        self.bias = 1

        self.weights = np.random.uniform(size=(self.num_kernels,) + convolution_shape)


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        b , c = self.input_tensor.shape[0] , self.input_tensor.shape[1]
        self.output = np.zeros((b , self.num_kernels , *self.input_tensor.shape[2:]))

        print((self.convolution_shape[1]//2, self.convolution_shape[1]//2))

        p_y = self.convolution_shape[1] -1
        p_x = self.convolution_shape[2] -1
        self.input_tensor = np.pad( self.input_tensor , ((0,0) ,(0, 0), (self.convolution_shape[1]//2, p_y - self.convolution_shape[1]//2), (self.convolution_shape[2]//2, p_x - self.convolution_shape[2]//2)))
        print(self.output.shape)
        for image_number in range(b) :
            for kernel_number in range(self.num_kernels):
                print(self.input_tensor[image_number, :].shape)
                output_tensor = signal.convolve(self.input_tensor[image_number,:], self.weights[kernel_number] , mode ="valid")
                output_tensor += self.bias
                if  len(self.stride_shape) > 1 and len(self.input_tensor.shape) > 3 :
                    print("HI")
                    print(output_tensor.shape)
                    output_tensor = output_tensor[0::self.stride_shape[0], 0::self.stride_shape[1]]
                    print(output_tensor.shape)

                elif len(self.stride_shape) < 2 and len(self.input_tensor.shape) > 3 :
                    output_tensor = output_tensor[:,0::self.stride_shape[0], 0::self.stride_shape[0]]
                else:
                    output_tensor = output_tensor[:,0::self.stride_shape[0]]
                self.output[image_number ,kernel_number , : ]=output_tensor

        print("FINAL :" , self.output.shape)
                # convolove
                # add bias
                # subsample in both dimensions
        return self.output

        # self.output_tensor = ndimage.convolve(self.input_tensor , self.weights, mode='constant')
        # self.output_tensor += self.bias


        return self.output_tensor
