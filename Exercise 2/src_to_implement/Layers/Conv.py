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


        self.one_D = False
        if len(convolution_shape) == 2:
            print('1D')
            convolution_shape = convolution_shape + (1,)
        if type(stride_shape) is list:
            self.one_D = True
            print('1D')
            self.stride_shape = (self.stride_shape[0], self.stride_shape[0])

        self.bias = 1
        self.weights = np.random.uniform(size=(self.num_kernels,) + convolution_shape)

        self._optimizer_bias  = None
        self._optimizer_weights = None


    def forward(self, input_tensor):
        if self.one_D:
            self.input_tensor = input_tensor.reshape(input_tensor.shape + (1,))
        else:
            self.input_tensor = input_tensor
        self.batch_size = self.input_tensor.shape[0]
        self.output_tensor = []
        #print(self.input_tensor.shape)
        for b in range(self.batch_size):
            kernel_layer = []
            for k in range(self.num_kernels):
                d1, d2 = self.convolution_shape[1] - 1, self.convolution_shape[2] - 1
                image = self.input_tensor[b, :]
                image = np.pad(image, ((0, 0), (d1//2, d1 - d1//2), (d2//2, d2 - d2//2)))
                kernel = self.weights[k]
                k_out = signal.correlate(image, kernel , mode='valid')
                s = k_out.shape[0]
                k_out = k_out[s//2]
                k_out += self.bias
                k_out = k_out[::self.stride_shape[0], ::self.stride_shape[1]]
                #print(k_out.shape)
                kernel_layer.append(k_out)
                #kernel_layer += self.bias
            self.output_tensor.append(kernel_layer)

        self.output_tensor = np.array(self.output_tensor)
        if self.output_tensor.shape[-1] == 1:
            self.output_tensor = np.reshape(self.output_tensor, self.output_tensor.shape[:-1])
        return self.output_tensor

    ##########################  Optimizers #################################
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, Optimizer):
        self._optimizer = Optimizer
        self._optimizer_bias = self._optimizer
        self._optimizer_weights = self._optimizer

    ######################### Backward ##########################################
    def backward(self, error_tensor):
        # The input is error tensor , error tensor is
        # Get error tensor for prevouis layer
        print(error_tensor.shape)
        print(self.weights.shape)

        for b in range(self.batch_size):


        error_tensor_prev_layer = np.matmul(error_tensor, self.weights[:-1, :])

        # Update weights
        if self._optimizer != None:
            # print(self.input_tensor.shape , error_tensor.shape)
            self.gradiant_tensor = np.matmul(self.input_tensor.T, error_tensor)
            self.weights = self._optimizer.calculate_update(self.weights, self.gradiant_tensor)

        return error_tensor_prev_layer



