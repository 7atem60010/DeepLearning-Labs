import numpy as np
from Optimization import Optimizers
from Layers import Initializers
from Layers.Base import Base
from scipy import signal
from math import ceil


class Conv(Base):
    def __init__(self, stride_shape, convolution_shape , num_kernels : int):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape

        self.one_D = False
        if len(convolution_shape) == 2:
            print('1D')
            convolution_shape = convolution_shape + (1,)
        if type(stride_shape) is list:
            self.one_D = True
            print('1D')
            self.stride_shape = (self.stride_shape[0], self.stride_shape[0])

        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.bias = np.ones(num_kernels) * 0.1
        self.weights = np.random.uniform(size=(self.num_kernels,) + convolution_shape)
        # self.initialize(Initializers.UniformRandom(), Initializers.UniformRandom())

        self._optimizer = None
        self._optimizer_weights = None

    def forward(self, input_tensor):
        if self.one_D:
            self.input_tensor = input_tensor.reshape(input_tensor.shape + (1,))
        else:
            self.input_tensor = input_tensor
        self.batch_size = self.input_tensor.shape[0]
        self.output_tensor = []

        for b in range(self.batch_size):
            kernel_layer = []
            for k in range(self.num_kernels):
                d1, d2 = self.convolution_shape[1] - 1, self.convolution_shape[2] - 1
                image = self.input_tensor[b, :]
                image = np.pad(image, ((0, 0), (d1 // 2, d1 - d1 // 2), (d2 // 2, d2 - d2 // 2)))
                kernel = self.weights[k]
                # print(image.shape, self.weights.shape, kernel.shape)
                k_out = signal.correlate(image, kernel, mode='valid')
                s = k_out.shape[0]
                k_out = k_out[s // 2]
                # print(self.bias[k])
                k_out += self.bias[k]
                k_out = k_out[::self.stride_shape[0], ::self.stride_shape[1]]
                # print(k_out.shape)
                kernel_layer.append(k_out)
                # kernel_layer += self.bias
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
        # print(output_tensor.shape)
        # print(self.weights.shape)
        gradient_input = np.zeros_like(self.input_tensor)


        if self.one_D:
            self.error_tensor = error_tensor.reshape(error_tensor.shape + (1,))
        else:
            self.error_tensor = error_tensor

        self.num_channels = self.weights.shape[1]

        self.prev_error = []
        print(self.output_tensor.shape)

        for b in range(self.batch_size):
            for c in range(self.weights.shape[1]):
                channel_layers = []
                for k in range(self.weights.shape[0]):
                    feature_layer = self.error_tensor[b,k,:]
                    feature_layer = signal.resample(feature_layer,
                                           feature_layer.shape[0] * self.stride_shape[0], axis=0)
                    feature_layer = signal.resample(feature_layer, feature_layer.shape[1] * self.stride_shape[1], axis=1)
                    feature_layer = feature_layer[:self.input_tensor.shape[2], :self.input_tensor.shape[3]]

                    # we need zero-interpolation, so we put zero for interpolated values
                    if self.stride_shape[1] > 1:
                        for i, row in enumerate(feature_layer):
                            for ii, element in enumerate(row):
                                if ii % self.stride_shape[1] != 0:
                                    row[ii] = 0
                    if self.stride_shape[0] > 1:
                        for i, row in enumerate(feature_layer):
                            for ii, element in enumerate(row):
                                if i % self.stride_shape[0] != 0:
                                    row[ii] = 0

                    kernel = self.weights[k,c,:,:]

                    # print("+++")
                    # print(feature_layer.shape)
                    # print(kernel.shape)
                    # print("==")

                    # d1, d2 = self.convolution_shape[1] - 1, self.convolution_shape[2] - 1
                    # feature_layer = np.pad(feature_layer,(((0, 0), (d1 // 2, d1 - d1 // 2), (d2 // 2, d2 - d2 // 2))))


                    feature_out = signal.convolve(feature_layer, kernel, mode='same')

                   # print(feature_out.shape)


                    # feature_out += self.bias[c]

                    channel_layers.append(feature_out)
                temp2 = np.stack(channel_layers, axis=0)

                # after stacking the output of the convolution of each channel,
                # we should sum it over channels to get a 2d one-channel image
                temp2 = temp2.sum(axis=0)

                # element-wise addition of bias for every kernel is NOT NEEDED here.
                gradient_input[b, c] = temp2

            #     self.prev_error.append(channel_layers)
            #
            # self.prev_error = np.array(self.prev_error)




        # Update weights
        # if self._optimizer != None:
        #     d1, d2 = self.convolution_shape[1] - 1, self.convolution_shape[2] - 1
        #     self.prev_output = np.pad(self.prev_error, ((0,0,) , (0, 0), (d1 // 2, d1 - d1 // 2), (d2 // 2, d2 - d2 // 2)))
        #
        #     feature_out = signal.convolve(feature_layer, kernel, mode='same')
        #
        #     pass

        if gradient_input.shape[-1] == 1:
            gradient_input = np.reshape(gradient_input, gradient_input.shape[:-1])


        self.gradient_bias = np.sum(self.error_tensor, axis=(0,2,3))
        # print(self.gradient_bias)
        d1, d2 = self.convolution_shape[1] - 1, self.convolution_shape[2] - 1
        padding = ((0,0), (0,0), (d1//2, d1 - d1//2), (d2//2, d2 - d2//2))
        self.input_tensor_padded = np.pad(self.input_tensor, padding)
        print(self.num_kernels, self.error_tensor.shape, self.input_tensor_padded.shape)
        # print(self.input_tensor_padded.shape, self.output_tensor.shape)
        grad = []
        for b in range(self.error_tensor.shape[0]):
            layer_kernel = []
            for k in range(self.num_kernels):
                E_k = self.error_tensor[b,k,:]
                E_k = np.reshape(E_k, (1,) + E_k.shape)
                layer = self.input_tensor_padded[b, :]
                D_weight = signal.correlate(layer, E_k, mode='valid')
                layer_kernel.append(D_weight)
            grad.append(layer_kernel)
        self.gradient_weights = np.array(grad)
        # self.gradient_weights = np.ones_like(self.weights) * -1
        if self._optimizer:
            self.weights = self._optimizer_weights.calculate_updates(self.weights, self.gradient_weights)
        # Update weights
        # if self._optimizer != None:
        #     # print(self.input_tensor.shape , error_tensor.shape)
        #     print(self.input_tensor.T.shape, self.output_tensor.shape)
        #     self.gradiant_tensor = np.matmul(self.input_tensor.T, self.output_tensor)
        #     self.weights = self._optimizer.calculate_update(self.weights, self.gradiant_tensor)

        return gradient_input


    def initialize(self, weights_initializer, bias_initializer):
        self.bias = bias_initializer.initialize(self.bias.shape, 1, 1)
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape),
                                                      np.prod(self.convolution_shape[1:]) * self.num_kernels)
