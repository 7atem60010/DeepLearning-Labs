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
#            print('1D')
            convolution_shape = convolution_shape + (1,)
        if type(stride_shape) is list:
            self.one_D = True
#            print('1D')
            self.stride_shape = (self.stride_shape[0], self.stride_shape[0])

        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.bias = np.random.rand(num_kernels)
        self.weights = np.random.uniform(size=(self.num_kernels,) + convolution_shape)

        self._optimizer = None
        self._bias_optimizer = None
        self.gradient_bias = None
        self.gradient_weights = np.zeros_like(self.weights)

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

    @optimizer.deleter
    def optimizer(self):
        del self._optimizer

    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @bias_optimizer.setter
    def bias_optimizer(self, Optimizer):
        self._bias_optimizer = Optimizer

    @bias_optimizer.deleter
    def bias_optimizer(self):
        del self._bias_optimizer

    ######################### Backward ##########################################
    def backward(self, error_tensor):
        # The input is error tensor , error tensor is
        # Get error tensor for prevouis layer
        # print(output_tensor.shape)
        # print(self.weights.shape)
        gradient_input = np.zeros_like(self.input_tensor)

        # Handling 1_D and 2_D stuff
        if self.one_D:
            self.error_tensor = error_tensor.reshape(error_tensor.shape + (1,))
        else:
            self.error_tensor = error_tensor

        self.num_channels = self.weights.shape[1]
        self.prev_error = []

        #print(self.output_tensor.shape)
        ########################## Input gradients #####################################
        for b in range(self.batch_size):
            for c in range(self.weights.shape[1]):
                channel_layers = []
                for k in range(self.weights.shape[0]):

                    # UpSampling

                    feature_layer = self.error_tensor[b,k,:]
                    print("befoooooooooooooooore" , feature_layer.shape)
                    print(feature_layer)
                    feature_layer = signal.resample(feature_layer,
                                           feature_layer.shape[0] * self.stride_shape[0], axis=0)
                    feature_layer = signal.resample(feature_layer, feature_layer.shape[1] * self.stride_shape[1], axis=1)
                    feature_layer = feature_layer[:self.input_tensor.shape[2], :self.input_tensor.shape[3]]

                    print("aaaaaaaaaaaaaaaaafter " , feature_layer.shape)
                    print(feature_layer)
                    # Zero-interpolation
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


                    feature_out = signal.convolve(feature_layer, kernel, mode='same')

                    channel_layers.append(feature_out)
                stacked_channels = np.stack(channel_layers, axis=0)
                stacked_channels = stacked_channels.sum(axis=0)

                gradient_input[b, c] = stacked_channels

        ##################################  Weights calculations ##################################

        # we only have 2D images for this part in our tests here.
        if len(self.convolution_shape) == 3 and self.error_tensor.shape[-1] != 1:
            # gradient of sth has always the same shape as it.
            # here "temp_gradient_weights" has one more dimension for the batches.
            temp_gradient_weights = np.zeros((error_tensor.shape[0], self.weights.shape[0], self.weights.shape[1],
                                              self.weights.shape[2], self.weights.shape[3]))

            # [PADDING] of input's width and height
            conv_plane_out = []
            for batch in range(self.input_tensor.shape[0]):
                ch_conv_out = []
                # loop over different kernels (output channels)
                for out_ch in range(self.input_tensor.shape[1]):
                    ch_conv_out.append(np.pad(self.input_tensor[batch, out_ch],
                                              ((self.convolution_shape[1] // 2, self.convolution_shape[1] // 2),
                                               (self.convolution_shape[2] // 2,
                                                self.convolution_shape[2] // 2)), mode='constant'))
                    if self.convolution_shape[2] % 2 == 0:
                        ch_conv_out[out_ch] = ch_conv_out[out_ch][:, :-1]
                    if self.convolution_shape[1] % 2 == 0:
                        ch_conv_out[out_ch] = ch_conv_out[out_ch][:-1, :]

                conv_plane = np.stack(ch_conv_out, axis=0)
                conv_plane.tolist()
                conv_plane_out.append(conv_plane)
            padded_input = np.stack(conv_plane_out, axis=0)

            # [CORRELATION operation] for the weight gradient there's no flipping, so we again use the correlation.
            # loop over batches
            for batch in range(error_tensor.shape[0]):
                # loop over different kernels (output channels)
                for out_ch in range(error_tensor.shape[1]):

                    # STRIDE implementation (up-sampling)
                    temp = signal.resample(error_tensor[batch, out_ch],
                                           error_tensor[batch, out_ch].shape[0] * self.stride_shape[0], axis=0)
                    temp = signal.resample(temp, error_tensor[batch, out_ch].shape[1] * self.stride_shape[1],
                                           axis=1)
                    # slice it to match the correct shape if the last step of up-sampling was not full
                    temp = temp[:self.input_tensor.shape[2], :self.input_tensor.shape[3]]
                    # we need zero-interpolation, so we put zero for interpolated values
                    if self.stride_shape[1] > 1:
                        for i, row in enumerate(temp):
                            for ii, element in enumerate(row):
                                if ii % self.stride_shape[1] != 0:
                                    row[ii] = 0
                    if self.stride_shape[0] > 1:
                        for i, row in enumerate(temp):
                            for ii, element in enumerate(row):
                                if i % self.stride_shape[0] != 0:
                                    row[ii] = 0

                    # loop over input channels
                    for in_ch in range(self.input_tensor.shape[1]):
                        temp_gradient_weights[batch, out_ch, in_ch] = signal.correlate(padded_input[batch, in_ch],
                                                                                       temp, mode='valid')
            # we have to sum over the batches.
            self.gradient_weights = temp_gradient_weights.sum(axis=0)



        ############################### Bias gradients ###########################
        self.gradient_bias = np.sum(self.error_tensor, axis=(0, 2, 3))

        ############################## Update weights ###########################
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        if self._bias_optimizer:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        if gradient_input.shape[-1] == 1:
            gradient_input = np.reshape(gradient_input, gradient_input.shape[:-1])

        return gradient_input


    def initialize(self, weights_initializer, bias_initializer):
        self.bias = bias_initializer.initialize(self.bias.shape, 1, 1)
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape),
                                                      np.prod(self.convolution_shape[1:]) * self.num_kernels)
