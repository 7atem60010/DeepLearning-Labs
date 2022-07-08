import scipy.sparse as sparse
import numpy as np
from Layers.Base import *
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(Base):
    def __init__(self, channels):
        self.channels = channels
        self.weights = np.ones((self.channels))
        self.bias = np.zeros((self.channels))
        self.trainable = True
        self.testing_phase = False

        self.input_tensor_hat =  None
        self.input_tensor =  None
        self.mean = 0
        self.var = 1
        self.prev_mean = 0
        self.prev_var = 1
        self.optimizer = None
        self.bias_optimizer = None


    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones((self.channels))
        self.bias = np.zeros((self.channels))

    def forward(self, input_tensor , alpha =  0.8):

        self.input_tensor = input_tensor
        # print("Input :" , input_tensor)

        if len(self.input_tensor.shape) == 2:
            input_tensor_temp = self.input_tensor

            self.mean = np.mean(input_tensor_temp , axis=0)
            self.var = np.var(input_tensor_temp , axis=0)

            # print("MEAN :" , self.mean)
            # print("VAR :" , self.var)

            # Testing phase
            if self.testing_phase == True:
                self.input_tensor_hat = (input_tensor_temp - self.mean_hat) / np.sqrt(self.var_hat + np.finfo(float).eps)

            # Training phase
            else:
                # mini-batch mean , var
                new_mean = np.mean(input_tensor_temp, axis=0)
                new_variance = np.var(input_tensor_temp, axis=0)

                # Moving avg mean , var
                self.mean_hat = alpha * self.mean + (1 - alpha) * self.mean
                self.var_hat = alpha * self.var + (1 - alpha) * self.var

                self.mean = new_mean
                self.var = new_variance

                self.input_tensor_hat  = (input_tensor_temp - self.mean)/np.sqrt(self.var + np.finfo(float).eps)
            # Output calculations
            output = self.weights * self.input_tensor_hat + self.bias


        if len(self.input_tensor.shape) == 4:
            # print(self.input_tensor.shape)
            input_tensor_temp = self.input_tensor.reshape(self.input_tensor.shape[0] , self.input_tensor.shape[1] , self.input_tensor.shape[2]*self.input_tensor.shape[3])
            input_tensor_temp = np.transpose(input_tensor_temp ,  (0, 2 ,1))
            input_tensor_temp = input_tensor_temp.reshape(input_tensor_temp.shape[0] * input_tensor_temp.shape[1] , input_tensor_temp.shape[2])

            #print(input_tensor_temp.shape)
            self.mean = np.mean(input_tensor_temp, axis=0)
            self.var = np.var(input_tensor_temp, axis=0)

            if self.testing_phase == True:
                self.input_tensor_hat = (input_tensor_temp - self.mean_hat) / np.sqrt(self.var_hat + np.finfo(float).eps)
            else:
                # mini-batch mean , var
                new_mean = np.mean(input_tensor_temp, axis=0)
                new_variance = np.var(input_tensor_temp, axis=0)

                # Moving avg mean , var
                self.mean_hat = alpha * self.mean + (1 - alpha) * self.mean
                self.var_hat = alpha * self.var + (1 - alpha) * self.var

                self.mean = new_mean
                self.var = new_variance

                self.input_tensor_hat = (input_tensor_temp - self.mean) / np.sqrt(self.var + np.finfo(float).eps)

            output = self.weights * self.input_tensor_hat + self.bias
            output = self.reformat(output)

        return output

    def backward(self , error_tensor):
        # print(error_tensor.shape)
        # print(self.input_tensor_hat.shape)

        if len(self.input_tensor.shape) == 2:

            error_tensor_prev_layer =  compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.var)
            self.weights_grad = np.sum(error_tensor * self.input_tensor_hat, axis= 0 )
            self.bias_grad = np.sum(error_tensor, axis= 0 )

        if len(self.input_tensor.shape) == 4:

            error_tensor_prev_layer = compute_bn_gradients(self.reformat(error_tensor), self.reformat(self.input_tensor), self.weights, self.mean,
                                                           self.var)
            error_tensor_prev_layer = self.reformat(error_tensor_prev_layer)

            self.weights_grad = np.sum(error_tensor * self.input_tensor_hat, axis=0)
            self.bias_grad = np.sum(error_tensor, axis=0)

        '''Update with optimizers'''
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        if self.bias_optimizer:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)


        return error_tensor_prev_layer

    def reformat(self , tensor):
        #print(tensor[0])
        if len(tensor.shape)  == 4:
            b , c , x, y = tensor.shape
            tensor_out = np.transpose(tensor, (0, 2, 3, 1)).reshape(b*x*y ,c)
            #print(tensor_out)
            return tensor_out
        elif len(tensor.shape)  == 2:
            b , c , x, y = self.input_tensor.shape
            tensor_out = np.reshape(tensor, (b , x , y , c ))
            tensor_out = np.transpose(tensor_out , (0,3,1 ,2))
            return tensor_out
