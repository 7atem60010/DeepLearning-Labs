import numpy
import numpy as np
from Layers import Sigmoid, TanH
from Layers import Initializers
from Layers import FullyConnected

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        print(input_size, hidden_size, output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memorize = False
        self.trainable = True
        self.hidden_state = np.zeros((1, hidden_size))
        self.bias_h = None
        self.bias_y = None
        self.weights_h = None
        self.weights_y = None
        self.sigmoid = Sigmoid.Sigmoid()
        self.tanh = TanH.TanH()
        self.FC_mid = FullyConnected.FullyConnected(input_size=input_size + hidden_size,
                                                    output_size=hidden_size)
        self.FC_out = FullyConnected.FullyConnected(input_size=hidden_size,
                                                    output_size=output_size)

    def initialize(self, weight_initializer, bias_initializer):
        self.FC_mid.initialize(weight_initializer, bias_initializer)
        self.FC_out.initialize(weight_initializer, bias_initializer)


    def forward(self, input):
        self.input_tensor = input
        self.output_tensor = np.zeros((self.input_tensor.shape[0], self.output_size))

        if self.memorize == False:
            self.hidden_state = np.zeros((1, self.hidden_size))

        self.hidden_tensor = [self.hidden_state]
        self.sigmoid_out = []
        self.tanh_out = []
        self.o = []
        self.u = []


        for t, X in enumerate(self.input_tensor):
            X = np.reshape(X, (1, X.shape[0]))
            self.hidden_tensor.append(self.hidden_state)
            hx = np.concatenate((self.hidden_state, X), axis=1)
            u = self.FC_mid.forward(hx)
            self.u.append(u)
            self.hidden_state = self.tanh.forward(u)
            self.tanh_out.append(self.hidden_state)

            o = self.FC_out.forward(self.hidden_state)
            self.o.append(o)
            y = self.sigmoid.forward(o)
            self.output_tensor[t] = y



        return self.output_tensor

    def backward(self, error_tensor):
        T = error_tensor.shape[0]
        self.grad_input = np.zeros((T, self.input_size))
        self.grad_hidden = np.zeros((T+1, self.hidden_size))
        self.grad_weights_out = 0
        self.grad_weights_mid = 0

        self.output_grad = []
        for t in reversed(range(len(error_tensor))):
            # output sigmoid
            self.sigmoid.fx = self.output_tensor[t]
            d_sigmoid = self.sigmoid.backward(error_tensor[t]).reshape((1,-1))
            d_o = d_sigmoid * error_tensor[t]
            h = self.hidden_tensor[t]
            d_wieghts_out = d_o * h.T
            d_bias_out = d_o


            #back to hidden
            self.FC_out.input_tensor = self.hidden_tensor[t]
            grad_out_in = self.FC_out.backward(d_sigmoid)

            self.grad_weights_out += self.FC_out.gradient_weights

            self.tanh.fx = h
            d_tanh = self.tanh.backward(self.u[t])
            w_hh = self.FC_mid.weights[:self.hidden_size]
            w_hy = self.FC_out.weights[1:]
            d_h = np.matmul(d_o, w_hy.T)

            d_w_hh = numpy.matmul(d_h.T, self.hidden_tensor[t-1])
            xt = self.input_tensor[t].reshape((1,-1))
            d_w_xh = np.matmul(d_h.T, xt)
            d_b_h = d_h * d_tanh

            self.output_grad.append(d_w_xh)
            # self.

        self.output_grad = np.array(self.output_grad)
        return self.output_grad


    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def weights(self):
        return self.FC_mid.weights
    @weights.setter
    def weights(self, value):
        self.FC_mid.weights = value






