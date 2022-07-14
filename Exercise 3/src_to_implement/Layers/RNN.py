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

        self.hidden_tensor = []

        for t, X in enumerate(self.input_tensor):
            X = np.reshape(X, (1, X.shape[0]))
            self.hidden_tensor.append(self.hidden_state)
            HX = np.concatenate((self.hidden_state, X), axis=1)
            HX_out = self.FC_mid.forward(HX)
            self.hidden_state = self.sigmoid.forward(HX_out)

            HY_out = self.FC_out.forward(self.hidden_state)
            y = self.sigmoid.forward(HY_out)
            self.output_tensor[t] = y

        return self.output_tensor

    @property
    def weights(self):
        return self.FC_mid.weights
    @weights.setter
    def weights(self, value):
        self.FC_mid.weights = value






