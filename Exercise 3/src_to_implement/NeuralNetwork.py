from Layers.Base import Base
import numpy as np
import copy


class NeuralNetwork(Base):
    def __init__(self, optimizer , weights_initializer , bias_initializer):
        self.optimizer =  optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        self.loss = []
        self.layers =  []
        self.data_layer = []
        self.loss_layer =  []


    def append_layer(self , layer):
        if layer.trainable == True:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
            self.layers.append(layer)
        else :
            self.layers.append(layer)

    def forward(self):
        self.input_tensor ,  self.label_tensor = self.data_layer.next()

        out_layer = self.layers[0].forward(self.input_tensor)

        for layer in self.layers[1:]:
            out_layer = layer.forward(out_layer)

        out_layer = self.loss_layer.forward(out_layer , self.label_tensor[0])
        return out_layer

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def train(self , num_iterations):
        for i in range(num_iterations):
            self.loss.append(self.forward())
            if (i + 1) % 200 == 0:
                print("training iteration", str(i + 1) + ":", 'loss =', self.loss[i])
            self.backward()

    def test(self , input_tensor):
        out_layer = self.layers[0].forward(input_tensor)

        for layer in self.layers[1:]:
            out_layer = layer.forward(out_layer)

        return out_layer

    @property
    def phase(self):
        return self.phase

    @phase.setter
    def phase(self, phase):
        self.phase = phase