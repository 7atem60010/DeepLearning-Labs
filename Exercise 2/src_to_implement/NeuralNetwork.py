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
        self.input_tensor = self.data_layer.next()
        self.label_tensor = self.data_layer.next()

        print(self.layers)
        print(self.input_tensor[0])

        out_layer = self.layers[0].forward(self.input_tensor[0])
        for layer in self.layers[1:]:
            out_layer = layer.forward(out_layer)

