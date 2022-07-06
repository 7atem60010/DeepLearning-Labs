import numpy as np
import base_optimizer


class Sgd(base_optimizer):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer != None:
            reg =  self.reregularizer.calculate_gradient(weight_tensor)
            return weight_tensor - self.learning_rate*reg - self.learning_rate * gradient_tensor
        else:
            return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum(base_optimizer):
    def __init__(self, learning_rate: float , momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.prev_v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        v = self.momentum_rate * self.prev_v - self.learning_rate*gradient_tensor
        self.prev_v = v
        return weight_tensor + v


class Adam(base_optimizer):
    def __init__(self, learning_rate: float , mu , roh):
        self.learning_rate = learning_rate
        self.mu = mu
        self.roh = roh
        self.prev_v = 0
        self.prev_r = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        v = self.mu * self.prev_v + (1-self.mu)*gradient_tensor
        r = self.roh * self.prev_r + (1-self.roh)*gradient_tensor*gradient_tensor
        self.prev_v = v
        self.prev_r = r

        v_hat = v / (1 - self.mu**self.k)
        r_hat = r / (1 - self.roh**self.k)
        self.k +=1

        weight_tensor = weight_tensor - self.learning_rate*(v_hat / np.sqrt(r_hat))
        return weight_tensor

