import numpy as np
from numpy.linalg import norm

class L1_Regularizer:
    def __init__(self , alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)

    def norm(self , weights):
        print(sum(sum(abs(weights))))
        reg = self.alpha * sum(sum(abs(weights)))
        return reg


class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * weights

    def norm(self , weights):
        print(weights)
        print(self.alpha)
        reg = self.alpha * sum(sum(np.power(weights,2)))
        return reg
