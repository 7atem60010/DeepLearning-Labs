import numpy as np


class Constant:
    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in, fan_out):
        out = np.ones(weights_shape) * self.value
        print(out.shape, 'constant')
        return out


class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        out = np.random.uniform(0, 1, weights_shape)
        print(out.shape, 'uniform')
        return out


class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt((2 / fan_in))
        out = np.random.normal(0, sigma, weights_shape)
        print(out.shape, 'he')
        return out


class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt((2 / (fan_in + fan_out)))
        out = np.random.normal(0, sigma, weights_shape)
        print(out.shape, 'xavier')
        return out
