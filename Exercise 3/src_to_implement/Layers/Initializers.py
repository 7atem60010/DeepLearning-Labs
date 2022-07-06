# import numpy as np
#
#
# class Constant:
#     def __init__(self, value=0.1):
#         self.value = value
#
#     def initialize(self, weights_shape, fan_in, fan_out):
#         out = np.ones(weights_shape) * self.value
#         print(out.shape, 'constant')
#         return out
#
#
# class UniformRandom:
#     def __init__(self):
#         pass
#
#     def initialize(self, weights_shape, fan_in, fan_out):
#         out = np.random.uniform(0, 1, weights_shape)
#         print(out.shape, 'uniform')
#         return out
#
#
# class He:
#     def __init__(self):
#         pass
#
#     def initialize(self, weights_shape, fan_in, fan_out):
#         sigma = np.sqrt((2 / fan_in))
#         out = np.random.normal(0, sigma, weights_shape)
#         print(out.shape, 'he')
#         return out
#
#
# class Xavier:
#     def __init__(self):
#         pass
#
#     def initialize(self, weights_shape, fan_in, fan_out):
#         sigma = np.sqrt((2 / (fan_in + fan_out)))
#         out = np.random.normal(0, sigma, weights_shape)
#         print(out.shape, 'xavier')
#         return out
import numpy as np
from Layers.Base import Base


class Constant(Base):
    '''
    self.shape = (output_size, input_size)
    '''
    def __init__(self, weight_initialization=0.1):
        '''
        :param weight_initialization: the constant value used for weight initialization.
        '''
        super().__init__()

        self.weight_initialization = weight_initialization

    def initialize(self, weights_shape, fan_in, fan_out):
        ''''
        returns an initialized tensor of the desired shape.
        '''
        return (np.zeros((fan_in, fan_out)) + self.weight_initialization)



class UniformRandom(Base):
    def __init__(self):
        super().__init__()

        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        ''''
        returns an initialized tensor of the desired shape.
        '''
        return np.random.rand(fan_in, fan_out)



class Xavier(Base):
    def __init__(self):
        super().__init__()

        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        ''''
        returns an initialized tensor of the desired shape.
        '''
        return np.random.normal(0, (2 / (fan_out + fan_in))**(1/2), weights_shape)



class He(Base):
    def __init__(self):
        super().__init__()

        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        ''''
        returns an initialized tensor of the desired shape.
        '''
        return np.random.normal(0, (2 / fan_in)**(1/2), weights_shape)