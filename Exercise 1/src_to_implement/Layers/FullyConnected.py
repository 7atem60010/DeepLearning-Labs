import numpy as np
from Optimization import Optimizers
from Layers.Base import Base



class FullyConnected(Base):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._optimizer = None
        self.trainable = True
        self.weights = np.random.uniform(size=(input_size+1, output_size))
        
        self.input_tensor =  None
        self.gradient_weights = None
        self.error = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_tensor = np.concatenate((np.ones((self.input_tensor.shape[0],1)),self.input_tensor),axis=1)
        print(self.input_tensor.shape)
        self.output_tensor = np.matmul(self.input_tensor, self.weights) 
        return self.output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, Optimizer):
        self._optimizer = Optimizer
    
    def gradiant_weights():
        return self.gradiant_tensor

  
    def backward(self, error_tensor):
        # The input is error tensor , error tensor is 

        # Get error tensor for prevouis layer
        error_tensor_prev_layer = np.matmul(error_tensor , self.weights[:-1,:].T )

        # Update weights
        if self._optimizer != None:
            #print(self.input_tensor.shape , error_tensor.shape)
            self.gradiant_tensor = np.matmul( self.input_tensor.T  , error_tensor)
            self.weights= self._optimizer.calculate_update( self.weights , self.gradiant_tensor )


        return error_tensor_prev_layer


