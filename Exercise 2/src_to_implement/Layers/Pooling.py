import numpy as np
from Optimization import Optimizers
from Layers import Initializers
from Layers.Base import Base
from scipy import signal
from math import ceil


class Pooling(Base):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = False
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # print(input_tensor.shape)
        output_layer = np.zeros((input_tensor.shape[0],input_tensor.shape[1] ,int((input_tensor.shape[2]-(self.pooling_shape[0]-1)-1)/self.stride_shape[0])+1 , int((input_tensor.shape[3]-(self.pooling_shape[1]-1)-1)/self.stride_shape[1] )+1))
        self.max_positions = np.zeros(input_tensor.shape)
        # print(output_layer.shape)
        for b in range(input_tensor.shape[0]):
            image = input_tensor[b, :]
            for c in range(image.shape[0]):
                for y in range(0,output_layer.shape[2]):
                    for x in range(0, output_layer.shape[3]):
                        #print(c ,y, x)
                        #print(y*self.stride_shape[0] ,(y+1)*self.stride_shape[0] )
                        #print(np.max(image[c , y*self.stride_shape[0]:(y+1)*self.stride_shape[0] , x*self.stride_shape[1]:(x+1)*self.stride_shape[1]]))
                        output_layer[b , c , y , x] = np.max(image[c , y*self.stride_shape[0]:(y+1)*self.stride_shape[0] , x*self.stride_shape[1]:(x+1)*self.stride_shape[1]])
                       # print(image[c , y*self.stride_shape[0]:(y+1)*self.stride_shape[0] , x*self.stride_shape[1]:(x+1)*self.stride_shape[1]])
                        array  = image[c , y*self.stride_shape[0]:(y+1)*self.stride_shape[0] , x*self.stride_shape[1]:(x+1)*self.stride_shape[1]]
                        max_position = np.unravel_index(np.argmax(array, axis=None), array.shape)
                      #  print(max_position)
                        #print(max_index_col)
                        self.max_positions[b ,c ,y*self.stride_shape[0] + max_position[0] , x*self.stride_shape[1]+max_position[1]] = 1
                        #print(self.max_positions)
                        #print(output_layer)
        #print(output_layer.shape)
        return output_layer

    def backward(self , error_tensor):
        #print(error_tensor.shape)
        print(error_tensor)
        #print(self.max_positions.shape)
        print(self.max_positions)
        error_prev = np.copy(self.max_positions)
        for b in range(self.max_positions.shape[0]):
            for c in range(self.max_positions.shape[1]):
                for y in range(0, error_tensor.shape[2]):
                    for x in range(0, error_tensor.shape[3]):
                        current_box = self.max_positions[b,c,y*self.stride_shape[0]:(y+1)*self.stride_shape[0], x*self.stride_shape[1]:(x+1)*self.stride_shape[1]]
                        #print(current_box)
                        error_update = error_tensor[b,c,y,x]*current_box
                        #print(error_update)
                        error_prev[b,c,y*self.stride_shape[0]:(y+1)*self.stride_shape[0], x*self.stride_shape[1]:(x+1)*self.stride_shape[1]] = error_update
        #print(error_prev)
        return error_prev