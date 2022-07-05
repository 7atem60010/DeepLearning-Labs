import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        masked = np.sum(prediction_tensor * label_tensor, axis=1)
        loss = - np.sum(np.log(masked + np.finfo(float).eps))
        return loss

    def backward(self, label_tensor):
        print(label_tensor)
        print("x"  , self.prediction_tensor)
        return np.divide(-1 * label_tensor , (self.prediction_tensor + np.finfo(float).eps))