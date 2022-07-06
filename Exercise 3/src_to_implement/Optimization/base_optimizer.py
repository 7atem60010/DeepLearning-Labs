

class base_optimizer:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.regularizer = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        pass

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer
