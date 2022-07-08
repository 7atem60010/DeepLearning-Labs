
class Base:
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.weights = None
        self.testing_phase = False


