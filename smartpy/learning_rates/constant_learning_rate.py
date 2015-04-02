from collections import OrderedDict
from smartpy.learning_rates import LearningRate


class ConstantLearningRate(LearningRate):
    def __init__(self, lr):
        LearningRate.__init__(self, lr)

    def __call__(self, gradients):
        return self.lr, OrderedDict()
