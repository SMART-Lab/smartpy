
from collections import OrderedDict
from smartpy.learning_rates import LearningRate


class DecreasingLearningRate(LearningRate):
    def __init__(self, lr, dc):
        """
        Implements a decreasing learning rule.

        Parameters
        ----------
        lr: float
            initial learning rate
        dc: float
            decreasing constant
        """
        LearningRate.__init__(self, lr)
        self.dc = dc

    def __call__(self, gradients):
        updates = OrderedDict()
        for param, gparam in gradients.items():
            updates[self.lr[param]] = self.lr[param] * self.dc

        return self.lr, updates
