import numpy as np

from collections import OrderedDict
from smartpy.misc.utils import sharedX

from smartpy.update_rules import UpdateRule


class Momentum(UpdateRule):
    """ Momentum(ratio) """
    __hyperparams__ = {'ratio': float}

    def __init__(self, ratio):
        """
        Implements a momentum update rule.

        Parameters
        ----------
        ratio: float
            decay of the importance of past gradients.
        """
        super(Momentum, self).__init__()
        self.ratio = ratio

    def apply(self, gradients):
        updates = OrderedDict()
        new_gradients = OrderedDict()

        for param, gparam in gradients.items():
            momentum = sharedX(np.zeros_like(param.get_value()), name='momentum_' + param.name)
            self.parameters[momentum.name] = momentum
            updates[momentum] = gparam + self.ratio * momentum
            new_gradients[param] = gparam + momentum

        return new_gradients, updates
