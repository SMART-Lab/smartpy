import numpy as np

from collections import OrderedDict
from smartpy.misc.utils import sharedX

from smartpy.update_rules import UpdateRule


class LearningRate(UpdateRule):
    """ LearningRate(self, lr, dc=0.) """
    __hyperparams__ = {'lr': float, 'dc': float}
    __optional__ = ['dc']

    def __init__(self, lr, dc=0.):
        """
        Implements a learning rate update rule.

        Parameters
        ----------
        lr: float
            learning rate
        dc: float
            decreasing constant (decay)
        """
        super(LearningRate, self).__init__()
        assert dc <= 1.
        assert dc >= 0.
        self.lr = lr
        self.dc = dc

    def apply(self, gradients):
        updates = OrderedDict()
        new_gradients = OrderedDict()

        for param, gparam in gradients.items():
            lr = sharedX(self.lr * np.ones_like(param.get_value()), name='lr_' + param.name)
            self.parameters[lr.name] = lr

            if self.dc != 0.:
                # Decrease the learning rate by a factor of `dc` after each update.
                updates[lr] = self.dc * lr

            new_gradients[param] = lr * gparam

        return new_gradients, updates
