import numpy as np

import theano.tensor as T

from collections import OrderedDict
from smartpy.misc.utils import sharedX

from smartpy.update_rules import UpdateRule


class RMSProp(UpdateRule):
    """ RMSProp(lr, dc=0.95, eps=1e-6) """
    __hyperparams__ = {'lr': float, 'dc': float, 'eps': float}
    __optional__ = ['dc', 'eps']

    def __init__(self, lr, dc=0.95, eps=1e-6):
        """
        Implements the RMSProp learning rule.

        Parameters
        ----------
        lr: float
            learning rate
        dc: float
            decay rate (related to the window of the moving average)
        eps: float
            epsilon needed to avoid division by zero.

        Reference
        ---------
        Tieleman, T. and Hinton, G. (2012) - Lecture 6.5 - rmsprop
        COURSERA: Neural Networks for Machine Learning
        """
        super(RMSProp, self).__init__()

        assert dc >= 0.
        assert dc < 1.

        self.dc = dc
        self.eps = eps

    def apply(self, gradients):
        updates = OrderedDict()
        new_gradients = OrderedDict()

        for param, gparam in gradients.items():
            # mean_squared_grad := \sum g_t^2
            mean_squared_grad = sharedX(np.zeros_like(param.get_value()), name='mean_squared_grad_' + param.name)
            self.parameters[mean_squared_grad.name] = mean_squared_grad

            # Accumulate gradient
            #new_mean_squared_grad = T.cast(self.dc*mean_squared_grad + (1-self.dc)*T.sqr(gparam), dtype=theano.config.floatX)
            new_mean_squared_grad = self.dc*mean_squared_grad + (1-self.dc)*gparam**2

            # Compute update
            root_mean_squared = T.sqrt(new_mean_squared_grad + self.eps)

            # Apply update
            updates[mean_squared_grad] = new_mean_squared_grad
            new_gradients[param] = self.base_lr/root_mean_squared * gparam

        return new_gradients, updates
