import theano.tensor as T

from collections import OrderedDict
from smartpy.misc.utils import sharedX

from smartpy.update_rules import UpdateRule


class ADAGRAD(UpdateRule):
    """ ADAGRAD(lr, eps=1e-6) """
    __hyperparams__ = {'lr': float, 'eps': float}
    __optional__ = ['eps']

    def __init__(self, lr, eps=1e-6):
        """
        Implements the ADAGRAD learning rule.

        Parameters
        ----------
        lr: float
            learning rate
        eps: float
            epsilon needed to avoid division by zero.

        Reference
        ---------
        Duchi, J., Hazan, E., & Singer, Y. (2010).
        Adaptive subgradient methods for online learning and stochastic optimization.
        Journal of Machine Learning
        """
        super(ADAGRAD, self).__init__()
        self.lr = lr
        self.eps = eps

    def apply(self, gradients):
        """
        Produces new gradients

        Parameters
        ----------
        gradients : dict
            gradients (values) of some loss function w.r.t. params (keys)
        """
        updates = OrderedDict()
        new_gradients = OrderedDict()

        for param, gparam in gradients.items():
            # sum_squared_grad := \sum g_t^2
            sum_squared_grad = sharedX(param.get_value() * 0., name='sum_squared_grad_' + param.name)
            self.parameters[sum_squared_grad.name] = sum_squared_grad

            # Accumulate gradient
            new_sum_squared_grad = sum_squared_grad + T.sqr(gparam)

            # Compute update
            root_sum_squared = T.sqrt(new_sum_squared_grad + self.eps)

            # Apply update
            updates[sum_squared_grad] = new_sum_squared_grad
            new_gradients[param] = (self.lr/root_sum_squared) * gparam

        return new_gradients, updates
