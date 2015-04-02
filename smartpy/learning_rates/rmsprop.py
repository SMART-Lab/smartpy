import theano
import theano.tensor as T

from collections import OrderedDict
from smartpy.misc.util import sharedX
from smartpy.learning_rates import LearningRate


class RMSProp(LearningRate):
    def __init__(self, lr, decay=0.95, epsilon=1e-6):
        """
        Implements the RMSProp learning rule.

        Parameters
        ----------
        lr: float
            learning rate
        decay: float
            decay rate (related to the window of the moving average)
        epsilon: float
            epsilon needed to avoid division by zero.

        Reference
        ---------
        Tieleman, T. and Hinton, G. (2012) - Lecture 6.5 - rmsprop
        COURSERA: Neural Networks for Machine Learning
        """
        LearningRate.__init__(self, lr)

        assert decay >= 0.
        assert decay < 1.
        self.decay = decay
        self.epsilon = epsilon
        self.parameters = []

    def __call__(self, grads):
        updates = OrderedDict()
        learning_rates = OrderedDict()

        for param in grads.keys():
            # mean_squared_grad := \sum g_t^2
            mean_squared_grad = sharedX(param.get_value() * 0.)
            self.parameters.append(mean_squared_grad)

            if param.name is not None:
                mean_squared_grad.name = 'mean_squared_grad_' + param.name

            # Accumulate gradient
            new_mean_squared_grad = T.cast(self.decay*mean_squared_grad + (1-self.decay)*T.sqr(grads[param]), dtype=theano.config.floatX)

            # Compute update
            root_mean_squared = T.sqrt(new_mean_squared_grad + self.epsilon)

            # Apply update
            updates[mean_squared_grad] = new_mean_squared_grad
            learning_rates[param] = self.base_lr / root_mean_squared

        return learning_rates, updates
