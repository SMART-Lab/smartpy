import theano.tensor as T

from collections import OrderedDict
from smartpy.misc.utils import sharedX
from smartpy.learning_rates import LearningRate


class ADAGRAD(LearningRate):
    def __init__(self, lr, epsilon=1e-6):
        """
        Implements the ADAGRAD learning rule.

        Parameters
        ----------
        lr: float
            learning rate
        epsilon: float
            epsilon needed to avoid division by zero.

        Reference
        ---------
        Duchi, J., Hazan, E., & Singer, Y. (2010).
        Adaptive subgradient methods for online learning and stochastic optimization.
        Journal of Machine Learning
        """
        LearningRate.__init__(self, lr)

        self.epsilon = epsilon
        self.parameters = []

    def __call__(self, grads):
        updates = OrderedDict()
        learning_rates = OrderedDict()

        params_names = map(lambda p: p.name, self.parameters)
        for param in grads.keys():
            # sum_squared_grad := \sum g_t^2
            sum_squared_grad = sharedX(param.get_value() * 0.)

            if param.name is not None:
                sum_squared_grad.name = 'sum_squared_grad_' + param.name

            # Check if param is already there before adding
            if sum_squared_grad.name not in params_names:
                self.parameters.append(sum_squared_grad)
            else:
                sum_squared_grad = self.parameters[params_names.index(sum_squared_grad.name)]

            # Accumulate gradient
            new_sum_squared_grad = sum_squared_grad + T.sqr(grads[param])

            # Compute update
            root_sum_squared = T.sqrt(new_sum_squared_grad + self.epsilon)

            # Apply update
            updates[sum_squared_grad] = new_sum_squared_grad
            learning_rates[param] = self.base_lr / root_sum_squared

        return learning_rates, updates
