import theano.tensor as T

from collections import OrderedDict
from smartpy.misc.utils import sharedX
from smartpy.learning_rates import LearningRate


class ADADELTA(LearningRate):
    def __init__(self, lr, decay=0.95, epsilon=None):
        """
        Implements the ADADELTA learning rule.

        Parameters
        ----------
        lr: float
            learning rate
        decay: float
            decay rate \rho in Algorithm 1 of the afore-mentioned paper.
        epsilon: float
            epsilon used as in Algorithm 1 of the afore-mentioned paper.

        Reference
        ---------
        Zeiler, M. (2012). ADADELTA: An Adaptive Learning Rate Method.
        arXiv Preprint arXiv:1212.5701
        """
        LearningRate.__init__(self, lr)

        assert decay >= 0.
        assert decay < 1.

        self.epsilon = epsilon if epsilon else lr
        self.decay = decay
        self.parameters = []

    def __call__(self, grads):
        updates = OrderedDict()
        learning_rates = OrderedDict()

        for param in grads.keys():
            # mean_squared_grad := E[g^2]_{t-1}
            mean_squared_grad = sharedX(param.get_value() * 0.)
            self.parameters.append(mean_squared_grad)
            # mean_squared_dx := E[(\Delta x)^2]_{t-1}
            mean_squared_dx = sharedX(param.get_value() * 0.)
            self.parameters.append(mean_squared_dx)

            if param.name is not None:
                mean_squared_grad.name = 'mean_squared_grad_' + param.name
                mean_squared_dx.name = 'mean_squared_dx_' + param.name

            # Accumulate gradient
            new_mean_squared_grad = self.decay * mean_squared_grad + (1 - self.decay) * T.sqr(grads[param])

            # Compute update
            rms_dx_tm1 = T.sqrt(mean_squared_dx + self.epsilon)
            rms_grad_t = T.sqrt(new_mean_squared_grad + self.epsilon)
            lr = rms_dx_tm1 / rms_grad_t
            delta_x_t = -lr * grads[param]

            # Accumulate updates
            new_mean_squared_dx = self.decay * mean_squared_dx + (1 - self.decay) * T.sqr(delta_x_t)

            # Apply update
            updates[mean_squared_grad] = new_mean_squared_grad
            updates[mean_squared_dx] = new_mean_squared_dx
            learning_rates[param] = lr

        return learning_rates, updates
