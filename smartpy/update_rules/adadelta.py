import theano.tensor as T

from collections import OrderedDict
from smartpy.misc.utils import sharedX

from smartpy.update_rules import UpdateRule


class ADADELTA(UpdateRule):
    """ ADADELTA(dc=0.95, eps=1e-6) """
    __hyperparams__ = {'dc': float, 'eps': float}
    __optional__ = ['dc', 'eps']

    def __init__(self, dc=0.95, eps=1e-6):
        """
        Implements the ADADELTA learning rule.

        Parameters
        ----------
        dc: float
            decay rate \rho in Algorithm 1 of the afore-mentioned paper.
        eps: float
            epsilon used as in Algorithm 1 of the afore-mentioned paper.

        Reference
        ---------
        Zeiler, M. (2012). ADADELTA: An Adaptive Learning Rate Method.
        arXiv Preprint arXiv:1212.5701
        """
        super(ADADELTA, self).__init__()

        assert dc >= 0.
        assert dc < 1.

        self.eps = eps
        self.dc = dc

    def apply(self, gradients):
        updates = OrderedDict()
        new_gradients = OrderedDict()

        for param, gparam in gradients.items():
            # mean_squared_grad := E[g^2]_{t-1}
            mean_squared_grad = sharedX(param.get_value() * 0., name='mean_squared_grad_' + param.name)
            self.parameters[mean_squared_grad.name] = mean_squared_grad

            # mean_squared_dx := E[(\Delta x)^2]_{t-1}
            mean_squared_dx = sharedX(param.get_value() * 0., name='mean_squared_dx_' + param.name)
            self.parameters[mean_squared_dx.name] = mean_squared_dx

            # Accumulate gradient
            new_mean_squared_grad = self.dc * mean_squared_grad + (1 - self.dc) * T.sqr(gparam)

            # Compute update
            rms_dx_tm1 = T.sqrt(mean_squared_dx + self.eps)
            rms_grad_t = T.sqrt(new_mean_squared_grad + self.eps)
            lr = rms_dx_tm1 / rms_grad_t
            delta_x_t = -lr * gparam

            # Accumulate updates
            new_mean_squared_dx = self.dc * mean_squared_dx + (1 - self.dc) * T.sqr(delta_x_t)

            # Apply update
            updates[mean_squared_grad] = new_mean_squared_grad
            updates[mean_squared_dx] = new_mean_squared_dx
            new_gradients[param] = lr * gparam

        return new_gradients, updates
