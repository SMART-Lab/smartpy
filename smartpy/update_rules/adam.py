import numpy as np

import theano
import theano.tensor as T

from collections import OrderedDict
from smartpy.misc.utils import sharedX
from smartpy.update_rules import UpdateRule


class AdamV1(UpdateRule):
    """ AdamV1(lr=0.0002, b1=0.1, b2=0.001, eps=1e-8) """
    __hyperparams__ = {'lr': float, 'b1': float, 'b2': float, 'eps': float}
    __optional__ = ['lr', 'b1', 'b2', 'eps']

    def __init__(self, lr=0.0002, b1=0.1, b2=0.001, eps=1e-8):
        super(AdamV1, self).__init__()

        self.lr = lr
        self.b1 = b1
        self.b2 = b2
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

        i = sharedX(np.float32(0.), name='i_AdamV1')
        self.parameters[i.name] = i
        i_t = i + 1.
        fix1 = 1. - (1. - self.b1) ** i_t
        fix2 = 1. - (1. - self.b2) ** i_t
        lr_t = self.lr * (T.sqrt(fix2) / fix1)

        for param, gparam in gradients.items():
            m = sharedX(param.get_value() * 0., name='m_' + param.name)
            self.parameters[m.name] = m

            v = sharedX(param.get_value() * 0., name='v_' + param.name)
            self.parameters[v.name] = v

            m_t = (self.b1 * gparam) + ((1. - self.b1) * m)
            v_t = (self.b2 * T.sqr(gparam)) + ((1. - self.b2) * v)
            g_t = m_t / (T.sqrt(v_t) + self.eps)

            updates[m] = m_t
            updates[v] = v_t

            # Apply update
            new_gradients[param] = lr_t * g_t

        updates[i] = i_t

        return new_gradients, updates
