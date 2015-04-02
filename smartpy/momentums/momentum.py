import numpy as np
import theano

from collections import OrderedDict


class Momentum(object):
    def __init__(self, momentum_ratio):
        self.momentum_ratio = momentum_ratio

    def __call__(self, gradients):
        momentums = {}
        updates = OrderedDict()

        for param, gparam in gradients.items():
            # Initialisation
            momentum = theano.shared(np.zeros_like(param.get_value()))
            # Next momentum
            updates[momentum] = gparam + self.momentum_ratio * momentum
            # Current momentum
            momentums[param] = gparam + momentum

        return momentums, updates
