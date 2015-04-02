from collections import OrderedDict
from smartpy.momentums.momentum import Momentum


class NoMomentum(Momentum):
    def __init__(self):
        Momentum.__init__(self, 0.0)

    def __call__(self, gradients):
        return gradients, OrderedDict()
