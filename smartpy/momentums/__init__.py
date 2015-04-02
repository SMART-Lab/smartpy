from smartpy.momentums.momentum import Momentum
from smartpy.momentums.no_momentum import NoMomentum

MOMENTUM_METHODS = ["NoMomentum", "Momentum"]


def factory(**hyperparams):
    """ Create a `Momentum` instance from `hyperparams`. """

    if hyperparams.get("momentum_ratio", 0.0) == 0.0:
        momentum_method = NoMomentum()
    else:
        momentum_method = Momentum(hyperparams["momentum_ratio"])

    return momentum_method
