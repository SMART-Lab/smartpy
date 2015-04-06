from smartpy.optimizers.optimizer import Optimizer
from smartpy.optimizers.sgd import SGD

OPTIMIZERS = ['sgd']


def factory(optimizer_type, loss, **kwargs):
    """ Creates a `Optimizer` instance that will optimize function `loss`. """

    if optimizer_type.upper() == "SGD":
        optimizer = SGD(loss, kwargs.get("batch_size", 1))
    else:
        raise ValueError("The following optimizer method is not implemented: " + optimizer_type)

    return optimizer
