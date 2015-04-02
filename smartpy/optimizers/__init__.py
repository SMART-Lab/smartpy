#from smartpy.optimizers.optimizer import Optimizer
from smartpy.optimizers.sgd import SGD

from smartpy import learning_rates
from smartpy import momentums

OPTIMIZERS = ['sgd']


def factory(model, dataset, hyperparams):
    """ Creates a `Optimizer` instance from `hyperparams` """

    learning_rate_method = learning_rates.factory(**hyperparams)
    momentum_method = momentums.factory(**hyperparams)

    if "optimizer_type" not in hyperparams:
        raise ValueError("Hyperparameter 'optimizer_type' is mandatory ({}).".format(", ".join(OPTIMIZERS)))
    elif hyperparams["optimizer_type"] == "sgd":
        optimizer = SGD(model, dataset, hyperparams.get("batch_size", 1), learning_rate_method, momentum_method)
    else:
        raise ValueError("The following optimizer method is not implemented: " + hyperparams["optimizer_type"])

    return optimizer
