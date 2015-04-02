#from smartpy.optimizers.optimizer import Optimizer
from smartpy.optimizers.sgd import SGD

OPTIMIZERS = ['sgd']


def factory(model, dataset, hyperparams):
    """ Creates a `Optimizer` instance from `hyperparams` """

    if "optimizer_type" not in hyperparams:
