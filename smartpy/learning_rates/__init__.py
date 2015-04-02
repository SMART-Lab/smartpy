from smartpy.learning_rates.learning_rate import LearningRate
from smartpy.learning_rates.constant_learning_rate import ConstantLearningRate
from smartpy.learning_rates.decreasing_learning_rate import DecreasingLearningRate
from smartpy.learning_rates.adagrad import ADAGRAD
from smartpy.learning_rates.adadelta import ADADELTA
from smartpy.learning_rates.rmsprop import RMSProp
#from smartpy.learning_rates.adam import Adam

LEARNING_RATE_METHODS = ["constant", "ADAGRAD", "ADADELTA", "decreasing", "RMSProp"]


def factory(**hyperparams):
    """ Create a `LearningRate` instance from `hyperparams`. """

    if "lr_type" not in hyperparams:
        raise ValueError("Hyperparameter 'lr_type' is mandatory ({}).".format(",".join(LEARNING_RATE_METHODS)))
    elif hyperparams["lr_type"] == "constant":
        lr_method = ConstantLearningRate(lr=hyperparams["lr"])
    elif hyperparams["lr_type"] == "ADAGRAD":
        lr_method = ADAGRAD(lr=hyperparams["lr"])
    elif hyperparams["lr_type"] == "ADADELTA":
        lr_method = ADADELTA(lr=hyperparams["lr"], dc=hyperparams["lr_dc"], epsilon=hyperparams["lr_eps"])
    elif hyperparams["lr_type"] == "RMSProp":
        lr_method = RMSProp(lr=hyperparams["lr"], decay=hyperparams["lr_dc"], epsilon=hyperparams["lr_eps"])
    else:
        raise ValueError("The following learning rate method is not implemented: " + hyperparams["lr_type"])

    return lr_method
