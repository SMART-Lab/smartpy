from smartpy.regularizations.regularization import Regularization
from smartpy.regularizations.no_regularization import NoRegularization
from smartpy.regularizations.l1_regularization import L1Regularization
from smartpy.regularizations.l2_regularization import L2Regularization

REGULARIZATION_METHODS = ["no", "L1", "L2"]


def factory(**hyperparams):
    """ Create a `Regularization` instance from `hyperparams`. """

    if "regularization" not in hyperparams or hyperparams.get("lambda", 0.0) == 0.0:
        regularization_method = NoRegularization()
    elif hyperparams.get("regularization") == "L1":
        regularization_method = L1Regularization(hyperparams["lambda"])
    elif hyperparams.get("regularization") == "L2":
        regularization_method = L2Regularization(hyperparams["lambda"])
    else:
        raise ValueError("The following regularization rate method is not implemented: " + hyperparams["regularization"])

    return regularization_method
