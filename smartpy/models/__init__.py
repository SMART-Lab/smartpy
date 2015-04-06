import numpy as np

from smartpy.models.model import Model

from smartpy import regularizations

from smartpy.misc import weights_initializer


def factory(model_name, input_size, hyperparams):
    regularization_method = regularizations.factory(**hyperparams)
    weights_initialization_method = weights_initializer.factory(**hyperparams)

    #rng = np.random.RandomState(hyperparams.get("seed", 1234))

    #Build model
    if model_name.lower() == "nade":
        from smartpy.models.nade import NADE
        model = NADE(input_size=input_size,
                     hidden_size=hyperparams["size"],
                     #regularization=regularization_method,
                     weights_initialization=weights_initialization_method
                     )

    return model
