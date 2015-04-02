import numpy as np

from smartpy import learning_rates
from smartpy import regularizations
from smartpy import momentums

from smartpy.misc import weight_initializer


def factory(model_name, input_size, hyperparams):
    learning_rate_method = learning_rates.factory(**hyperparams)
    regularization_method = regularizations.factory(**hyperparams)
    momentum_method = momentums.factory(**hyperparams)
    weights_initialization_method = weight_initializer.factory(**hyperparams)

    #rng = np.random.RandomState(hyperparams.get("seed", 1234))

    #Build model
    if model_name == "nade":
        from smartpy.models.nade import NADE
        model = NADE(input_size=input_size,
                     hidden_size=hyperparams["size"],
                     learning_rate=learning_rate_method,
                     regularization=regularization_method,
                     momentum=momentum_method,
                     weights_initialization=weights_initialization_method
                     )

    return model
