from smartpy.models.model import Model

from smartpy import regularizations


def factory(model_name, input_size, hyperparams):
    regularization_method = regularizations.factory(**hyperparams)

    #Build model
    if model_name.lower() == "NADE".lower():
        from smartpy.models.nade import NADE
        model = NADE(input_size=input_size,
                     hidden_size=hyperparams["size"],
                     hidden_activation=hyperparams["hidden_activation"],
                     #regularization=regularization_method
                     )
    elif model_name.lower() == "NestedNADE".lower():
        from smartpy.models.nested_nade import NestedNADE
        model = NestedNADE(trained_nade=hyperparams["nade"],
                           hidden_size=hyperparams["size"],
                           hidden_activation=hyperparams["hidden_activation"],
                           gamma=hyperparams["gamma"]
                           )

    return model
