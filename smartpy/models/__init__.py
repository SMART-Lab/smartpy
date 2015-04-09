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
        from smartpy.models.nade import NADE
        from smartpy.models.nested_nade import NestedNADE
        trained_nade = NADE.create(hyperparams["nade"])
        model = NestedNADE(input_size=input_size,
                           hidden_size=hyperparams["size"],
                           trained_nade=trained_nade,
                           hidden_activation=hyperparams["hidden_activation"],
                           gamma=hyperparams["gamma"]
                           )

    return model
