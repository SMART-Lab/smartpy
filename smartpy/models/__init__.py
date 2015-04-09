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

        hidden_size = hyperparams["size"]
        if hidden_size is None:
            hidden_size = trained_nade.hyperparams["hidden_size"]

        hidden_activation = hyperparams["hidden_activation"]
        if hidden_activation is None:
            hidden_activation = trained_nade.hyperparams["hidden_activation"]

        model = NestedNADE(input_size=input_size,
                           hidden_size=hidden_size,
                           trained_nade=trained_nade,
                           hidden_activation=hidden_activation,
                           gamma=hyperparams["gamma"]
                           )

    return model
