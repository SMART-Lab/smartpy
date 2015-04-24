from smartpy.regularizations import Regularization


class NoRegularization(Regularization):
    def __init__(self):
        Regularization.__init__(self, 0.0)

    def __call__(self, param):
        return 0.0
