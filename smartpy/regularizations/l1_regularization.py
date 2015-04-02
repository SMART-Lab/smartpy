from smartpy.regularizations import Regularization


class L1Regularization(Regularization):
    def __init__(self, decay):
        Regularization.__init__(self, decay)

    def __call__(self, param):
        return self.decay * abs(param).sum()
