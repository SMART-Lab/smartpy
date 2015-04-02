from smartpy.regularizations import Regularization


class L2Regularization(Regularization):
    def __init__(self, decay):
        Regularization.__init__(self, decay)

    def __call__(self, param):
        return 2*self.decay * (param**2).sum()
