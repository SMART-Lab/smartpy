import theano.tensor as T


class Model(object):
    def __init__(self):
        self.parameters = []

    def get_gradients(self, loss):
        gparams = T.grad(loss, self.parameters)
        gradients = dict(zip(self.parameters, gparams))
        return gradients, {}
