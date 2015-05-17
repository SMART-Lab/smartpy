import numpy as np
import theano

WEIGHTS_INITIALIZERS = ['uniform', 'zeros', 'diagonal', 'orthogonal', 'gaussian']


class WeightsInitializer(object):

    def __init__(self, random_seed=None):
        self.rng = np.random.mtrand.RandomState(random_seed)

    def _init_range(self, dim):
        return np.sqrt(6. / (dim[0] + dim[1]))

    def uniform(self, dim):
        init_range = self._init_range(dim)
        return np.asarray(self.rng.uniform(low=-init_range, high=init_range, size=dim), dtype=theano.config.floatX)

    def zeros(self, dim):
        return np.zeros(dim, dtype=theano.config.floatX)

    def diagonal(self, dim):
        W_values = self.zeros(dim)
        np.fill_diagonal(W_values, 1)
        return W_values

    def orthogonal(self, dim):
        max_dim = max(dim)
        return np.linalg.svd(self.uniform((max_dim, max_dim)))[2][:dim[0], :dim[1]]

    def gaussian(self, dim):
        return np.asarray(self.rng.normal(loc=0, scale=self._init_range(dim), size=dim), dtype=theano.config.floatX)


def factory(**hyperparams):
    """ Gets a weights initialization function from `hyperparams`. """

    weights_initializer = WeightsInitializer(hyperparams.get("seed", 1234))

    if "weights_initialization" not in hyperparams:
        raise ValueError("Hyperparameter 'weights_initialization' is mandatory ({}).".format(",".join(WEIGHTS_INITIALIZERS)))
    elif hyperparams["weights_initialization"] == "uniform":
        weights_initialization_method = weights_initializer.uniform
    elif hyperparams["weights_initialization"] == "zeros":
        weights_initialization_method = weights_initializer.zeros
    elif hyperparams["weights_initialization"] == "diagonal":
        weights_initialization_method = weights_initializer.diagonal
    elif hyperparams["weights_initialization"] == "orthogonal":
        weights_initialization_method = weights_initializer.orthogonal
    elif hyperparams["weights_initialization"] == "gaussian":
        weights_initialization_method = weights_initializer.gaussian
    else:
        raise ValueError("The following weight initialization method is not implemented: " + hyperparams["weights_initialization"])

    return weights_initialization_method
