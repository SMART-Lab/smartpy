import theano
import theano.tensor as T

import numpy as np

from smartpy.models.nade import NADE
from smartpy.misc.utils import ACTIVATION_FUNCTIONS


class NestedNADE(NADE):
    __hyperparams__ = {'input_size': int, 'hidden_size': int, 'hidden_activation': ACTIVATION_FUNCTIONS.keys(), 'tied_weights': bool}
    __optional__ = ['hidden_activation', 'tied_weights']

    def __init__(self,
                 trained_nade,
                 hidden_size=None,
                 gamma=1.,
                 hidden_activation=None,
                 *args, **kwargs):

        # Load already trained NADE
        self.trained_nade = NADE.create(trained_nade)

        if hidden_size is None:
            hidden_size = self.trained_nade.hyperparams["hidden_size"]

        if hidden_activation is None:
            hidden_activation = self.trained_nade.hyperparams["hidden_activation"]

        super(NestedNADE, self).__init__(self.trained_nade.input_size,
                                         hidden_size,
                                         hidden_activation=hidden_activation,
                                         *args, **kwargs)

        self.hyperparams['trained_nade'] = trained_nade
        self.hyperparams['gamma'] = gamma

        self.gamma = gamma
        #self.noise = theano.shared(np.zeros((1, self.input_size), dtype=theano.config.floatX),
        #                            name='noise', borrow=True)

    def noise_contrastive_cost(self, input, noise):
        G = lambda u: (-self.get_nll(u)) - (-self.trained_nade.get_nll(u))
        h = lambda u: T.nnet.sigmoid(G(u))

        noise_contrastive_costs = T.log(h(input)) + T.log(1 - h(noise))
        return 0.5 * noise_contrastive_costs.mean()

    def loss(self, input, noise):
        mean_nll_loss = self.mean_nll_loss(input)
        noise_contrastive_cost = self.noise_contrastive_cost(input, noise)
        return mean_nll_loss - self.gamma * noise_contrastive_cost
