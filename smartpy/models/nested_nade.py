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
                 noise_lambda=1.,
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
        self.hyperparams['noise_lambda'] = noise_lambda

        self.gamma = gamma
        self.noise_lambda = noise_lambda
        #self.noise = theano.shared(np.zeros((1, self.input_size), dtype=theano.config.floatX),
        #                            name='noise', borrow=True)

    def noise_contrastive_cost(self, input, noise):
        # Difference of the log likelihoods of NestedNADE and trainedNADE
        G = lambda u: (-self.get_nll(u, u)) - (-self.trained_nade.get_nll(u, u))
        #h = lambda u: T.nnet.sigmoid(G(u))

        alike_term = -T.nnet.softplus(-G(input))
        noise_term = -T.nnet.softplus(G(noise))

        self.alike_term_mean = alike_term.mean()
        self.noise_term_mean = noise_term.mean()

        noise_contrastive_costs = alike_term + self.noise_lambda * noise_term
        return 0.5 * noise_contrastive_costs.mean()

    def loss(self, input, noise):
        mean_nll_loss = self.mean_nll_loss(input, input)
        noise_contrastive_cost = self.noise_contrastive_cost(input, noise)
        return mean_nll_loss - self.gamma * noise_contrastive_cost
