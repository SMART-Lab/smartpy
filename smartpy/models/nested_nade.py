import theano
import theano.tensor as T

import numpy as np

from smartpy.models.nade import NADE
from smartpy.misc.utils import ACTIVATION_FUNCTIONS
from smartpy.misc.weights_initializer import WeightsInitializer


class NestedNADE(NADE):
    __hyperparams__ = {'input_size': int, 'hidden_size': int, 'hidden_activation': ACTIVATION_FUNCTIONS.keys(), 'tied_weights': bool}
    __optional__ = ['hidden_activation', 'tied_weights']

    def __init__(self,
                 input_size,
                 hidden_size,
                 trained_nade,
                 gamma=1.,
                 hidden_activation="sigmoid",
                 tied_weights=False):

        self.hyperparams = {'input_size': input_size,
                            'hidden_size': hidden_size,
                            'hidden_activation': hidden_activation,
                            'tied_weights': tied_weights}

        self.trained_nade = trained_nade
        self.gamma = gamma
        self.hidden_activation = ACTIVATION_FUNCTIONS[hidden_activation]
        self.tied_weights = tied_weights

        # Define layers weights and biases (a.k.a parameters)
        self.W = theano.shared(value=np.zeros((input_size, hidden_size), dtype=theano.config.floatX), name='W', borrow=True)
        self.bhid = theano.shared(value=np.zeros(hidden_size, dtype=theano.config.floatX), name='bhid', borrow=True)
        self.bvis = theano.shared(value=np.zeros(input_size, dtype=theano.config.floatX), name='bvis', borrow=True)
        self.parameters = [self.W, self.bhid, self.bvis]

        self.V = self.W
        if not tied_weights:
            self.V = theano.shared(value=np.zeros((input_size, hidden_size), dtype=theano.config.floatX), name='V', borrow=True)
            self.parameters.append(self.V)

    def initialize(self, weights_initialization=None):
        if weights_initialization is None:
            weights_initialization = WeightsInitializer().uniform

        self.W.set_value(weights_initialization(self.W.get_value().shape))

        if not self.tied_weights:
            self.V.set_value(weights_initialization(self.V.get_value().shape))

    def noise_contrastive_loss(self, input, noise):
        #noise, updates = self.trained_nade.sample(input)

        G = lambda u: (-self.get_nll(u)) - (-self.trained_nade.get_nll(u))
        h = lambda u: T.nnet.sigmoid(G(u))

        noise_contrastive_losses = T.log(h(input)) + T.log(1 - h(noise))
        return 0.5 * noise_contrastive_losses.mean()

    def loss(self, input, noise):
        mean_nll_loss = self.mean_nll_loss(input)
        noise_contrastive_loss = self.noise_contrastive_loss(input, noise)
        return mean_nll_loss + self.gamma * noise_contrastive_loss

    # def get_gradients(self, loss):
    #     gparams = T.grad(loss, self.parameters)
    #     gradients = dict(zip(self.parameters, gparams))
    #     return gradients, {}
