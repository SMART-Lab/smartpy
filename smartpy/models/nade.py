import theano
import theano.tensor as T

import numpy as np

from smartpy.models import Model
from smartpy.misc.utils import ACTIVATION_FUNCTIONS
from smartpy.misc.weights_initializer import WeightsInitializer


class NADE(Model):
    __hyperparams__ = {'input_size': int, 'hidden_size': int, 'hidden_activation': ACTIVATION_FUNCTIONS.keys(), 'tied_weights': bool}
    __optional__ = ['hidden_activation', 'tied_weights']

    def __init__(self,
                 input_size,
                 hidden_size,
                 hidden_activation="sigmoid",
                 tied_weights=False):

        self.hyperparams = {'input_size': input_size,
                            'hidden_size': hidden_size,
                            'hidden_activation': hidden_activation,
                            'tied_weights': tied_weights}

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

    def fprop(self, input, return_output_preactivation=False):
        input_times_W = input.T[:, :, None] * self.W[:, None, :]

        # This uses the SplitOp which isn't available yet on the GPU.
        # acc_input_times_W = T.concatenate([T.zeros_like(input_times_W[[0]]), T.cumsum(input_times_W, axis=0)[:-1]], axis=0)
        # Hack to stay on the GPU
        acc_input_times_W = T.cumsum(input_times_W, axis=0)
        acc_input_times_W = T.set_subtensor(acc_input_times_W[1:], acc_input_times_W[:-1])
        acc_input_times_W = T.set_subtensor(acc_input_times_W[0, :], 0.0)

        acc_input_times_W += self.bhid[None, None, :]
        h = self.hidden_activation(acc_input_times_W)

        pre_output = T.sum(h * self.V[:, None, :], axis=2) + self.bvis[:, None]
        output = T.nnet.sigmoid(pre_output)

        if return_output_preactivation:
            return output, pre_output

        return output

    def get_nll(self, input):
        output, pre_output = self.fprop(input, return_output_preactivation=True)
        nll = T.sum(T.nnet.softplus(-input.T * pre_output + (1 - input.T) * pre_output), axis=0)
        return nll

    def mean_nll_loss(self, input):
        nll = self.get_nll(input)
        return nll.mean()
