import theano
import theano.tensor as T

import numpy as np

from smartpy.models import Model
from smartpy.misc.utils import ACTIVATION_FUNCTIONS
from smartpy.misc.weights_initializer import WeightsInitializer

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


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

        self.input_size = input_size
        self.hidden_size = hidden_size
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

        # Build sampling function
        theano_rng = RandomStreams(1234)
        bit = T.iscalar('bit')
        input = T.matrix('input')
        pre_acc = T.dot(input, self.W) + self.bhid
        h = self.hidden_activation(pre_acc)
        pre_output = T.sum(h * self.V[bit], axis=1) + self.bvis[bit]
        probs = T.nnet.sigmoid(pre_output)
        bits = theano_rng.binomial(p=probs, size=probs.shape, n=1, dtype=theano.config.floatX)
        self.sample_bit_plus = theano.function([input, bit], bits)

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
            return output.T, pre_output.T

        return output.T

    def get_nll(self, input):
        output, pre_output = self.fprop(input, return_output_preactivation=True)
        #nll = T.sum(T.nnet.softplus(-input * pre_output + (1 - input) * pre_output), axis=1)
        nll = T.sum(T.nnet.softplus(-input.T * pre_output.T + (1 - input.T) * pre_output.T), axis=0)
        return nll

    def mean_nll_loss(self, input):
        nll = self.get_nll(input)
        return nll.mean()

    # def sample_scan(self, X, seed=1234):
    #     theano_rng = RandomStreams(seed)
    #     samples = T.zeros_like(X)

    #     def _sample_bit(bit, input):
    #         probs = self.fprop(input)[:, [bit]]
    #         bits = theano_rng.binomial(p=probs, size=(X.shape[0], 1), n=1, dtype=theano.config.floatX)
    #         return T.set_subtensor(input[:, [bit]], bits)

    #     partial_samples, updates = theano.scan(_sample_bit,
    #                                            sequences=[np.arange(self.input_size)],
    #                                            outputs_info=[samples])

    #     return partial_samples[-1], updates
    #     def _sample_bit(bit, acc_input_times_W, last_bit):
    #         #acc_input_times_W += self.bhid
    #         h = self.hidden_activation(acc_input_times_W)
    #         pre_output = T.sum(h * self.V[bit], axis=1) + self.bvis[bit]
    #         probs = T.nnet.sigmoid(pre_output)
    #         bits = theano_rng.binomial(p=probs, size=probs.shape, n=1, dtype=theano.config.floatX)
    #         return acc_input_times_W + bits[:, None] * self.W[[bit], :], bits

    #     acc, samples, updates = theano.scan(_sample_bit,
    #                                         sequences=[np.arange(self.input_size)],
    #                                         outputs_info=[(self.bhid[:, None] + S).T, S])

    #     return acc, samples, updates

    def sample(self, nb_samples):
        samples = np.zeros((nb_samples, self.input_size), dtype="float32")
        for bit in range(self.input_size):
            samples[:, bit] = self.sample_bit_plus(samples, bit)

        return samples
