import theano
import theano.tensor as T

import numpy as np

from smartpy.models import Model
from smartpy.misc.utils import ACTIVATION_FUNCTIONS, Timer
from smartpy.misc.weights_initializer import WeightsInitializer

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class NADE(Model):
    __hyperparams__ = {'input_size': int,
                       'hidden_size': int,
                       'hidden_activation': ACTIVATION_FUNCTIONS.keys(),
                       'tied_weights': bool}
    __optional__ = ['hidden_activation', 'tied_weights']

    def __init__(self,
                 input_size,
                 hidden_size,
                 hidden_activation="sigmoid",
                 tied_weights=False,
                 ordering_seed=None,
                 *args, **kwargs):
        super(NADE, self).__init__(*args, **kwargs)

        self.hyperparams['input_size'] = input_size
        self.hyperparams['hidden_size'] = hidden_size
        self.hyperparams['hidden_activation'] = hidden_activation
        self.hyperparams['tied_weights'] = tied_weights
        self.hyperparams['ordering_seed'] = ordering_seed

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_activation = ACTIVATION_FUNCTIONS[hidden_activation]
        self.tied_weights = tied_weights
        self.ordering_seed = ordering_seed

        # Define layers weights and biases (a.k.a parameters)
        self.W = theano.shared(value=np.zeros((input_size, hidden_size), dtype=theano.config.floatX), name='W', borrow=True)
        self.bhid = theano.shared(value=np.zeros(hidden_size, dtype=theano.config.floatX), name='bhid', borrow=True)
        self.bvis = theano.shared(value=np.zeros(input_size, dtype=theano.config.floatX), name='bvis', borrow=True)
        self.parameters.extend([self.W, self.bhid, self.bvis])

        self.V = self.W
        if not tied_weights:
            self.V = theano.shared(value=np.zeros((input_size, hidden_size), dtype=theano.config.floatX), name='V', borrow=True)
            self.parameters.append(self.V)

        self.ordering = np.arange(self.input_size)
        if self.ordering_seed is not None:
            rng = np.random.RandomState(self.ordering_seed)
            rng.shuffle(self.ordering)

        self.ordering_reverse = np.argsort(self.ordering)

    def build_sampling_function(self, seed=None):
        # Build sampling function
        rng = np.random.RandomState(seed)
        theano_rng = RandomStreams(rng.randint(2**30))
        bit = T.iscalar('bit')
        input = T.matrix('input')
        pre_acc = T.dot(input, self.W) + self.bhid
        h = self.hidden_activation(pre_acc)
        pre_output = T.sum(h * self.V[bit], axis=1) + self.bvis[bit]
        probs = T.nnet.sigmoid(pre_output)
        bits = theano_rng.binomial(p=probs, size=probs.shape, n=1, dtype=theano.config.floatX)
        sample_bit_plus = theano.function([input, bit], bits)

        def _sample(nb_samples):
            with Timer("Generating {} samples from NADE".format(nb_samples)):
                samples = np.zeros((nb_samples, self.input_size), dtype="float32")
                for bit in range(self.input_size):
                    samples[:, bit] = sample_bit_plus(samples, bit)

                return samples
        return _sample

    def build_conditional_sampling_function(self, seed=None):
        # Build sampling function
        rng = np.random.RandomState(seed)
        theano_rng = RandomStreams(rng.randint(2**30))
        bit = T.iscalar('bit')
        input = T.matrix('input')
        pre_acc = T.dot(input, self.W) + self.bhid
        h = self.hidden_activation(pre_acc)
        pre_output = T.sum(h * self.V[bit], axis=1) + self.bvis[bit]
        probs = T.nnet.sigmoid(pre_output)
        bits = theano_rng.binomial(p=probs, size=probs.shape, n=1, dtype=theano.config.floatX)
        sample_bit_plus = theano.function([input, bit], bits)

        def _sample(examples, alpha=0):
            """
            alpha : ratio of input units to condition on.
            """
            assert alpha >= 0 and alpha <= 1
            start = int(np.ceil(alpha*self.input_size))
            samples = examples[:, self.ordering]  # Change input units ordering.
            samples[:, start:] = 0.
            for bit in range(start, self.input_size):
                samples[:, bit] = sample_bit_plus(samples, bit)

            # Change back the input units ordering.
            return samples[:, self.ordering_reverse]
        return _sample

    def initialize(self, weights_initialization=None):
        if weights_initialization is None:
            weights_initialization = WeightsInitializer().uniform

        self.W.set_value(weights_initialization(self.W.get_value().shape))

        if not self.tied_weights:
            self.V.set_value(weights_initialization(self.V.get_value().shape))

    def fprop(self, input, return_output_preactivation=False):
        input = input[:, self.ordering]  # Does not matter if ordering is the default one, indexing is fast.
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

    def get_nll(self, input, target):
        #self.sum_diff_input_target = T.sum(T.sum(abs(target-input), axis=1) > 0)

        target = target[:, self.ordering]  # Does not matter if ordering is the default one, indexing is fast.
        output, pre_output = self.fprop(input, return_output_preactivation=True)
        nll = T.sum(T.nnet.softplus(-target.T * pre_output.T + (1 - target.T) * pre_output.T), axis=0)
        #nll = T.sum(T.nnet.softplus(-input.T * pre_output.T + (1 - input.T) * pre_output.T), axis=0)

        # The following does not give the same results, numerical precision error?
        #nll = T.sum(T.nnet.softplus(-target * pre_output + (1 - target) * pre_output), axis=1)
        #nll = T.sum(T.nnet.softplus(-input * pre_output + (1 - input) * pre_output), axis=1)

        return nll

    def mean_nll_loss(self, input, target):
        nll = self.get_nll(input, target)
        return nll.mean()
