import theano
import theano.tensor as T

import numpy as np

from weights_initializer import WeightsInitializer


class NADE(object):
    def __init__(self,
                 input_size,
                 hidden_size,
                 hidden_activation=T.nnet.sigmoid,
                 weights_initialization=None,
                 tied_weights=False):

        self.hidden_size = hidden_size
        self.hidden_activation = hidden_activation

        if weights_initialization is None:
            weights_initialization = WeightsInitializer().zeros

        # Initialize layers
        self.W = theano.shared(value=weights_initialization((input_size, hidden_size)), name='W', borrow=True)
        self.bhid = theano.shared(value=np.zeros(hidden_size, dtype=theano.config.floatX), name='bhid', borrow=True)
        self.bvis = theano.shared(value=np.zeros(input_size, dtype=theano.config.floatX), name='bvis', borrow=True)
        self.parameters = [self.W, self.bhid, self.bvis]

        self.V = self.W
        if not tied_weights:
            self.V = theano.shared(value=weights_initialization((input_size, hidden_size)), name='V', borrow=True)
            self.parameters.append(self.V)

    def get_fprop(self, input, return_output_preactivation=False):
        input_times_W = input.T[:, :, None] * self.W[:, None, :]

        # This uses the SplitOp which isn't available yet on the GPU.
        # acc_input_times_W = T.concatenate([T.zeros_like(input_times_W[[0]]), T.cumsum(input_times_W, axis=0)[:-1]], axis=0)
        # Hack to stay on the GPU
        acc_input_times_W = T.cumsum(input_times_W, axis=0)
        acc_input_times_W = T.set_subtensor(acc_input_times_W[1:], acc_input_times_W[:-1])
        acc_input_times_W = T.set_subtensor(acc_input_times_W[0, :], 0.0)

        acc_input_times_W += self.b[None, None, :]
        h = self.hidden_activation(acc_input_times_W)

        pre_output = T.sum(h * self.W_prime[:, None, :], axis=2) + self.b_prime[:, None]
        output = T.nnet.sigmoid(pre_output)

        if return_output_preactivation:
            return output, pre_output

        return output

    def get_nll(self, input):
        output, pre_output = self.fprop(input, return_output_preactivation=True)
        nll = T.sum(T.nnet.softplus(-input.T * pre_output + (1 - input.T) * pre_output), axis=0)
        return nll

    def get_loss(self, input):
        nll = self.get_nll(input)
        return nll.mean()

    def get_gradients(self, input):
        loss = self.get_loss(input)

        gparams = T.grad(loss, self.parameters)
        gradients = dict(zip(self.parameters, gparams))
        return gradients
