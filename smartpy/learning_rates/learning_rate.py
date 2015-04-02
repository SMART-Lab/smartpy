import theano
import numpy as np

from collections import defaultdict


class CustomDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        defaultdict.__init__(self, *args, **kwargs)

    def __getitem__(self, key):
        return defaultdict.__getitem__(self, str(key))

    def __setitem__(self, key, val):
        defaultdict.__setitem__(self, str(key), val)


class CustomDict(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, str(key))

    def __setitem__(self, key, val):
        dict.__setitem__(self, str(key), val)


class LearningRate:
    def __init__(self, lr):
        self.base_lr = lr
        self.lr = CustomDefaultDict(lambda: theano.shared(np.array(lr, dtype=theano.config.floatX)))

    def set_individual_lr(self, param, lr):
        self.lr[param].set_value(lr)

    def __call__(self, gradients):
        raise NameError('Should be implemented by inheriting class!')

    def __getstate__(self):
        # Convert defaultdict into a dict
        self.__dict__.update({"lr": CustomDict(self.lr)})
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

        if type(self.lr) is not CustomDict:
            self.lr = CustomDict()
            for k, v in state['lr'].items():
                self.lr[k] = v

        # Make sure each learning rate have the right dtype
        self.lr = CustomDict({k: theano.shared(v.get_value().astype(theano.config.floatX), name='lr_' + k) for k, v in self.lr.items()})
