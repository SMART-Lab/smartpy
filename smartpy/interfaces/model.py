import theano.tensor as T
from collections import OrderedDict
from abc import ABCMeta, abstractmethod, abstractproperty


class abstractclassmethod(classmethod):
    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)


class Model(object):
    __metaclass__ = ABCMeta

    def get_gradients(self, loss):
        gparams = T.grad(loss, list(self.parameters.values()))
        gradients = dict(zip(self.parameters.values(), gparams))
        return gradients, OrderedDict()

    @abstractproperty
    def parameters(self):
        raise NotImplementedError("Subclass of 'Model' must implement property 'parameters'.")

    @abstractmethod
    def save(path):
        raise NotImplementedError("Subclass of 'Model' must implement 'save(path)'.")

    @abstractclassmethod
    def load(path):
        raise NotImplementedError("Subclass of 'Model' must implement 'load(path)'.")
