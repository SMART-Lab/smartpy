import numpy as np
from os.path import join as pjoin

from smartpy.misc.utils import HyperparamsMeta


class UpdateRule(object):
    __metaclass__ = HyperparamsMeta

    def __init__(self):
        self.parameters = {}

    def apply(self, gradients):
        raise NameError('Should be implemented by inheriting classes!')

    def save(self, savedir="./", filename="update_rule"):
        parameters = {name: param.get_value() for name, param in self.parameters.items()}
        np.savez(pjoin(savedir, filename + ".npz"), **parameters)

    def load(self, loaddir="./", filename="update_rule"):
        params = np.load(pjoin(loaddir, filename + ".npz"))
        for name, param in self.parameters.items():
            param.set_value(params[name])
