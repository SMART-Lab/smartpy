import numpy as np
import theano.tensor as T

from os.path import join as pjoin
from smartpy.misc.utils import save_dict_to_json_file, load_dict_from_json_file
from smartpy.misc.utils import HyperparamsMeta


class Model(object):
    __metaclass__ = HyperparamsMeta

    def __init__(self):
        self.parameters = []
        self.hyperparams = {}

    def get_gradients(self, loss):
        gparams = T.grad(loss, self.parameters)
        gradients = dict(zip(self.parameters, gparams))
        return gradients, {}

    def save(self, savedir="./", hyperparams_filename="hyperparams", params_filename="params"):
        save_dict_to_json_file(pjoin(savedir, hyperparams_filename + ".json"), self.hyperparams)
        params = {param.name: param.get_value() for param in self.parameters}
        np.savez(pjoin(savedir, params_filename + ".npz"), **params)

    def load(self, loaddir="./", params_filename="params"):
        params = np.load(pjoin(loaddir, params_filename + ".npz"))
        for param in self.parameters:
            param.set_value(params[param.name])

    @classmethod
    def create(cls, loaddir="./", hyperparams_filename="hyperparams", params_filename="params"):
        hyperparams = load_dict_from_json_file(pjoin(loaddir, hyperparams_filename + ".json"))
        model = cls(**hyperparams)
        model.load(loaddir, params_filename)
        return model
