from smartpy.misc.utils import HyperparamsMeta


class UpdateRule(object):
    __metaclass__ = HyperparamsMeta

    def __init__(self):
        self.parameters = {}

    def apply(self, gradients):
        raise NameError('Should be implemented by inheriting classes!')

    def save(self):
        pass

    def load(self, state):
        pass

    # def __getstate__(self):
    #     # Convert defaultdict into a dict
    #     self.__dict__.update({"update_rules": CustomDict(self.lr)})
    #     return self.__dict__

    # def __setstate__(self, state):
    #     self.__dict__.update(state)

    #     if type(self.lr) is not CustomDict:
    #         self.lr = CustomDict()
    #         for k, v in state['update_rules'].items():
    #             self.lr[k] = v

    #     # Make sure each update rule have the right dtype
    #     self.lr = CustomDict({k: theano.shared(v.get_value().astype(theano.config.floatX), name='update_rule_' + k) for k, v in self.lr.items()})
