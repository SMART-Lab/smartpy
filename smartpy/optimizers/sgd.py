import theano
import theano.tensor as T

import numpy as np
from collections import OrderedDict
from smartpy.optimizers import Optimizer


class SGD(Optimizer):
    def __init__(self, loss, batch_size=1, update_rules=[]):
        super(SGD, self).__init__(loss, update_rules=update_rules)
        self.batch_size = batch_size

    def initialize(self, model, *datasets):
        self.updates = OrderedDict()
        self.datasets = [theano.shared(dataset, name='data', borrow=True) for dataset in datasets]
        self.nb_updates_per_epoch = int(np.ceil(len(datasets[0]) / self.batch_size))

        # Build learner
        self.inputs = [T.matrix('input' + str(i)) for i in range(len(datasets))]
        self.objective = self.loss(*self.inputs)

        self.gradients, updates = model.get_gradients(self.objective)
        self.updates.update(updates)

        # Apply update rules
        for update_rule in self.update_rules:
            gradients, updates = update_rule.apply(self.gradients)
            self.gradients.update(gradients)
            self.updates.update(updates)  # Add updates from update_rule

        # Update parameters
        for param, gparam in self.gradients.items():
            self.updates[param] = param - self.gradients[param]

    def build_learning_function(self, extra_updates={}):
        if not hasattr(self, "updates"):
            raise NameError("Optimizer has not been initialized! Please use method `initialize(model, *datasets)` first.")

        self.updates.update(extra_updates)
        no_batch = T.iscalar('no_batch')
        givens = {input: dataset[no_batch * self.batch_size:(no_batch + 1) * self.batch_size] for input, dataset in zip(self.inputs, self.datasets)}
        learn = theano.function([no_batch],
                                updates=self.updates,
                                givens=givens,
                                name="learn")
        return learn
