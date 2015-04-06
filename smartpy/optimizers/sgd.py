import theano
import theano.tensor as T

import numpy as np
from collections import OrderedDict
from smartpy.optimizers import Optimizer


class SGD(Optimizer):
    def __init__(self, loss, batch_size=1, update_rules=[]):
        super(SGD, self).__init__(loss, update_rules=update_rules)
        self.batch_size = batch_size

    def initialize(self, model, dataset):
        self.updates = OrderedDict()
        self.dataset = theano.shared(dataset, name='data', borrow=True)
        self.nb_updates_per_epoch = int(np.ceil(len(dataset) / self.batch_size))

        # Build learner
        self.input = T.matrix('input')
        loss = self.loss(self.input)

        self.gradients, updates = model.get_gradients(loss)
        self.updates.update(updates)

        # Apply update rules
        for update_rule in self.update_rules:
            self.gradients, updates = update_rule.apply(self.gradients)
            self.updates.update(updates)  # Add updates from update_rule

        # Update parameters
        for param, gparam in self.gradients.items():
            self.updates[param] = param - self.gradients[param]

    def build_learning_function(self, extra_updates={}):
        if not hasattr(self, "updates"):
            raise NameError("Optimizer has not been initialized! Please use method `initialize(model, dataset)` first.")

        self.updates.update(extra_updates)
        no_batch = T.iscalar('no_batch')
        learn = theano.function([no_batch],
                                updates=self.updates,
                                givens={self.input: self.dataset[no_batch * self.batch_size:(no_batch + 1) * self.batch_size]},
                                name="learn")
        return learn
