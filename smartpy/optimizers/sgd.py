import theano
import theano.tensor as T

import numpy as np
from collections import OrderedDict

from smartpy.learning_rates import ConstantLearningRate
from smartpy.momentums import NoMomentum


class SGD(object):
    def __init__(self, model, dataset, batch_size=1, learning_rate=None, momentum=None):
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate if learning_rate is not None else ConstantLearningRate(lr=1.)
        self.momentum = momentum if momentum is not None else NoMomentum()

        self.nb_updates_per_epoch = int(np.ceil(len(dataset) / self.batch_size))
        self.dataset = theano.shared(dataset, name='data', borrow=True)

        self.updates = OrderedDict()

    def build_learning_function(self, extra_updates={}):
        # Build learner
        self.input = T.matrix('input')
        self.gradients, updates = self.model.get_gradients(self.input)
        self.updates.update(updates)

        # Apply momentum for all params given their gradient.
        self.gradients, updates_momentum = self.momentum(self.gradients)

        # Get learning rates for all params given their gradient.
        self.lr, updates_lr = self.learning_rate(self.gradients)

        self.updates.update(updates_lr)  # Add updates from learning_rate
        self.updates.update(updates_momentum)  # Add updates from momentum

        # Updates parameters
        for param, gparam in self.gradients.items():
            self.updates[param] = param - self.lr[param] * self.gradients[param]

        self.updates.update(extra_updates)

        no_batch = T.iscalar('no_batch')
        learn = theano.function([no_batch],
                                updates=self.updates,
                                givens={self.input: self.dataset[no_batch * self.batch_size:(no_batch + 1) * self.batch_size]},
                                name="learn"
                                )

        return learn
