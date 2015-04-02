import numpy as np
from collections import OrderedDict
from itertools import count

from threading import Thread

import theano

#from smartpy.optimizers import SGD
from smartpy.trainers import TrainingStatus


class Trainer(Thread):
    def __init__(self, optimizer, training_status=None):
        super(Trainer, self).__init__()

        self.optimizer = optimizer
        self.training_status = training_status if training_status is not None else TrainingStatus()

        self.stopping_criteria = []
        self.tasks = []

        self.updates = OrderedDict()

    def add_stopping_criterion(self, criterion):
        self.stopping_criteria.append(criterion)

    def add_task(self, task):
        self.updates.update(task.updates)
        self.tasks.append(task)

    def track_variable(self, var, shape, name=""):
        var_shared = theano.shared(np.zeros(shape, dtype=theano.config.floatX), name=name)
        self.updates[var_shared] = var
        return var_shared

    def _init(self):
        for task in self.tasks:
            task.init(self.training_status)

    def _pre_epoch(self):
        for task in self.tasks:
            task.pre_epoch(self.training_status)

    def _pre_update(self):
        for task in self.tasks:
            task.pre_update(self.training_status)

    def _post_update(self):
        for task in self.tasks:
            task.post_update(self.training_status)

    def _post_epoch(self):
        for task in self.tasks:
            task.post_epoch(self.training_status)

    def _finished(self):
        for task in self.tasks:
            task.finished(self.training_status)

    def run(self):
        learn = self.optimizer.build_learning_function(extra_updates=self.updates)
        theano.printing.pydotprint(learn, '{0}_learn_{1}'.format(self.optimizer.model.__class__.__name__, theano.config.device), with_ids=True)

        self._init()

        # Learning
        for no_epoch in count(self.training_status.current_epoch):
            self.training_status.current_epoch = no_epoch

            # Check stopping criteria
            if any([stopping_criterion.check(self.training_status) for stopping_criterion in self.stopping_criteria]):
                self.training_status.current_epoch -= 1  # We did not complete that epoch
                break

            self._pre_epoch()

            for no_update in xrange(1, self.optimizer.nb_updates_per_epoch+1):
                self.training_status.relative_update = no_update
                self.training_status.current_update += 1
                self._pre_update()
                learn(no_update-1)
                self._post_update()

            self._post_epoch()

        self._finished()
