import numpy as np
from collections import OrderedDict
from itertools import count

from threading import Thread

import theano

from smartpy.trainers import Status


class Trainer(Thread):
    def __init__(self, model, datasets, optimizer, status=None):
        super(Trainer, self).__init__()

        self.model = model
        self.datasets = datasets
        self.optimizer = optimizer
        self.optimizer.initialize(model, *datasets)

        self.status = status if status is not None else Status()

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
            task.init(self.status)

    def _pre_epoch(self):
        for task in self.tasks:
            task.pre_epoch(self.status)

    def _pre_update(self):
        for task in self.tasks:
            task.pre_update(self.status)

    def _post_update(self):
        for task in self.tasks:
            task.post_update(self.status)

    def _post_epoch(self):
        for task in self.tasks:
            task.post_epoch(self.status)

    def _finished(self):
        for task in self.tasks:
            task.finished(self.status)

    def save(self, savedir="./"):
        self.status.save(savedir)
        self.optimizer.save(savedir)
        self.model.save(savedir)

    def load(self, loaddir="./"):
        self.status.load(loaddir)
        self.optimizer.load(loaddir)
        self.model.load(loaddir)

    def run(self):
        learn = self.optimizer.build_learning_function(extra_updates=self.updates)
        #theano.printing.pydotprint(learn, '{0}_learn_{1}'.format(self.model.__class__.__name__, theano.config.device), with_ids=True)

        self._init()

        # Learning
        for no_epoch in count(self.status.current_epoch+1):
            self.status.current_epoch = no_epoch

            # Check stopping criteria
            if any([stopping_criterion.check(self.status) for stopping_criterion in self.stopping_criteria]):
                self.status.current_epoch -= 1  # We did not complete that epoch
                break

            self._pre_epoch()

            for no_update in xrange(1, self.optimizer.nb_updates_per_epoch+1):
                self.status.relative_update = no_update
                self.status.current_update += 1
                self._pre_update()
                learn(no_update-1)
                self._post_update()

            self._post_epoch()

        self._finished()
