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

        #self.model = model
        #self.dataset = dataset

        #self.optimizer = optimizer if optimizer is not None else SGD(model)
        self.training_status = training_status if training_status is not None else TrainingStatus(epoch=1)

        #self.batch_size = batch_size if batch_size is not None else len(dataset)
        #self.starting_epoch = starting_epoch
        #self.nb_updates_per_epoch = self.optimizer.(dataset)

        self.stopping_criteria = []
        self.tasks = []

        self.no_epoch = 0
        self.no_update = 0
        self.final_epoch = None
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

    def init(self):
        for task in self.tasks:
            task.init(self.no_epoch, 0)

    def pre_epoch(self):
        for task in self.tasks:
            task.pre_epoch(self.no_epoch, 0)

    def pre_update(self):
        for task in self.tasks:
            task.pre_update(self.no_epoch, self.no_update)

    def post_update(self):
        for task in self.tasks:
            task.post_update(self.no_epoch, self.no_update)

    def post_epoch(self):
        for task in self.tasks:
            task.post_epoch(self.no_epoch, self.no_update)

    def finished(self):
        for task in self.tasks:
            task.finished(self.final_epoch, self.no_update)

    def run(self):
        data = theano.shared(self.dataset, name='data', borrow=True)
        learn = self.optimizer.build_learning_function(data, extra_updates=self.update)
        #theano.printing.pydotprint(learn, '{0}_learn_{1}'.format(model.__class__.__name__, config.device), with_ids=True)

        self.init()

        # Learning
        for self.no_epoch in count(self.starting_epoch):
            # Check stopping criteria
            if any([stopping_criterion.check(self.no_epoch) for stopping_criterion in self.stopping_criteria]):
                break

            self.pre_epoch()

            for self.no_update in xrange(1, self.optimizer.nb_updates_per_epoch+1):
                self.pre_update()
                learn(self.no_update-1)
                self.post_update()

            self.post_epoch()

        self.finished()
        self.final_epoch = self.no_epoch-1
