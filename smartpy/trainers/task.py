# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

import os
from time import time
from os.path import join as pjoin

import mlpython.misc.annealed_importance_sampling as AIS

from mlpython.misc.report import write_results_in_csv, write_result_in_cloud
from mlpython.misc import build_average_classification_error
from mlpython.misc import build_average_nll, build_average_free_energy


class StoppingCriterion:
    def check(self, no_epoch):
        raise NotImplementedError("Subclass has to implement this function.")


class Task(object):
    def __init__(self):
        self.updates = OrderedDict()

    def init(self, no_epoch, no_update):
        pass

    def pre_epoch(self, no_epoch, no_update):
        pass

    def pre_update(self, no_epoch, no_update):
        pass

    def post_update(self, no_epoch, no_update):
        pass

    def post_epoch(self, no_epoch, no_update):
        pass

    def finished(self, no_epoch, no_update):
        pass


class View(Task):
    def __init__(self):
        super(View, self).__init__()
        self.value = None
        self.last_epoch = -1
        self.last_update = -1

    def view(self, no_epoch, no_update):
        if self.last_epoch != no_epoch or self.last_update != no_update:
            self.update(no_epoch, no_update)
            self.last_epoch = no_epoch
            self.last_update = no_update

        return self.value

    def update(self, no_epoch, no_update):
        raise NotImplementedError("Subclass has to implement this function.")

    def __str__(self):
        return "{0}".format(self.value)


class Print(Task):
    def __init__(self, view, msg="{0}", each_epoch=1, each_update=0):
        super(Print, self).__init__()
        self.msg = msg
        self.each_epoch = each_epoch
        self.each_update = each_update
        self.view_obj = view

        # Get updates of the view object.
        self.updates.update(view.updates)

    def post_update(self, no_epoch, no_update):
        self.view_obj.post_update(no_epoch, no_update)

        if self.each_update != 0 and no_update % self.each_update == 0:
            value = self.view_obj.view(no_epoch, no_update)
            print self.msg.format(value)

    def post_epoch(self, no_epoch, no_update):
        self.view_obj.post_epoch(no_epoch, no_update)

        if self.each_epoch != 0 and no_epoch % self.each_epoch == 0:
            value = self.view_obj.view(no_epoch, no_update)
            print self.msg.format(value)

    def init(self, no_epoch, no_update):
        self.view_obj.init(no_epoch, no_update)

    def pre_epoch(self, no_epoch, no_update):
        self.view_obj.pre_epoch(no_epoch, no_update)

    def pre_update(self, no_epoch, no_update):
        self.view_obj.pre_update(no_epoch, no_update)

    def finished(self, no_epoch, no_update):
        self.view_obj.finished(no_epoch, no_update)


class PrintEpochDuration(Task):
    def __init__(self):
        super(PrintEpochDuration, self).__init__()

    def init(self, no_epoch, no_update):
        self.training_start_time = time()

    def pre_epoch(self, no_epoch, no_update):
        self.epoch_start_time = time()

    def post_epoch(self, no_epoch, no_update):
        print "Epoch {0} done in {1:.03f} sec.".format(no_epoch, time() - self.epoch_start_time)

    def finished(self, no_epoch, no_update):
        print "Training done in {:.03f} sec.".format(time() - self.training_start_time)
