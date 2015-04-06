# -*- coding: utf-8 -*-
from __future__ import division

from collections import OrderedDict
from time import time


class StoppingCriterion:
    def check(self, status):
        raise NotImplementedError("Subclass has to implement this function.")


class Task(object):
    def __init__(self):
        self.updates = OrderedDict()

    def init(self, status):
        pass

    def pre_epoch(self, status):
        pass

    def pre_update(self, status):
        pass

    def post_update(self, status):
        pass

    def post_epoch(self, status):
        pass

    def finished(self, status):
        pass


class View(Task):
    def __init__(self):
        super(View, self).__init__()
        self.value = None
        self.last_update = -1

    def view(self, status):
        if self.last_update != status.current_update:
            self.update(status)
            self.last_update = status.current_update

        return self.value

    def update(self, status):
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

    def post_update(self, status):
        self.view_obj.post_update(status)

        if self.each_update != 0 and status.current_update % self.each_update == 0:
            value = self.view_obj.view(status)
            print self.msg.format(value)

    def post_epoch(self, status):
        self.view_obj.post_epoch(status)

        if self.each_epoch != 0 and status.current_epoch % self.each_epoch == 0:
            value = self.view_obj.view(status)
            print self.msg.format(value)

    def init(self, status):
        self.view_obj.init(status)

    def pre_epoch(self, status):
        self.view_obj.pre_epoch(status)

    def pre_update(self, status):
        self.view_obj.pre_update(status)

    def finished(self, status):
        self.view_obj.finished(status)


class PrintEpochDuration(Task):
    def __init__(self):
        super(PrintEpochDuration, self).__init__()

    def init(self, status):
        self.training_start_time = time()

    def pre_epoch(self, status):
        self.epoch_start_time = time()

    def post_epoch(self, status):
        print "Epoch {0} done in {1:.03f} sec.".format(status.current_epoch, time() - self.epoch_start_time)

    def finished(self, status):
        print "Training done in {:.03f} sec.".format(time() - self.training_start_time)


class MaxEpochStopping(StoppingCriterion):
    def __init__(self, nb_epochs_max):
        self.nb_epochs_max = nb_epochs_max

    def check(self, status):
        return status.current_epoch > self.nb_epochs_max
