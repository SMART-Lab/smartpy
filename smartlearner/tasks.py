# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
from time import time

from .interfaces import Task, RecurrentTask


class MonitorVariable(Task):
    def __init__(self, var):
        super().__init__()
        self.var = self.track_variable(var)

    @property
    def value(self):
        return self.var.get_value()


class PrintVariable(RecurrentTask):
    def __init__(self, msg, *variables, **recurrent_options):
        # TODO: docstring should include **recurrent_options.
        super(PrintVariable, self).__init__(**recurrent_options)
        self.msg = msg
        self.variables = [self.track_variable(v) for v in variables]

    def execute(self, status):
        print(self.msg.format(*[v.get_value() for v in self.variables]))


class PrintEpochDuration(RecurrentTask):
    def __init__(self, **recurrent_options):
        # TODO: docstring should include **recurrent_options.
        super(PrintEpochDuration, self).__init__(**recurrent_options)

    def execute(self, status):
        print("Epoch {0} done in {1:.03f} sec.".format(status.current_epoch, time() - self.epoch_start_time))

    def pre_epoch(self, status):
        self.epoch_start_time = time()


class PrintTrainingDuration(Task):
    def init(self, status):
        self.start_time = time()

    def finished(self, status):
        print("Training done in {:.03f} sec.".format(time() - self.start_time))


class PrintAverageTrainingLoss(Task):
    def __init__(self, loss):
        super().__init__()
        self.loss = self.track_variable(loss._loss, name="Objective")

    def pre_epoch(self, status):
        self.values = []

    def post_update(self, status):
        self.values.append(self.loss.get_value())

    def post_epoch(self, status):
        print("Average training loss: {}".format(np.mean(self.values)))


class Breakpoint(RecurrentTask):
    def __init__(self, **recurrent_options):
        # TODO: docstring should include **recurrent_options.
        super(Breakpoint, self).__init__(**recurrent_options)

    def execute(self, status):
        try:
            from ipdb import set_trace as dbg
        except ImportError:
            from pdb import set_trace as dbg
        dbg()


class Print(RecurrentTask):
    def __init__(self, msg, *views, **recurrent_options):
        # TODO: docstring should include **recurrent_options.
        super(Print, self).__init__(**recurrent_options)
        self.msg = msg
        self.views = views

        # Add updates of the views.
        for view in self.views:
            self.updates.update(view.updates)

    def execute(self, status):
        values = [view.view(status) for view in self.views]
        print(self.msg.format(*values))


class Callback(RecurrentTask):
    def __init__(self, callback, **recurrent_options):
        # TODO: docstring should include **recurrent_options.
        super(Callback, self).__init__(**recurrent_options)
        self.callback = callback

    def execute(self, status):
        self.callback(self, status)
