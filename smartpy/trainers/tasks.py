# -*- coding: utf-8 -*-
from __future__ import division

import os
import numpy as np
from collections import OrderedDict
from time import time

from smartpy.misc import utils


class StoppingCriterion(object):
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

    def execute(self, status):
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


class ItemGetter(View):
    def __init__(self, view, attribute):
        """ Retrieves `attribute` from a `view` which outputs a dictionnary """
        super(ItemGetter, self).__init__()
        self.view_obj = view
        self.attribute = attribute

    def update(self, status):
        infos = self.view_obj.view(status)
        self.value = infos[self.attribute]


class PrintSharedVariable(Task):
    def __init__(self, shared_vars, msg="{0}", each_epoch=1, each_update=0):
        super(PrintSharedVariable, self).__init__()
        self.msg = msg
        self.each_epoch = each_epoch
        self.each_update = each_update
        self.shared_vars = shared_vars

    def post_update(self, status):
        if self.each_update != 0 and status.current_update % self.each_update == 0:
            print self.msg.format(*[shared_var.get_value() for shared_var in self.shared_vars])

    def post_epoch(self, status):
        if self.each_epoch != 0 and status.current_epoch % self.each_epoch == 0:
            print self.msg.format(*[shared_var.get_value() for shared_var in self.shared_vars])


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
        self.training_start_time = time()  # TOFIX: In case we are resuming

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
        return status.current_epoch >= self.nb_epochs_max


class Evaluate(View):
    def __init__(self, func):
        super(Evaluate, self).__init__()
        self.func = func

    def update(self, status):
        self.value = self.func()


class Loss(Evaluate):
    def __init__(self, loss, *datasets):
        import theano
        import theano.tensor as T

        datasets = [theano.shared(dataset, name='data', borrow=True) for dataset in datasets]
        #inputs = [T.matrix('input' + str(i)) for i in range(len(datasets))]
        compute_loss = theano.function([], loss(*datasets), name="Loss")
        super(Loss, self).__init__(compute_loss)


class AverageObjective(Task):
    def __init__(self, trainer):
        super(AverageObjective, self).__init__()
        self.objective = trainer.track_variable(trainer.optimizer.objective, shape=np.float32(0).shape, name="Objective")

    def pre_epoch(self, status):
        self.values = []

    def post_update(self, status):
        self.values.append(self.objective.get_value())

    def post_epoch(self, status):
        print "Average objective: {}".format(np.mean(self.values))


class EvaluateNLL(Evaluate):
    def __init__(self, nll, datasets, batch_size=None):
        import theano
        import theano.tensor as T

        shared_datasets = []
        for i, dataset in enumerate(datasets):
            dataset_shared = dataset
            if isinstance(dataset, np.ndarray):
                dataset_shared = theano.shared(dataset, name='data_'+str(i), borrow=True)

            shared_datasets.append(dataset_shared)

        if batch_size is None:
            batch_size = len(shared_datasets[0].get_value())

        nb_batches = int(np.ceil(len(shared_datasets[0].get_value()) / batch_size))

        inputs = [T.matrix('input' + str(i)) for i in range(len(shared_datasets))]
        objective = nll(*inputs)
        no_batch = T.iscalar('no_batch')
        givens = {input: dataset[no_batch * batch_size:(no_batch + 1) * batch_size] for input, dataset in zip(inputs, shared_datasets)}
        compute_nll = theano.function([no_batch], objective, givens=givens, name="NLL")

        def _nll_mean_and_std():
            nlls = []
            for i in range(nb_batches):
                nlls.append(compute_nll(i))

            nlls = np.concatenate(nlls)
            return round(nlls.mean(), 6), round(nlls.std() / np.sqrt(nlls.shape[0]), 6)

        super(EvaluateNLL, self).__init__(_nll_mean_and_std)

    @property
    def mean(self):
        return ItemGetter(self, attribute=0)

    @property
    def std(self):
        return ItemGetter(self, attribute=1)


class EarlyStopping(Task, StoppingCriterion):
    def __init__(self, objective, lookahead, save_task=None, eps=0.):
        super(EarlyStopping, self).__init__()

        self.objective = objective
        self.lookahead = lookahead
        self.save_task = save_task
        self.eps = eps
        self.stopping = False

    def init(self, status):
        if 'best_epoch' not in status.extra:
            status.extra['best_epoch'] = 0

        if 'best_objective' not in status.extra:
            status.extra['best_objective'] = float(np.inf)

    def check(self, status):
        objective = self.objective.view(status)
        if objective + self.eps < status.extra['best_objective']:
            print "Best epoch {} ({:.20f})".format(status.current_epoch, objective)
            status.extra['best_objective'] = float(objective)
            status.extra['best_epoch'] = status.current_epoch

            if self.save_task is not None:
                self.save_task.execute(status)

        return status.current_epoch - status.extra['best_epoch'] >= self.lookahead


class SaveTraining(Task):
    def __init__(self, trainer, savedir, each_epoch=1):
        super(SaveTraining, self).__init__()

        self.savedir = savedir
        self.trainer = trainer
        self.each_epoch = each_epoch

        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)

    def execute(self, status):
        self.trainer.save(self.savedir)

    def post_epoch(self, status):
        if status.current_epoch % self.each_epoch == 0:
            self.execute(status)


class LogResultCSV(Task):
    def __init__(self, log_file, log_entry, formatting={}):
        self.log_file = log_file
        self.log_entry = log_entry
        self.formatting = formatting

    def execute(self, status):
        header = []
        entry = []
        for k, v in self.log_entry.items():
            value = v
            if callable(v):
                value = v(status)
            elif isinstance(v, View):
                value = v.view(status)

            header.append(k)
            entry.append(self.formatting.get(k, "{}").format(value))

        utils.write_log_file(self.log_file, header, entry)


class LogResultGSheet(Task):
    def __init__(self, sheet_id, email, password, worksheet_name, log_entry, formatting={}):
        self.gsheet_params = (sheet_id, email, password)
        self.worksheet_name = worksheet_name
        self.log_entry = log_entry
        self.formatting = formatting

    def execute(self, status):
        header = []
        entry = []
        for k, v in self.log_entry.items():
            value = v
            if callable(v):
                value = v(status)
            elif isinstance(v, View):
                value = v.view(status)

            header.append(k)
            entry.append(self.formatting.get(k, "{}").format(value))

        from smartpy.misc.gsheet import GSheet
        gsheet = GSheet(*self.gsheet_params)
        utils.write_log_gsheet(gsheet, self.worksheet_name, header, entry)


class Sampling(Task):
    def __init__(self, sampling_func, nb_samples, shape, each_epoch=0):
        super(Sampling, self).__init__()

        import theano
        self.sampling_func = sampling_func
        self.nb_samples = nb_samples
        self.each_epoch = each_epoch
        self.samples = theano.shared(np.zeros((nb_samples,) + tuple(shape), dtype=theano.config.floatX),
                                     name='samples', borrow=True)

    def _sample(self):
        self.samples.set_value(self.sampling_func(self.nb_samples))

    def init(self, status):
        self._sample()

    def post_epoch(self, status):
        if self.each_epoch != 0 and status.current_epoch % self.each_epoch == 0:
            self._sample()
