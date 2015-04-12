#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from collections import OrderedDict
import argparse

from smartpy.misc import utils
from smartpy.trainers import tasks, Status
from smartpy.models.nade import NADE
from smartpy.misc.dataset import UnsupervisedDataset as Dataset


DATASETS = ['binarized_mnist']


def buildArgsParser():
    DESCRIPTION = "Evaluate NADE model on a given dataset."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('experiment', type=str, help='folder where to find the experiment')
    p.add_argument('--dataset', type=str, help='dataset to use [{0}].'.format(', '.join(DATASETS)), default=DATASETS[0], choices=DATASETS)

    # General parameters (optional)
    p.add_argument('--gsheet', type=str, metavar="SHEET_ID EMAIL PASSWORD", help="log results into a Google's Spreadsheet.")

    p.add_argument('-v', '--verbose', action='store_true', help='produce verbose output')
    p.add_argument('-f', '--force',  action='store_true', help='permit overwriting')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    with utils.Timer("Loading dataset"):
        dataset = Dataset(args.dataset)

    with utils.Timer("Loading model"):
        nade = NADE.create(args.experiment)

    # with utils.Timer("Loading experiment"):
    #     trainer = trainers.load(args.experiment)
    #     #nade = trainer.model

    ### Temporary patch ###
    import pickle
    from os.path import join as pjoin
    from smartpy.misc.utils import load_dict_from_json_file
    status = load_dict_from_json_file(pjoin(args.experiment, "status.json"))
    best_epoch = status["extra"]["best_epoch"]
    command = pickle.load(open(pjoin(args.experiment, "command.pkl")))
    lr = float(command[command.index("--AdamV1") + 1])
    ######

    nll_train = tasks.EvaluateNLL(nade.get_nll, dataset.trainset, batch_size=100)
    nll_valid = tasks.EvaluateNLL(nade.get_nll, dataset.validset, batch_size=100)
    nll_test = tasks.EvaluateNLL(nade.get_nll, dataset.testset, batch_size=100)

    log_entry = OrderedDict()
    log_entry["Learning Rate"] = lr  # trainer.optimizer.update_rules[0].lr
    log_entry["Random Seed"] = 1234
    log_entry["Hidden Size"] = nade.hyperparams["hidden_size"]
    log_entry["Activation Function"] = nade.hyperparams["hidden_activation"]
    log_entry["Tied Weights"] = nade.hyperparams["tied_weights"]
    log_entry["Best Epoch"] = best_epoch  # trainer.status.extra["best_epoch"]
    log_entry["Look Ahead"] = 10  # trainer.stopping_criteria[0].lookahead
    log_entry["Batch Size"] = 100  # trainer.optimizer.batch_size
    log_entry["Update Rule"] = "AdamV1"  # trainer.optimizer.update_rules[0].__class__.__name__
    log_entry["Weights Initialization"] = "Uniform"
    log_entry["Training NLL"] = nll_train.get_mean
    log_entry["Training NLL std"] = nll_train.get_std
    log_entry["Validation NLL"] = nll_valid.get_mean
    log_entry["Validation NLL std"] = nll_valid.get_std
    log_entry["Testing NLL"] = nll_test.get_mean
    log_entry["Testing NLL std"] = nll_test.get_std
    log_entry["Training Time"] = None  # trainer.status.training_time
    log_entry["Experiment"] = os.path.abspath(args.experiment)

    formatting = {}
    formatting["Training NLL"] = "{0:.6f}"
    formatting["Training NLL std"] = "{0:.6f}"
    formatting["Validation NLL"] = "{0:.6f}"
    formatting["Validation NLL std"] = "{0:.6f}"
    formatting["Testing NLL"] = "{0:.6f}"
    formatting["Testing NLL std"] = "{0:.6f}"
    #formatting["Training Time"] = "{0:.4f}"

    status = Status()
    with utils.Timer("Evaluating"):
        logging_task = tasks.LogResultCSV("results_{}.csv".format(dataset.name), log_entry, formatting)
        logging_task.execute(status)

        if args.gsheet is not None:
            gsheet_id, gsheet_email, gsheet_password = args.gsheet.split()
            #logging_task = tasks.LogResultGSheet(gsheet_id, gsheet_email, gsheet_password, dataset.name, log_entry, formatting)
            logging_task = tasks.LogResultGSheet(gsheet_id, gsheet_email, gsheet_password, "NADE", log_entry, formatting)
            logging_task.execute(status)

if __name__ == '__main__':
    main()
