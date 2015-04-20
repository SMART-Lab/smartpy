#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from os.path import join as pjoin
import json
import argparse
import datetime

import pickle
import theano.tensor as T
import numpy as np

from smartpy.misc import utils
from smartpy.misc.utils import save_dict_to_json_file, load_dict_from_json_file
from smartpy.misc.dataset import UnsupervisedDataset as Dataset

from smartpy import models
from smartpy import optimizers

from smartpy import update_rules
from smartpy.optimizers import OPTIMIZERS
from smartpy.misc.weights_initializer import WEIGHTS_INITIALIZERS
from smartpy.misc.utils import ACTIVATION_FUNCTIONS

from smartpy.trainers.trainer import Trainer
from smartpy.trainers import tasks


DATASETS = ['binarized_mnist']
MODELS = ['nested_nade']
WEIGHTS_INITIALIZERS = ["NADE"] + WEIGHTS_INITIALIZERS


def build_launch_experiment_argsparser(subparser):
    DESCRIPTION = "Train a Nested NADE model on a specific dataset using Theano."

    p = subparser.add_parser("launch",
                             description=DESCRIPTION,
                             help=DESCRIPTION,
                             #formatter_class=argparse.ArgumentDefaultsHelpFormatter
                             )

    # General parameters (required)
    p.add_argument('--dataset', type=str, help='dataset to use [{0}].'.format(', '.join(DATASETS)),
                   default=DATASETS[0], choices=DATASETS)
    p.add_argument('--model', type=str, help='unsupervised model to use [{0}]'.format(', '.join(MODELS)),
                   default=MODELS[0], choices=MODELS)

    # NADE-like's hyperparameters
    nested_nade = p.add_argument_group("Nested NADE")
    nested_nade.add_argument('nade', type=str, help='folder where to find an already trained NADE model')
    nested_nade.add_argument('--sampling', metavar="N", type=int, help='sampling will be done at N epoch (Default: only once at the beginning)', default=0)
    nested_nade.add_argument('--size', type=int, help='number of hidden neurons.')
    nested_nade.add_argument('--hidden_activation', type=str, help="Activation functions: {}".format(ACTIVATION_FUNCTIONS.keys()), choices=ACTIVATION_FUNCTIONS.keys())
    nested_nade.add_argument('--gamma', type=float, help="'tradeoff' between nll loss and noise-contrastive loss.", default=1.)
    nested_nade.add_argument('--noise_lambda', type=float, help='weight on the noise term in the noise-contrastive loss.', default=1.)
    nested_nade.add_argument('--weights_initialization', type=str, help='which type of initialization to use when creating weights [{0}].'.format(", ".join(WEIGHTS_INITIALIZERS)), choices=WEIGHTS_INITIALIZERS, default=WEIGHTS_INITIALIZERS[0])
    #nested_nade.add_argument('--use_nade_weights', action='store_true', help='initialize NN-NADE with weights of NADE')

    # Update rules hyperparameters
    utils.create_argument_group_from_hyperparams_registry(p, update_rules.UpdateRule.registry, dest="update_rules", title="Update rules")

    # Optimizer hyperparameters
    optimizer = p.add_argument_group("Optimizer")
    optimizer.add_argument('--optimizer', type=str, help='optimizer to use for training: [{0}]'.format(OPTIMIZERS),
                           default=OPTIMIZERS[0], choices=OPTIMIZERS)
    optimizer.add_argument('--batch_size', type=int, help='size of the batch to use when training the model.', default=1)

    # Trainer parameters
    trainer = p.add_argument_group("Trainer")
    trainer.add_argument('--max_epoch', type=int, help='maximum number of epochs.')
    trainer.add_argument('--lookahead', type=int, help='use early stopping with this lookahead.')

    # General parameters (optional)
    p.add_argument('--name', type=str, help='name of the experiment.')
    p.add_argument('--out', type=str, help='directory that will contain the experiment.', default="./")
    p.add_argument('--seed', type=int, help='seed used to generate random numbers. Default=1234', default=1234)

    p.add_argument('-v', '--verbose', action='store_true', help='produce verbose output')
    p.add_argument('-f', '--force',  action='store_true', help='permit overwriting')


def build_resume_experiment_argsparser(subparser):
    DESCRIPTION = 'Resume a specific experiment.'

    p = subparser.add_parser("resume",
                             description=DESCRIPTION,
                             help=DESCRIPTION,
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument(dest='experiment', type=str, help="experiment's directory")


def buildArgsParser():
    DESCRIPTION = "Script to launch/resume unsupervised experiment using Theano."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--keep', dest='save_frequency', action='store', type=int, help='save model every N epochs. Default=once finished', default=np.inf)
    p.add_argument('--report', dest='report_frequency', action='store', type=int, help="report results every N epochs. Default=once finished", default=np.inf)
    p.add_argument('--gsheet', type=str, metavar="SHEET_ID EMAIL PASSWORD", help="log results into a Google's Spreadsheet.")
    p.add_argument('--view', action='store_true', help="show filters during training.")
    p.add_argument('--dry', action='store_true', help='only print folder used and quit')

    subparser = p.add_subparsers(title="subcommands", metavar="", dest="subcommand")
    build_launch_experiment_argsparser(subparser)
    build_resume_experiment_argsparser(subparser)
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    print args

    if args.subcommand == "launch":
        out_dir = os.path.abspath(args.out)
        if not os.path.isdir(out_dir):
            parser.error('"{0}" must be an existing folder!'.format(out_dir))

        launch_command = " ".join(sys.argv[sys.argv.index('launch'):])

        # If experiment's name was not given generate one by hashing `launch_command`.
        if args.name is None:
            uid = utils.generate_uid_from_string(launch_command)
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            args.name = current_time + "__" + uid

        data_dir = pjoin(out_dir, args.name)
        if args.dry:
            print "Would use:\n" if os.path.isdir(data_dir) else "Would create:\n", data_dir
            return

        if os.path.isdir(data_dir):
            print "Using:\n", data_dir
        else:
            os.mkdir(data_dir)
            print "Creating:\n", data_dir

        # Save launched command to txt file
        pickle.dump(sys.argv[sys.argv.index('launch'):], open(pjoin(data_dir, "command.pkl"), 'w'))

    elif args.subcommand == "resume":
        if not os.path.isdir(args.experiment):
            parser.error("Cannot find specified experiment folder: '{}'".format(args.experiment))

        # Load command to resume
        data_dir = args.experiment
        launch_command = pickle.load(open(pjoin(args.experiment, "command.pkl")))
        command_to_resume = sys.argv[1:sys.argv.index('resume')] + launch_command
        args = parser.parse_args(command_to_resume)

        args.subcommand = "resume"

    print "Loading dataset..."
    dataset = Dataset(args.dataset)

    print "Building model..."
    nested_nade = models.factory("NestedNADE", input_size=dataset.input_size, hyperparams=vars(args))

    sample_from_nade = nested_nade.trained_nade.build_sampling_function(args.seed)
    sampling_task = tasks.Sampling(sample_from_nade,
                                   nb_samples=len(dataset.trainset),
                                   shape=(nested_nade.input_size, ),
                                   each_epoch=args.sampling)

    if args.weights_initialization == 'NADE':
        nested_nade.W.set_value(nested_nade.trained_nade.W.get_value())
        nested_nade.bhid.set_value(nested_nade.trained_nade.bhid.get_value())
        nested_nade.bvis.set_value(nested_nade.trained_nade.bvis.get_value())
        nested_nade.V.set_value(nested_nade.trained_nade.V.get_value())
    else:
        from smartpy.misc import weights_initializer
        weights_initialization_method = weights_initializer.factory(**vars(args))
        nested_nade.initialize(weights_initialization_method)

    ### Build trainer ###
    optimizer = optimizers.factory(args.optimizer, loss=nested_nade.loss, **vars(args))
    optimizer.add_update_rule(*args.update_rules)

    trainer = Trainer(model=nested_nade, datasets=[dataset.trainset, sampling_task.samples], optimizer=optimizer)

    # Print time for one epoch
    trainer.add_task(tasks.PrintEpochDuration())
    nll_valid = tasks.EvaluateNLL(nested_nade.get_nll, dataset.validset, batch_size=100)
    trainer.add_task(tasks.Print(nll_valid.mean, msg="Average NLL on the validset: {0}"))
    trainer.add_task(sampling_task)

    #noise_term_mean = trainer.track_variable(nested_nade.noise_term_mean, shape=np.float64(0).shape)
    #alike_term_mean = trainer.track_variable(nested_nade.alike_term_mean, shape=np.float64(0).shape)
    #trainer.add_task(tasks.PrintSharedVariable([alike_term_mean, noise_term_mean], msg="Term: {0}\t{1}", each_update=10, each_epoch=0))

    # Add stopping criteria
    if args.max_epoch is not None:
        # Stop when max number of epochs is reached.
        print "Will train {0} for a total of {1} epochs.".format(args.model, args.max_epoch)
        trainer.add_stopping_criterion(tasks.MaxEpochStopping(args.max_epoch))

    # Do early stopping bywatching the average NLL on the validset.
    if args.lookahead is not None:
        print "Will train {0} using early stopping with a lookahead of {1} epochs.".format(args.model, args.lookahead)
        save_task = tasks.SaveTraining(trainer, savedir=data_dir)
        early_stopping = tasks.EarlyStopping(nll_valid.mean, args.lookahead, save_task, eps=1e-3)
        trainer.add_stopping_criterion(early_stopping)
        trainer.add_task(early_stopping)

    # Add a task to save the whole training process
    if args.save_frequency < np.inf:
        save_task = tasks.SaveTraining(trainer, savedir=data_dir, each_epoch=args.save_frequency)
        trainer.add_task(save_task)

    if args.subcommand == "resume":
        print "Loading existing trainer..."
        trainer.load(data_dir)

    trainer.run()
    trainer.status.save(savedir=data_dir)

    if not args.lookahead:
        trainer.save(savedir=data_dir)

    # Evaluate model on train, valid and test sets
    nll_train = tasks.EvaluateNLL(nested_nade.get_nll, dataset.trainset, batch_size=100)
    nll_valid = tasks.EvaluateNLL(nested_nade.get_nll, dataset.validset, batch_size=100)
    nll_test = tasks.EvaluateNLL(nested_nade.get_nll, dataset.testset, batch_size=100)

    command_nade = pickle.load(open(pjoin(args.nade, "command.pkl")))
    lr_nade = float(command_nade[command_nade.index("--AdamV1") + 1])  # TOFIX

    from collections import OrderedDict
    log_entry = OrderedDict()
    log_entry["Learning Rate"] = trainer.optimizer.update_rules[0].lr
    log_entry["Learning Rate NADE"] = lr_nade
    log_entry["Hidden Size"] = nested_nade.hyperparams["hidden_size"]
    log_entry["Activation Function"] = nested_nade.hyperparams["hidden_activation"]
    log_entry["Gamma"] = nested_nade.hyperparams["gamma"]
    log_entry["Noise Lambda"] = nested_nade.hyperparams["noise_lambda"]
    log_entry["Sampling"] = int(args.sampling)
    log_entry["Initialization Seed"] = args.seed
    log_entry["Ordering Seed"] = args.ordering_seed
    log_entry["Tied Weights"] = nested_nade.hyperparams["tied_weights"]
    log_entry["Best Epoch"] = trainer.status.extra["best_epoch"] if args.lookahead else trainer.status.current_epoch
    log_entry["Max Epoch"] = trainer.stopping_criteria[0].nb_epochs_max if args.max_epoch else ''

    if args.max_epoch:
        log_entry["Look Ahead"] = trainer.stopping_criteria[1].lookahead if args.lookahead else ''
        log_entry["Look Ahead eps"] = trainer.stopping_criteria[1].eps if args.lookahead else ''
    else:
        log_entry["Look Ahead"] = trainer.stopping_criteria[0].lookahead if args.lookahead else ''
        log_entry["Look Ahead eps"] = trainer.stopping_criteria[0].eps if args.lookahead else ''

    log_entry["Batch Size"] = trainer.optimizer.batch_size
    log_entry["Update Rule"] = trainer.optimizer.update_rules[0].__class__.__name__
    log_entry["Weights Initialization"] = args.weights_initialization
    log_entry["Training NLL"] = nll_train.mean
    log_entry["Training NLL std"] = nll_train.std
    log_entry["Validation NLL"] = nll_valid.mean
    log_entry["Validation NLL std"] = nll_valid.std
    log_entry["Testing NLL"] = nll_test.mean
    log_entry["Testing NLL std"] = nll_test.std
    log_entry["Training Time"] = trainer.status.training_time
    log_entry["Experiment"] = os.path.abspath(data_dir)
    log_entry["NADE"] = args.nade

    formatting = {}
    formatting["Training NLL"] = "{:.6f}"
    formatting["Training NLL std"] = "{:.6f}"
    formatting["Validation NLL"] = "{:.6f}"
    formatting["Validation NLL std"] = "{:.6f}"
    formatting["Testing NLL"] = "{:.6f}"
    formatting["Testing NLL std"] = "{:.6f}"
    formatting["Training Time"] = "{:.4f}"

    from smartpy.trainers import Status
    status = Status()
    with utils.Timer("Evaluating"):
        logging_task = tasks.LogResultCSV("results_{}_{}.csv".format("NestedNADE", dataset.name), log_entry, formatting)
        logging_task.execute(status)

        if args.gsheet is not None:
            gsheet_id, gsheet_email, gsheet_password = args.gsheet.split()
            logging_task = tasks.LogResultGSheet(gsheet_id, gsheet_email, gsheet_password, "NestedNADE", log_entry, formatting)
            logging_task.execute(status)


if __name__ == '__main__':
    main()
