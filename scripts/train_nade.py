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

from smartpy.trainers.trainer import Trainer
from smartpy.trainers import tasks


DATASETS = ['binarized_mnist']
MODELS = ['nade']


def build_launch_experiment_argsparser(subparser):
    DESCRIPTION = "Train a NADE model on a specific dataset using Theano."

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
    nade = p.add_argument_group("NADE")
    nade.add_argument('--size', type=int, help='number of hidden neurons.', default=20)
    nade.add_argument('--weights_initialization', type=str, help='which type of initialization to use when creating weights [{0}].'.format(", ".join(WEIGHTS_INITIALIZERS)),
                      default=WEIGHTS_INITIALIZERS[0], choices=WEIGHTS_INITIALIZERS)

    # Update rules hyperparameters
    utils.create_argument_group_from_hyperparams_registry(p, update_rules.UpdateRule.registry, dest="update_rules", title="Update rules")

    # Optimizer hyperparameters
    optimizer = p.add_argument_group("Optimizer")
    optimizer.add_argument('--optimizer', type=str, help='optimizer to use for training: [{0}]'.format(OPTIMIZERS),
                           default=OPTIMIZERS[0], choices=OPTIMIZERS)
    optimizer.add_argument('--batch_size', type=int, help='size of the batch to use when training the model.', default=1)

    # Trainer parameters
    trainer = p.add_argument_group("Trainer")
    trainer.add_argument('--max_epochs', type=int, help='maximum number of epochs.')
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
    p.add_argument('--no_cloud', action='store_true', help="disable cloud reporting (using gspread).")
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
        #open(pjoin(data_dir, "command.txt"), 'w').write(launch_command)
        pickle.dump(sys.argv[sys.argv.index('launch'):], open(pjoin(data_dir, "command.pkl"), 'w'))

    elif args.subcommand == "resume":
        if not os.path.isdir(args.experiment):
            parser.error("Cannot find specified experiment folder: '{}'".format(args.experiment))

        # Load command to resume
        data_dir = args.experiment
        launch_command = pickle.load(open(pjoin(args.experiment, "command.pkl")))
        command_to_resume = sys.argv[1:sys.argv.index('resume')] + launch_command
        #command_to_resume += open(pjoin(args.experiment, "command.txt")).read()
        print command_to_resume
        args = parser.parse_args(command_to_resume)

        args.subcommand = "resume"

    print "Loading dataset..."
    dataset = Dataset(args.dataset)

    # TODO: remove, only there for debugging purpose.
    dataset.downsample(0.1, rng_seed=1234)

    print "Building model..."
    nade = models.factory("NADE", input_size=dataset.input_size, hyperparams=vars(args))

    from smartpy.misc import weights_initializer
    weights_initialization_method = weights_initializer.factory(**vars(args))
    nade.initialize(weights_initialization_method)

    ### Build trainer ###
    optimizer = optimizers.factory(args.optimizer, loss=nade.mean_nll_loss, **vars(args))
    optimizer.add_update_rule(*args.update_rules)

    trainer = Trainer(model=nade, dataset=dataset.trainset, optimizer=optimizer)

    # Add stopping criteria
    if args.max_epochs is not None:
        # Stop when max number of epochs is reached.
        print "Will train {0} for a total of {1} epochs.".format(args.model, args.max_epochs)
        trainer.add_stopping_criterion(tasks.MaxEpochStopping(args.max_epochs))

    # Print time for one epoch
    trainer.add_task(tasks.PrintEpochDuration())

    avg_nll_on_valid = tasks.AverageNLL(nade.get_nll, dataset.validset)
    trainer.add_task(tasks.Print(avg_nll_on_valid, msg="Average NLL on the valiset: {0}"))

    save_model_task = tasks.SaveModel(nade, data_dir)

    # Do early stopping bywatching the average NLL on the validset.
    if args.lookahead is not None:
        print "Will train {0} using early stopping with a lookahead of {1} epochs.".format(args.model, args.lookahead)
        early_stopping = tasks.EarlyStopping(avg_nll_on_valid, args.lookahead, save_model_task, eps=10)
        trainer.add_stopping_criterion(early_stopping)
        trainer.add_task(early_stopping)

    # Add a task to save the whole training process
    if args.save_frequency < np.inf:
        save_task = tasks.SaveTraining(trainer, savedir=data_dir, save_frequency=args.save_frequency)
        trainer.add_task(save_task)

    if args.subcommand == "resume":
        print "Loading existing trainer..."
        trainer.load(data_dir)

    trainer.run()

if __name__ == '__main__':
    main()
