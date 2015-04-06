#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from os.path import join as pjoin
import json
import argparse
import datetime

import theano.tensor as T
import numpy as np

from smartpy.misc import utils
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

    # Update rules hyperparameters
    utils.create_argument_group_from_hyperparams_registry(p, update_rules.UpdateRule.registry, dest="update_rules", title="Update rules")

    # NADE-like's hyperparameters
    nade = p.add_argument_group("NADE")
    nade.add_argument('--size', type=int, help='number of hidden neurons.', default=20)
    nade.add_argument('--weights_initialization', type=str, help='which type of initialization to use when creating weights [{0}].'.format(", ".join(WEIGHTS_INITIALIZERS)),
                      default=WEIGHTS_INITIALIZERS[0], choices=WEIGHTS_INITIALIZERS)

    # Optimizer hyperparameters
    optimizer = p.add_argument_group("Optimizer")
    optimizer.add_argument('--optimizer', type=str, help='optimizer to use for training: [{0}]'.format(OPTIMIZERS),
                           default=OPTIMIZERS[0], choices=OPTIMIZERS)
    optimizer.add_argument('--batch_size', type=int, help='size of the batch to use when training the model.', default=1)

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

    p.add_argument(dest='experiment', action='store', type=str, help="experiment's directory")


def buildArgsParser():
    DESCRIPTION = "Script to launch/resume unsupervised experiment using Theano."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--max_epochs', type=int, help='maximum number of epochs.')
    p.add_argument('--lookahead', type=int, help='use early stopping with this lookahead.')
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

        hyperparams = {"model": args.model,
                       "dataset": args.dataset,

                       # NADE-like hyperparameters
                       "size": args.size,
                       "weights_initialization": args.weights_initialization,

                       # Update rules hyperparameters
                       #"update_rules": args.update_rules,

                       # Optimizer hyperparameters
                       "optimizer": args.optimizer,
                       "batch_size": args.batch_size,

                       # General parameters
                       "seed": args.seed,
                       }

        # If experiment's name was not given generate one by hashing `hyperparams`.
        if args.name is None:
            uid = utils.generate_uid_from_string(repr(hyperparams))
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

        # Write hyperparams info in a JSON
        json_file = pjoin(data_dir, "hyperparams.json")
        with open(json_file, 'w') as f:
            json.dump(hyperparams, f, sort_keys=True, indent=4, separators=(',', ': '))

    elif args.subcommand == "resume":
        # General parameters (require)
        path = args.experiment

        # Optional parameters
        is_dry = args.is_dry

        pkl = path
        if os.path.isdir(path):
            last_epoch = max([int(f[:-4]) for f in os.listdir(path) if all(map(str.isdigit, f[:-4]))])
            pkl = os.path.join(pkl, str(last_epoch) + ".pkl")

        if not os.path.isfile(pkl):
            parser.error("Path must be either a folder containing a pickled model, or a pickle file!")

        folder = os.path.dirname(pkl)
        data_dir = folder
        out_dir = pjoin(folder, "..")

        # Create model from the json file containing hyperparams values.
        json_file = os.path.join(folder, "hyperparams.json")
        hyperparams = json.load(open(json_file))

        if is_dry:
            print "Model's infos"
            print "-------------"
            print "\n".join(["{0}: {1}".format(k, v) for k, v in hyperparams.items()])
            return

    print "Loading dataset..."
    dataset = Dataset(hyperparams['dataset'])

    print "Building model..."
    nade = models.factory("NADE", input_size=dataset.input_size, hyperparams=hyperparams)

    #no_epoch = 1
    #if args.subcommand == "resume" or not args.is_forcing:
    #    model, no_epoch = mllearners.robust_load(model, data_dir)

    ### Build trainer ###
    optimizer = optimizers.factory(hyperparams["optimizer"], loss=nade.mean_nll_loss, **hyperparams)
    optimizer.add_update_rule(*args.update_rules)

    trainer = Trainer(model=nade, dataset=dataset.trainset, optimizer=optimizer)

    # Add stopping criteria
    if args.max_epochs is not None:
        # Stop when max number of epochs is reached.
        print "Will train {0} for a total of {1} epochs.".format(hyperparams['model'], args.max_epochs)
        trainer.add_stopping_criterion(tasks.MaxEpochStopping(args.max_epochs))

    # Print time for one epoch
    trainer.add_task(tasks.PrintEpochDuration())
    #avg_reconstruction_error = tasks.AverageReconstructionError(model.CD.chain_start, model.CD.chain_end, len(trainset))
    #trainer.add_task(tasks.Print(avg_reconstruction_error, msg="Avg. reconstruction error: {0:.1f}"))

    # if view:
    #     #trainer.add_viewer(viewers.ViewImage(model.W, shape=(28, 28), border=1))
    #     from mlpython.trainers.viewer import LearningViewer, ImageView

    #     filters_view = ImageView(model.W, shape=(28, 28), border=1)
    #     neg_samples_view = ImageView(neg_samples, shape=(28, 28), border=1)

    #     # from mlpython.trainers.viewer import CustomizableTrainingPlot
    #     # plots = {"W": ImageView(model.W, shape=(28, 28), border=1),
    #     #          "Negative samples": ImageView(neg_samples, shape=(28, 28), border=1)}
    #     # CustomizableTrainingPlot(plots, nb_components=2).configure_traits()

    #     viewer = LearningViewer(filters_view, neg_samples_view)
    #     #viewer.watch(trainer)
    #     #viewer.launch()
    #     trainer.add_viewer(viewer)

    trainer.run()

if __name__ == '__main__':
    main()
