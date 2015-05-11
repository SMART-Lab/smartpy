#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from os.path import join as pjoin
import json
import argparse
import datetime

import pickle
import theano
import theano.tensor as T
import numpy as np

from smartpy.misc import utils
from smartpy.misc.utils import save_dict_to_json_file, load_dict_from_json_file, sharedX
from smartpy.misc.dataset import load_unsupervised_dataset, Dataset

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
    denade = p.add_argument_group("Denoising NADE")
    denade.add_argument('nade', type=str, help='folder where to find an already trained NADE model')
    denade.add_argument('--sampling', metavar="N", type=int, help='sampling will be done at N epoch (Default: only once at the beginning)', default=0)
    denade.add_argument('--size', type=int, help='number of hidden neurons.')
    denade.add_argument('--hidden_activation', type=str, help="Activation functions: {}".format(ACTIVATION_FUNCTIONS.keys()), choices=ACTIVATION_FUNCTIONS.keys())
    denade.add_argument('--alpha', type=float, help="ratio of input units to condition on.", default=0.5)
    denade.add_argument('--weights_initialization', type=str, help='which type of initialization to use when creating weights [{0}].'.format(", ".join(WEIGHTS_INITIALIZERS)), choices=WEIGHTS_INITIALIZERS, default=WEIGHTS_INITIALIZERS[0])
    denade.add_argument('--ordering_seed', type=int, help='if provided, pixel will be shuffling using this random seed.')
    denade.add_argument('--noise_weight', type=float, help="weight of noise's NLL.", default=1.)

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
    trainer.add_argument('--lookahead_eps', type=float, help='in early stopping, an improvement is whenever the objective improve of at least `eps`.', default=1e-3)

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

    with utils.Timer("Building model"):
        from smartpy.models.nade import NADE
        denade = NADE.create(args.nade)

    with utils.Timer("Loading dataset"):
        dataset = load_unsupervised_dataset(args.dataset)

    with utils.Timer("Augmenting dataset"):
        trainset = dataset.trainset
        sample_file = pjoin(data_dir, "samples_{}.npz".format(args.seed))
        if not os.path.isfile(sample_file):
            sample_conditionally = denade.build_conditional_sampling_function(seed=args.seed)
            samples = sample_conditionally(trainset.inputs, alpha=args.alpha)
            np.savez(sample_file, samples=samples, targets=trainset.inputs)
        else:
            samples = np.load(sample_file)["samples"]

        inputs = np.zeros((2*len(trainset), trainset.input_shape[0]), dtype=theano.config.floatX)
        targets = np.zeros((2*len(trainset), trainset.input_shape[0]), dtype=theano.config.floatX)
        inputs[::2] = trainset.inputs
        targets[::2] = trainset.inputs
        inputs[1::2] = samples[::-1]
        targets[1::2] = trainset.inputs[::-1]
        augmented_trainset = Dataset(name="augmented_trainset", inputs=inputs, targets=targets)

        # import pylab as plt
        # from mlpython.misc.utils import show_samples
        # show_samples(trainset.inputs, title="Examples")
        # show_samples(samples, title="Conditional samples with alpha={}".format(args.alpha))
        # plt.show()

    with utils.Timer("Building loss function"):
        def mean_nll_loss(input, target):
            # Weigh noise's nll
            nll = denade.get_nll(input, target)
            nll *= args.noise_weight * (T.sum(abs(target-input), axis=1) > 0)
            return nll.mean()

    with utils.Timer("Building optimizer"):
        optimizer = optimizers.factory(args.optimizer, loss=mean_nll_loss, **vars(args))
        if args.update_rules is not None:
            optimizer.add_update_rule(*args.update_rules)
        else:
            command_nade = pickle.load(open(pjoin(args.nade, "command.pkl")))
            lr_nade = float(command_nade[command_nade.index("--ADAGRAD") + 1])
            from smartpy.update_rules import ADAGRAD
            optimizer.add_update_rule(ADAGRAD(lr=lr_nade))

    with utils.Timer("Building trainer"):
        trainer = Trainer(model=denade, datasets=[augmented_trainset.inputs_shared, augmented_trainset.targets_shared], optimizer=optimizer)

        # Print time for one epoch
        trainer.add_task(tasks.PrintEpochDuration())
        nll_valid = tasks.EvaluateNLL(denade.get_nll, [dataset.validset.inputs_shared]*2, batch_size=100)
        trainer.add_task(tasks.Print(nll_valid.mean, msg="Average NLL on the validset: {0}"))
        from smartpy.trainers import Status
        nade_valid_nll = nll_valid.mean.view(Status())
        print "NADE - Validation NLL: {:.6f}".format(nade_valid_nll)

        # Add stopping criteria
        if args.max_epoch is not None:
            # Stop when max number of epochs is reached.
            print "Will train {0} for a total of {1} epochs.".format(args.model, args.max_epoch)
            trainer.add_stopping_criterion(tasks.MaxEpochStopping(args.max_epoch))

        # Do early stopping bywatching the average NLL on the validset.
        if args.lookahead is not None:
            print "Will train {0} using early stopping with a lookahead of {1} epochs.".format(args.model, args.lookahead)
            save_task = tasks.SaveTraining(trainer, savedir=data_dir)
            early_stopping = tasks.EarlyStopping(nll_valid.mean, args.lookahead, save_task, eps=args.lookahead_eps, skip_epoch0=True)
            trainer.add_stopping_criterion(early_stopping)
            trainer.add_task(early_stopping)

        # Add a task to save the whole training process
        if args.save_frequency < np.inf:
            save_task = tasks.SaveTraining(trainer, savedir=data_dir, each_epoch=args.save_frequency)
            trainer.add_task(save_task)

        if args.subcommand == "resume":
            print "Loading existing trainer..."
            trainer.load(data_dir)

    with utils.Timer("Training"):
        trainer.run()
        trainer.status.save(savedir=data_dir)

        if not args.lookahead:
            trainer.save(savedir=data_dir)

    with utils.Timer("Reporting"):
        # Evaluate model on train, valid and test sets
        nll_train_augmented = tasks.EvaluateNLL(denade.get_nll, [augmented_trainset.inputs_shared, augmented_trainset.targets_shared], batch_size=100)
        nll_train = tasks.EvaluateNLL(denade.get_nll, [dataset.trainset.inputs_shared]*2, batch_size=100)
        nll_valid = tasks.EvaluateNLL(denade.get_nll, [dataset.validset.inputs_shared]*2, batch_size=100)
        nll_test = tasks.EvaluateNLL(denade.get_nll, [dataset.testset.inputs_shared]*2, batch_size=100)

        command_nade = pickle.load(open(pjoin(args.nade, "command.pkl")))
        lr_nade = float(command_nade[command_nade.index("--ADAGRAD") + 1])  # TOFIX

        from collections import OrderedDict
        log_entry = OrderedDict()
        log_entry["Learning Rate"] = trainer.optimizer.update_rules[0].lr
        log_entry["Learning Rate NADE"] = lr_nade
        log_entry["Hidden Size"] = denade.hyperparams["hidden_size"]
        log_entry["Activation Function"] = denade.hyperparams["hidden_activation"]
        log_entry["Noise Weight"] = args.noise_weight
        log_entry["alpha"] = args.alpha
        log_entry["Sampling"] = int(args.sampling)
        log_entry["Initialization Seed"] = args.seed
        log_entry["Ordering Seed"] = denade.hyperparams["ordering_seed"]
        log_entry["Tied Weights"] = denade.hyperparams["tied_weights"]
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
        log_entry["Augmented Training NLL"] = nll_train_augmented.mean
        log_entry["Augmented Training NLL std"] = nll_train_augmented.std
        log_entry["Training NLL"] = nll_train.mean
        log_entry["Training NLL std"] = nll_train.std
        log_entry["Validation NLL"] = nll_valid.mean
        log_entry["Validation NLL std"] = nll_valid.std
        log_entry["Testing NLL"] = nll_test.mean
        log_entry["Testing NLL std"] = nll_test.std
        log_entry["Training Time"] = trainer.status.training_time
        log_entry["Experiment"] = os.path.abspath(data_dir)
        log_entry["NADE"] = args.nade
        log_entry["NADE Validation NLL"] = nade_valid_nll

        formatting = {}
        formatting["Augmented Training NLL"] = "{:.6f}"
        formatting["Augmented Training NLL std"] = "{:.6f}"
        formatting["Training NLL"] = "{:.6f}"
        formatting["Training NLL std"] = "{:.6f}"
        formatting["Validation NLL"] = "{:.6f}"
        formatting["Validation NLL std"] = "{:.6f}"
        formatting["Testing NLL"] = "{:.6f}"
        formatting["Testing NLL std"] = "{:.6f}"
        formatting["Training Time"] = "{:.4f}"
        formatting["NADE Validation NLL"] = "{:.6f}"

        from smartpy.trainers import Status
        status = Status()
        logging_task = tasks.LogResultCSV("results_{}_{}.csv".format("DeNADE", dataset.name), log_entry, formatting)
        logging_task.execute(status)

        if args.gsheet is not None:
            gsheet_id, gsheet_email, gsheet_password = args.gsheet.split()
            logging_task = tasks.LogResultGSheet(gsheet_id, gsheet_email, gsheet_password, "DeNADE", log_entry, formatting)
            logging_task.execute(status)


if __name__ == '__main__':
    main()
