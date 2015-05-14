#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join as pjoin
import argparse

import numpy as np

from smartpy.misc import utils
from smartpy.models.nade import NADE

from smartpy.misc.dataset import UnsupervisedDataset as Dataset


def buildArgsParser():
    DESCRIPTION = "Generate samples from a NADE model."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('nade', type=str, help='folder where to find a trained NADE model')
    p.add_argument('count', type=int, help='number of samples to generate.')
    p.add_argument('--out', type=str, help='name of the samples file')
    p.add_argument('--type', type=str, help="type of sampling: 'uniform', 'conditional' or 'independent'.", default="uniform")
    p.add_argument('--alpha', type=float, help='Ratio of input units to condition on.')
    p.add_argument('--dataset', type=str, help='name of dataset for conditional sampling', default="binarized_mnist")

    # General parameters (optional)
    p.add_argument('--seed', type=int, help='seed used to generate random numbers.')
    p.add_argument('--view', action='store_true', help="show samples.")

    p.add_argument('-v', '--verbose', action='store_true', help='produce verbose output')
    p.add_argument('-f', '--force',  action='store_true', help='permit overwriting')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    with utils.Timer("Loading model"):
        nade = NADE.create(args.nade)

    if args.type == "conditional":
        with utils.Timer("Loading dataset"):
            dataset = Dataset(args.dataset)

            rng = np.random.RandomState(args.seed)
            idx = np.arange(len(dataset.validset))
            rng.shuffle(idx)
            examples = dataset.validset[idx[:args.count]]

        with utils.Timer("Generating {} conditional samples from NADE with alpha={}".format(len(examples), args.alpha)):
            sample_conditionally = nade.build_conditional_sampling_function(seed=args.seed)
            samples = sample_conditionally(examples, alpha=args.alpha)

    elif args.type == "independent":
        with utils.Timer("Loading dataset"):
            dataset = Dataset(args.dataset)

            rng = np.random.RandomState(args.seed)
            idx = np.arange(len(dataset.validset))
            rng.shuffle(idx)
            examples = dataset.validset[idx[:args.count]]

        with utils.Timer("Generating {} independent samples".format(len(examples))):
            sample_independently = nade.build_independent_sampling_function(seed=args.seed)
            samples = sample_independently(examples)

    else:
        with utils.Timer("Generating {} samples from NADE".format(args.count)):
            sample = nade.build_sampling_function(seed=args.seed)
            samples = sample(args.count)

    if args.out is not None:
        outfile = pjoin(args.nade, args.out)
        with utils.Timer("Saving {0} samples to '{1}'".format(args.count, outfile)):
            np.save(outfile, samples)

    if args.view:
        import pylab as plt
        from mlpython.misc.utils import show_samples

        if args.type == "conditional":
            show_samples(examples, title="Examples")
            show_samples(samples, title="'Conditional' samples with alpha={}".format(args.alpha))
        elif args.type == "independent":
            show_samples(examples, title="Examples")
            show_samples(samples, title="'Independent'")
        else:
            show_samples(samples, title="'Uniform' samples")

        plt.show()

if __name__ == '__main__':
    main()
