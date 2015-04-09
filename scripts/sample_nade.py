#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join as pjoin
import argparse

import numpy as np

from smartpy.misc import utils
from smartpy.models.nade import NADE


def buildArgsParser():
    DESCRIPTION = "Generate samples fomr a NADE model."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('nade', type=str, help='folder where to find a trained NADE model')
    p.add_argument('count', type=int, help='number of samples to generate.')
    p.add_argument('--out', type=str, help='name of the samples file', default="samples")

    # General parameters (optional)
    p.add_argument('--seed', type=int, help='seed used to generate random numbers. Default=1234', default=1234)
    p.add_argument('--view', action='store_true', help="show samples.")

    p.add_argument('-v', '--verbose', action='store_true', help='produce verbose output')
    p.add_argument('-f', '--force',  action='store_true', help='permit overwriting')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    with utils.Timer("Loading model"):
        nade = NADE.create(args.nade)

    with utils.Timer("Sampling ({0} samples)".format(args.count)):
        samples = nade.sample(args.count)

    outfile = pjoin(args.nade, args.out)
    with utils.Timer("Saving {0} samples to '{1}'".format(args.count, outfile)):
        np.save(outfile, samples)

    if args.view:
        import pylab as plt
        from mlpython.misc.utils import show_samples
        show_samples(samples)
        plt.show()

if __name__ == '__main__':
    main()
