# MLPython datasets wrapper
import os
import theano
import numpy as np
import mlpython.mlproblems.generic as mlpb
import mlpython.datasets.store as mlstore


# List of supported datasets
DATASETS = ['adult',
            'binarized_mnist',
            'connect4',
            'dna',
            'mushrooms',
            'nips',
            'ocr_letters',
            'rcv1',
            'rcv2_russ',
            'web']


class UnsupervisedDataset(object):
    def __init__(self, dataset_name):
        self.datasets = {}

        #Temporary patch until we build the dataset manager
        dataset_npy = os.path.join(os.environ['MLPYTHON_DATASET_REPO'], dataset_name, 'data.npz')
        if not os.path.isfile(dataset_npy):
            if not os.path.isdir(os.path.join(os.environ['MLPYTHON_DATASET_REPO'], dataset_name)):
                mlstore.download(dataset_name)

            if dataset_name in mlstore.classification_names:
                trainset, validset, testset = mlstore.get_classification_problem(dataset_name)
                trainset, validset, testset = mlpb.SubsetFieldsProblem(trainset), mlpb.SubsetFieldsProblem(validset), mlpb.SubsetFieldsProblem(testset)
            elif dataset_name in mlstore.distribution_names:
                trainset, validset, testset = mlstore.get_distribution_problem(dataset_name)
            else:
                print "Not supported type of dataset!"
                return

            trainset, validset, testset = np.array([x for x in trainset]), np.array([x for x in validset]), np.array([x for x in testset])
            np.savez(dataset_npy, trainset=trainset, validset=validset, testset=testset)

        # Load dataset from a numpy file
        self.name = dataset_name
        self.datasets = np.load(dataset_npy)
        self._trainset = self.datasets['trainset'].astype(theano.config.floatX)
        self._validset = self.datasets['validset'].astype(theano.config.floatX)
        self._testset = self.datasets['testset'].astype(theano.config.floatX)
        self._trainset_idx = None
        self._validset_idx = None
        self._testset_idx = None
        self._trainset_shared = None
        self._validset_shared = None
        self._testset_shared = None

        self.input_size = len(self._trainset[0])

    @property
    def trainset(self):
        if self._trainset_idx is not None:
            return self._trainset[self._trainset_idx]
        else:
            return self._trainset

    @property
    def validset(self):
        if self._validset_idx is not None:
            return self._validset[self._validset_idx]
        else:
            return self._validset

    @property
    def testset(self):
        if self._testset_idx is not None:
            return self._testset[self._testset_idx]
        else:
            return self._testset

    @property
    def trainset_shared(self):
        if self._trainset_shared is None:
            theano.shared(self.trainset, name="trainset", borrow=True)

        return self._trainset_shared

    @property
    def validset_shared(self):
        if self._validset_shared is None:
            theano.shared(self.validset, name="validset", borrow=True)

        return self._validset_shared

    @property
    def testset_shared(self):
        if self._testset_shared is None:
            theano.shared(self.testset, name="testset", borrow=True)

        return self._testset_shared

    def downsample(self, percent_to_keep, rng_seed=None):
        # Validate parameters
        if not (0 < percent_to_keep or percent_to_keep <= 1):
            raise ValueError('percent_to_keep must be in (0,1]')

        if percent_to_keep == 1:
            self._trainset_idx = None
            self._validset_idx = None
            self._testset_idx = None

        rng = np.random.RandomState(rng_seed)
        self._trainset_idx = rng.choice(np.arange(len(self.trainset)), size=int(len(self.trainset) * percent_to_keep), replace=False)
        self._validset_idx = rng.choice(np.arange(len(self.validset)), size=int(len(self.validset) * percent_to_keep), replace=False)
        self._testset_idx = rng.choice(np.arange(len(self.testset)), size=int(len(self.testset) * percent_to_keep), replace=False)

        self._trainset_shared = None
        self._validset_shared = None
        self._testset_shared = None