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


class Dataset(object):
    def __init__(self, inputs, targets=None, name=None):
        self.name = name
        self.has_target = targets is not None
        self.input_shape = inputs[0].shape
        self.target_shape = targets[0].shape if self.has_target else None
        self._inputs = inputs
        self._targets = targets
        self._ordering = None
        self._inputs_shared = None
        self._targets_shared = None

    @property
    def inputs(self):
        if self._ordering is not None:
            return self._inputs[self._ordering]

        return self._inputs

    @property
    def targets(self):
        if self._targets is not None and self._ordering is not None:
            return self._targets[self._ordering]

        return self._targets

    @property
    def inputs_shared(self):
        if self._inputs_shared is None:
            self._inputs_shared = theano.shared(self.inputs, name=self.name, borrow=True)

        return self._inputs_shared

    @property
    def targets_shared(self):
        if self._targets_shared is None:
            self._targets_shared = theano.shared(self.targets, name=self.name + "_targets", borrow=True)

        return self._targets_shared

    def __len__(self):
        return len(self.inputs)

    def downsample(self, percent_to_keep, rng_seed=None):
        # Validate parameters
        if not (0 < percent_to_keep or percent_to_keep <= 1):
            raise ValueError('percent_to_keep must be in (0,1]')

        self._ordering = None
        self._inputs_shared = None
        self._targets_shared = None

        if percent_to_keep < 1:
            rng = np.random.RandomState(rng_seed)
            self._ordering = rng.choice(np.arange(len(self._inputs)), size=int(len(self._inputs) * percent_to_keep), replace=False)


class MetaDataset(object):
    def __init__(self, name, *datasets):
        self.name = name
        self.has_target = datasets[0].has_target
        self.input_shape = datasets[0].input_shape
        self.target_shape = datasets[0].target_shape
        self.datasets = datasets

    @property
    def trainset(self):
        return self.datasets[0]

    @property
    def validset(self):
        return self.datasets[1]

    @property
    def testset(self):
        return self.datasets[2]

    def downsample(self, percent_to_keep, rng_seed=None):
        for dataset in self.datasets:
            dataset.downsample(percent_to_keep, rng_seed)


def load_unsupervised_dataset(dataset_name):
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

    data = np.load(dataset_npy)
    datasets = [Dataset(data['trainset'].astype(theano.config.floatX), name="trainset"),
                Dataset(data['validset'].astype(theano.config.floatX), name="validset"),
                Dataset(data['testset'].astype(theano.config.floatX), name="testset")]

    return MetaDataset(dataset_name, *datasets)


class UnsupervisedDataset(object):
    def __init__(self, dataset_name):
        # Load dataset from a numpy file
        dataset_npy = os.path.join(os.environ['MLPYTHON_DATASET_REPO'], dataset_name, 'data.npz')
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
            self._trainset_shared = theano.shared(self.trainset, name="trainset", borrow=True)

        return self._trainset_shared

    @property
    def validset_shared(self):
        if self._validset_shared is None:
            self._validset_shared = theano.shared(self.validset, name="validset", borrow=True)

        return self._validset_shared

    @property
    def testset_shared(self):
        if self._testset_shared is None:
            self._testset_shared = theano.shared(self.testset, name="testset", borrow=True)

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
