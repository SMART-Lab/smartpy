import os

import numpy as np
import theano
import theano.tensor as T

from smartlearner import Trainer, Dataset, Model
from smartlearner import tasks
from smartlearner import views
from smartlearner import stopping_criteria
import smartlearner.initializers as initer
from smartlearner.utils import sharedX
from smartlearner.optimizers import SGD
from smartlearner.direction_modifiers import ConstantLearningRate
from smartlearner.batch_schedulers import MiniBatchScheduler
from smartlearner.losses.classification_losses import NegativeLogLikelihood as NLL

DATASETS_ENV = 'DATASETS'


def load_mnist():
    dataset_name = "mnist"
    datasets_repo = os.environ.get(DATASETS_ENV, os.path.join(os.environ["HOME"], '.smartdatasets'))
    if not os.path.isdir(datasets_repo):
        os.mkdir(datasets_repo)

    repo = os.path.join(datasets_repo, dataset_name)
    dataset_npy = os.path.join(repo, 'data.npz')

    if not os.path.isfile(dataset_npy):
        if not os.path.isdir(repo):
            os.mkdir(repo)

        import urllib.request
        urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_train.txt', os.path.join(repo, 'mnist_train.txt'))
        urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_valid.txt', os.path.join(repo, 'mnist_valid.txt'))
        urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_test.txt', os.path.join(repo, 'mnist_test.txt'))

        train_file, valid_file, test_file = [os.path.join(repo, 'mnist_' + ds + '.txt') for ds in ['train', 'valid', 'test']]

        def parse_file(filename):
            return np.array([np.fromstring(l, dtype=np.float32, sep=" ") for l in open(filename)])

        trainset, validset, testset = parse_file(train_file), parse_file(valid_file), parse_file(test_file)
        trainset_inputs, trainset_targets = trainset[:, :-1], trainset[:, [-1]]
        validset_inputs, validset_targets = validset[:, :-1], validset[:, [-1]]
        testset_inputs, testset_targets = testset[:, :-1], testset[:, [-1]]

        np.savez(dataset_npy,
                 trainset_inputs=trainset_inputs, trainset_targets=trainset_targets,
                 validset_inputs=validset_inputs, validset_targets=validset_targets,
                 testset_inputs=testset_inputs, testset_targets=testset_targets)

    data = np.load(dataset_npy)
    trainset = Dataset(data['trainset_inputs'].astype(theano.config.floatX), data['trainset_targets'].astype(theano.config.floatX))
    validset = Dataset(data['validset_inputs'].astype(theano.config.floatX), data['validset_targets'].astype(theano.config.floatX))
    testset = Dataset(data['testset_inputs'].astype(theano.config.floatX), data['testset_targets'].astype(theano.config.floatX))

    return trainset, validset, testset


class Perceptron(Model):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = sharedX(value=np.zeros((input_size, output_size)), name='W', borrow=True)
        self.b = sharedX(value=np.zeros(output_size), name='b', borrow=True)

    def initialize(self, weights_initializer=initer.UniformInitializer(random_seed=1234)):
        weights_initializer(self.W)

    @property
    def updates(self):
        return {}  # No updates.

    @property
    def parameters(self):
        return [self.W, self.b]

    def get_output(self, X):
        preactivation = T.dot(X, self.W) + self.b
        probs = T.nnet.softmax(preactivation)
        return probs

    def use(self, X):
        probs = self.get_output(X)
        return T.argmax(probs, axis=1, keepdims=True)

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass


def test_simple_perceptron():
    #Loading dataset
    trainset, validset, testset = load_mnist()

    #Creating model
    nb_classes = 10
    model = Perceptron(trainset.input_size, nb_classes)
    model.initialize()  # By default, uniform initialization.

    #Building optimizer
    loss = NLL(model, trainset)
    optimizer = SGD(loss=loss)
    optimizer.append_direction_modifier(ConstantLearningRate(0.1))

    # Train for 10 epochs
    batch_scheduler = MiniBatchScheduler(trainset, 100)

    trainer = Trainer(optimizer, batch_scheduler)
    trainer.append_task(stopping_criteria.MaxEpochStopping(10))

    # Print time for one epoch
    trainer.append_task(tasks.PrintEpochDuration())
    trainer.append_task(tasks.PrintTrainingDuration())
    trainer.append_task(tasks.PrintAverageTrainingLoss(loss))

    # Print mean/stderror of classification errors.
    classif_error = views.ClassificationError(model.use, validset)
    trainer.append_task(tasks.Print("Validset - Classif error: {0:.1%} ± {1:.1%}", classif_error.mean, classif_error.stderror))

    trainer.train()
