from abc import ABCMeta, abstractmethod, abstractproperty


class BatchScheduler(object):
    __metaclass__ = ABCMeta

    def __init__(self, dataset):
        self.dataset = dataset

    @property
    def tasks(self):
        return []

    @abstractproperty
    def updates(self):
        raise NotImplementedError("Subclass of 'BatchScheduler' must implement property 'updates'.")

    @abstractproperty
    def givens(self):
        raise NotImplementedError("Subclass of 'BatchScheduler' must implement property 'givens'.")

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError("Subclass of 'BatchScheduler' must implement '__iter__()'.")
