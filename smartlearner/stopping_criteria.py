import numpy as np

from .interfaces import Task


class TrainingExit(Exception):
    def __init__(self, status):
        self.status = status

    def __str__(self):
        return "Training exited with \n" + repr(self.status)


class MaxEpochStopping(Task):
    def __init__(self, nb_epochs_max):
        super().__init__()
        self.nb_epochs_max = nb_epochs_max

    def post_epoch(self, status):
        if status.current_epoch >= self.nb_epochs_max:
            raise TrainingExit(status)


class EarlyStopping(Task):
    """ Stops the training if it's been a while without any improvement.

    At the end of each epoch the cost function is evaluated. If there is no
    improvement for a given number of epochs, we stop the training and the best
    model is restored.

    Parameters
    ----------
    cost : `smartlearner.tasks.views.View` object
        The cost that will be evaluated after each epoch.
    lookahead : int
        Number of epochs allowed without any improvement.
    eps : float (optional)
        This allows to ignore small improvement. To be considered as improvement a
        cost will need to be lower than the best cost minus this epsilon. Default: 0.
    min_nb_epochs : int (optional)
        Start using early stopping after that many epochs. Default: 0.
    callback : callable (optional)
        If provided, this will be called each time there is improvement with this
        instance of `EarlyStopping` and the current `smartlearner.status.Status` object.
    """
    def __init__(self, cost, lookahead, eps=0., min_nb_epochs=0, callback=None):
        super().__init__()

        self.cost = cost
        self.best_epoch = 0
        self.best_cost = np.inf

        self.lookahead = lookahead
        self.callback = callback
        self.eps = eps
        self.min_nb_epochs = min_nb_epochs

    def post_epoch(self, status):
        cost = float(self.cost.view(status))
        if cost + self.eps < self.best_cost:
            self.best_epoch = status.current_epoch
            self.best_cost = cost

            self._stash_model(status)

            if self.callback is not None:
                self.callback(self, status)

        if status.current_epoch - self.best_epoch >= self.lookahead and status.current_epoch >= self.min_nb_epochs:
            raise TrainingExit(status)

    def finished(self, status):
        self._restore_model(status)
        print("Early Stopping : training finished with best cost {:.3f} at epoch {}.".format(self.best_cost, self.best_epoch))

    def _stash_model(self, status):
        model = status.trainer._optimizer.loss.model
        self.stash = [p.get_value() for p in model.parameters]

    def _restore_model(self, status):
        model = status.trainer._optimizer.loss.model
        for param, value in zip(model.parameters, self.stash):
            param.set_value(value)
