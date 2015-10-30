import numpy as np

import theano.tensor as T

from smartlearner.utils import sharedX


class Dataset(object):
    """ Dataset interface.

    Attributes
    ----------
    symb_inputs : `theano.tensor.TensorType` object
        Symbolic variables representing the inputs.
    symb_targets : `theano.tensor.TensorType` object or None
        Symbolic variables representing the targets.

    Notes
    -----
    `symb_inputs` and `symb_targets` have test value already tagged to them. Use
    THEANO_FLAGS="compute_test_value=warn" to use them.
    """
    def __init__(self, inputs, targets=None, name="dataset", keep_on_cpu=False):
        """
        Parameters
        ----------
        inputs : ndarray
            Training examples
        targets : ndarray (optional)
            Target for each training example.
        name : str (optional)
            The name of the dataset is used to name Theano variables. Default: 'dataset'.
        """
        self.name = name
        self.keep_on_cpu = keep_on_cpu
        self.inputs = inputs
        self.targets = targets
        self.symb_inputs = T.TensorVariable(type=T.TensorType("floatX", [False]*self.inputs.ndim),
                                            name=self.name+'_symb_inputs')
        self.symb_inputs.tag.test_value = self.inputs.get_value()  # For debugging Theano graphs.

        self.symb_targets = None
        if self.has_targets:
            self.symb_targets = T.TensorVariable(type=T.TensorType("floatX", [False]*self.targets.ndim),
                                                 name=self.name+'_symb_targets')
            self.symb_targets.tag.test_value = self.targets.get_value()  # For debugging Theano graphs.

    @property
    def inputs(self):
        return self._inputs_shared

    @inputs.setter
    def inputs(self, value):
        self._inputs_shared = sharedX(value, name=self.name+"_inputs", keep_on_cpu=self.keep_on_cpu)

    @property
    def targets(self):
        return self._targets_shared

    @targets.setter
    def targets(self, value):
        if value is not None:
            self._targets_shared = sharedX(np.array(value), name=self.name+"_targets", keep_on_cpu=self.keep_on_cpu)
        else:
            self._targets_shared = None

    @property
    def has_targets(self):
        return self.targets is not None

    @property
    def input_shape(self):
        return self.inputs.get_value().shape[1:]

    @property
    def target_shape(self):
        if self.has_targets:
            return self.targets.get_value().shape[1:]

        return None

    @property
    def input_size(self):
        # TODO: is this property really useful? If needed one could just call directly `dataset.input_shape[-1]`.
        return self.input_shape[-1]

    @property
    def target_size(self):
        # TODO: is this property really useful? If needed one could just call directly `dataset.target_shape[-1]`.
        if self.has_targets:
            return self.target_shape[-1]

        return None

    def __len__(self):
        return len(self.inputs.get_value())

    def derive_dataset(self, inputs, targets=None, name_suffix=None):
        """ Used to create derived dataset, such as a validset from a training set.

        """
        new_name = self.name + name_suffix if name_suffix is not None else '_derived'
        base = Dataset(inputs, targets, name=new_name, keep_on_cpu=self.keep_on_cpu)
        base.symb_inputs = self.symb_inputs
        base.symb_targets = self.symb_targets

        return base
