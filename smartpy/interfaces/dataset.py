import theano


class Dataset(object):
    def __init__(self, inputs, targets, name="dataset"):
        self.name = name
        self.inputs = inputs
        self.targets = targets
        self.symb_inputs = theano.tensor.matrix(name=self.name)
        self.symb_targets = None if targets is None else theano.tensor.matrix(name=self.name+'_target')

    @property
    def inputs(self):
        return self._inputs_shared

    @inputs.setter
    def inputs(self, value):
        self._inputs_shared = theano.shared(value, name=self.name + "_inputs", borrow=True)

    @property
    def targets(self):
        return self._targets_shared

    @targets.setter
    def targets(self, value):
        if value is not None:
            self._targets_shared = theano.shared(value, name=self.name + "_targets", borrow=True)
        else:
            self._targets_shared = None

    @property
    def input_size(self):
        return len(self.inputs.get_value()[0])

    @property
    def target_size(self):
        if self.targets is None:
            return 0
        else:
            return len(self.targets.get_value()[0])

    def __len__(self):
        return len(self.inputs.get_value())


class SyntheticDataset(Dataset):
    def __init__(self, generator, generate_targets=True, name='SyntheticDataset', **kwargs):
        inputs, targets = generator.generate_dataset(generate_targets, **kwargs)
        super().__init__(inputs, targets, name)
