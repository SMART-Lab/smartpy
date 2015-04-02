class Regularization():
    def __init__(self, decay):
        self.decay = decay

    def __call__(self, param):
        raise NameError('Should be implemented by inheriting class!')
