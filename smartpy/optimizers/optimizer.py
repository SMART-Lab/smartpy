class Optimizer(object):
    def __init__(self, loss, update_rules=[]):
        self.loss = loss
        self.update_rules = update_rules
        self.nb_updates_per_epoch = 1

    def add_update_rule(self, *update_rules):
        self.update_rules += update_rules
