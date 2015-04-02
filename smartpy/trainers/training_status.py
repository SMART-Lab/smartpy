
class TrainingStatus(object):
    def __init__(self, no_epoch=1, no_update=1):
        self.current_epoch = no_epoch
        self.current_update = no_update
        self.relative_update = 1

        #self.learning_rate_status
