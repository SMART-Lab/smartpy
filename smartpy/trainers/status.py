
class Status(object):
    def __init__(self, starting_epoch=1, starting_update=1):
        self.current_epoch = starting_epoch
        self.current_update = starting_update
        self.relative_update = 1

        #self.learning_rate_status
