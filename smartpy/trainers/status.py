from os.path import join as pjoin
from smartpy.misc.utils import save_dict_to_json_file, load_dict_from_json_file


class Status(object):
    def __init__(self, starting_epoch=0, starting_update=0):
        self.current_epoch = starting_epoch
        self.current_update = starting_update
        self.relative_update = 1
        self.extra = {}

    def save(self, savedir="./"):
        state = {}
        state['current_epoch'] = self.current_epoch
        state['current_update'] = self.current_update
        state['relative_update'] = self.relative_update
        state['extra'] = self.extra
        save_dict_to_json_file(pjoin(savedir, "status.json"), state)

    def load(self, loaddir="./"):
        state = load_dict_from_json_file(pjoin(loaddir, "status.json"))
        self.current_epoch = state['current_epoch']
        self.current_update = state['current_update']
        self.relative_update = state['relative_update']
        self.extra = state['extra']
