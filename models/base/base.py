import abc
from dataloader.data_utils import *

from utils import Averager, Timer


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.args = set_up_datasets(self.args)
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {'train_loss': [],
                      'val_loss': [],
                      'test_loss': [],
                      'train_acc': [],
                      'val_acc': [],
                      'test_acc': [],
                      'max_acc_epoch': 0,
                      'max_acc': [0.0] * args.sessions}

    @abc.abstractmethod
    def train(self):
        pass
