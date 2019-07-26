import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

class precision_and_recall(object):
    def __init__(self, args):
        # parameters
        self.args = args
        self.data_dir = args.data_dir
        self.result_dir = args.result_dir
        self.batch_size = args.batch_size
        self.cpu = args.cpu
        self.seed = args.seed

    def run(self):
        return


class realism(object):
    def __init__(self, args):
        # parameters
        self.args = args
        self.data_dir = args.data_dir
        self.result_dir = args.result_dir
        self.batch_size = args.batch_size
        self.cpu = args.cpu
        self.seed = args.seed   

    def run(self):
        return