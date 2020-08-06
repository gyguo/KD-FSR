import shutil
import os
import sys
import errno
import numpy as np
import torch
import random


def rm(path):
  try:
    shutil.rmtree(path)
  except OSError as e:
    if e.errno != errno.ENOENT:
      raise


def mkdir(path):
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise


class AttrDict(dict):
    """
    Subclass dict and define getter-setter.
    This behaves as both dict and obj.
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


class Logger(object):
    def __init__(self,filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename,'a')

    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)