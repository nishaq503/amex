import logging
import os
import random

import numpy

from . import constants


def make_logger(name: str):
    logger_ = logging.getLogger(name)
    logger_.setLevel(constants.KAGGLE_LOG)
    return logger_


logger = make_logger(__name__)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    return

