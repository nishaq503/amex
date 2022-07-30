import logging
import os
import random
import typing

import numpy

from . import constants

LogLevels = typing.Literal[
    'NOTSET',
    'DEBUG',
    'INFO',
    'WARN',
    'ERROR',
    'CRITICAL',
]


def make_logger(name: str, level: LogLevels = None):
    logger_ = logging.getLogger(name)
    logger_.setLevel(constants.KAGGLE_LOG if level is None else level)
    return logger_


logger = make_logger(__name__)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    return


__all__ = [
    'make_logger',
    'seed_everything',
]
