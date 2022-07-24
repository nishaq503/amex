import logging

from . import constants


def make_logger(name: str):
    logger_ = logging.getLogger(name)
    logger_.setLevel(constants.KAGGLE_LOG)
    return logger_


logger = make_logger(__name__)
