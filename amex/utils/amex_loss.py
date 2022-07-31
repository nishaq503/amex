import datatable
import numpy
import pandas
import tensorflow
import torch

from . import amex_metric


def amex_loss_official(y_true: pandas.DataFrame, y_pred: pandas.DataFrame) -> float:
    return 1 - amex_metric.amex_metric_official(y_true, y_pred)


def amex_loss_datatable(y_true: datatable.Frame, y_pred: datatable.Frame) -> datatable.float32:
    return 1 - amex_metric.amex_metric_datatable(y_true, y_pred)


def amex_loss_numpy(y_true: numpy.array, y_pred: numpy.array) -> numpy.float32:
    return 1 - amex_metric.amex_metric_numpy(y_true, y_pred)


def amex_loss_tensorflow(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor) -> tensorflow.float32:
    return 1 - amex_metric.amex_metric_tensorflow(y_true, y_pred)


def amex_loss_pytorch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.float32:
    return 1 - amex_metric.amex_metric_pytorch(y_true, y_pred)
