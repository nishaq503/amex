""" Adapted from:
https://github.com/dreamquark-ai/tabnet/blob/develop/pytorch_tabnet/metrics.py
"""
import abc

import numpy
import tensorflow
from tensorflow.python import keras
from tensorflow.python.keras import metrics

from amex.utils import amex_metric
from amex.utils import constants


class Metric(keras.metrics.Metric):

    def __init__(self, name: str, maximize: bool):
        super().__init__(name=name)
        self.__name = name
        self.__maximize = maximize

    @property
    def name(self) -> str:
        return self.__name

    @property
    def maximize(self) -> bool:
        return self.__maximize

    @abc.abstractmethod
    def __call__(self, y_true, y_pred, obf_vars):
        pass

    # @classmethod
    # def get_metrics_by_names(cls: 'Metric', names: list):
    #     """ Get list of metric classes.
    #
    #     Args:
    #         cls: Metric class.
    #         names: list of metric names.
    #
    #     Returns:
    #         list of metric classes.
    #     """
    #     available_metrics = cls.__subclasses__()
    #     available_names = [metric().name for metric in available_metrics]
    #     metrics = list()
    #     for name in names:
    #         assert name in available_names, f"{name} is not available, choose in {available_names}"
    #         idx = available_names.index(name)
    #         metric = available_metrics[idx]()
    #         metrics.append(metric)
    #     return metrics


class AmexTabnet(Metric):

    def __init__(self):
        super().__init__('amex_tabnet', True)

    def __call__(self, y_true, y_pred, _=None):
        amex = amex_metric.amex_metric_numpy(y_true, y_pred[:, 1])
        return max(amex, 0.)

    def _serialize_to_tensors(self):
        raise NotImplementedError

    def _restore_from_tensors(self, restored_tensors):
        raise NotImplementedError

    def update_state(self, *args, **kwargs):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError


class UnsupervisedMetric(Metric):

    def __init__(self):
        super().__init__('unsup_loss', False)

    def __call__(
            self,
            y_pred: tensorflow.Tensor,
            embedded_x: tensorflow.Tensor,
            obf_vars: tensorflow.Tensor,
    ):
        """ Compute MSE (Mean Squared Error) of predictions.

        Implements an unsupervised loss function. This differs from original
        paper as it's scaled to be batch-size independent and number of features
        reconstructed independent (by taking the mean)

        Args:
            y_pred: Reconstructed prediction (with embeddings)
            embedded_x: Original input embedded by network
            obf_vars: Binary mask for obfuscated variables. 1 means the variable
             was obfuscated so reconstruction is based on this.

        Returns:
            unsupervised loss, average value over batch samples.
        """

        errors = y_pred - embedded_x
        reconstruction_errors = tensorflow.multiply(errors, obf_vars) ** 2
        batch_means = tensorflow.reduce_mean(embedded_x, axis=0)
        batch_means[batch_means == 0] = 1

        batch_stds = tensorflow.math.reduce_std(embedded_x, axis=0) ** 2
        batch_stds[batch_stds == 0] = batch_means[batch_stds == 0]
        features_loss = tensorflow.matmul(reconstruction_errors, 1 / batch_stds)
        # compute the number of obfuscated variables to reconstruct
        nb_reconstructed_variables = tensorflow.reduce_mean(obf_vars, axis=1)
        # take the mean of the reconstructed variable errors
        features_loss = features_loss / (nb_reconstructed_variables + constants.EPSILON)
        # here we take the mean per batch, contrary to the paper
        loss = tensorflow.reduce_mean(features_loss)

        return loss.item()

    def _serialize_to_tensors(self):
        raise NotImplementedError

    def _restore_from_tensors(self, restored_tensors):
        raise NotImplementedError

    def update_state(self, *args, **kwargs):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError


class UnsupervisedNumpyMetric(Metric):

    def __init__(self):
        super().__init__('unsup_loss_numpy', False)

    def __call__(
            self,
            y_pred: numpy.ndarray,
            embedded_x: numpy.ndarray,
            obf_vars: numpy.ndarray,
    ):
        """ Compute MSE (Mean Squared Error) of predictions.

        Args:
            y_pred: Reconstructed prediction (with embeddings)
            embedded_x: Original input embedded by network
            obf_vars: Binary mask for obfuscated variables. 1 means the variable
             was obfuscated so reconstruction is based on this.

        Returns:
            unsupervised loss, average value over batch samples.
        """
        errors = y_pred - embedded_x
        reconstruction_errors = numpy.multiply(errors, obf_vars) ** 2
        batch_means = numpy.mean(embedded_x, axis=0)
        batch_means = numpy.where(batch_means == 0, 1, batch_means)

        batch_stds = numpy.std(embedded_x, axis=0, ddof=1) ** 2
        batch_stds = numpy.where(batch_stds == 0, batch_means, batch_stds)
        features_loss = numpy.matmul(reconstruction_errors, 1 / batch_stds)
        # compute the number of obfuscated variables to reconstruct
        nb_reconstructed_variables = numpy.sum(obf_vars, axis=1)
        # take the mean of the reconstructed variable errors
        features_loss = features_loss / (nb_reconstructed_variables + constants.EPSILON)
        # here we take the mean per batch, contrary to the paper
        loss = numpy.mean(features_loss)

        return loss

    def _serialize_to_tensors(self):
        raise NotImplementedError

    def _restore_from_tensors(self, restored_tensors):
        raise NotImplementedError

    def update_state(self, *args, **kwargs):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError


# def check_metrics(metrics):
#     """ Check if custom metrics are provided.
#
#     Args:
#         metrics : list of built-in metrics (str) or custom metrics (classes).
#
#     Returns:
#         list of metric names.
#     """
#     val_metrics = []
#     for metric in metrics:
#         if isinstance(metric, str):
#             val_metrics.append(metric)
#         elif issubclass(metric, Metric):
#             val_metrics.append(metric().name)
#         else:
#             raise TypeError("You need to provide a valid metric format")
#     return val_metrics


# @dataclass
# class UnsupMetricContainer:
#     """ Container holding a list of metrics.
#     """
#
#     metric_names: list[str]
#     prefix: str = ""
#
#     def __post_init__(self):
#         self.metrics = Metric.get_metrics_by_names(self.metric_names)
#         self.names = [self.prefix + name for name in self.metric_names]
#
#     def __call__(self, y_pred, embedded_x, obf_vars):
#         """Compute all metrics and store into a dict.
#
#         Args:
#             y_pred: torch.Tensor or np.array
#                 Reconstructed prediction (with embeddings)
#             embedded_x: torch.Tensor
#                 Original input embedded by network
#             obf_vars: torch.Tensor
#                 Binary mask for obfuscated variables.
#                 1 means the variables was obfuscated so reconstruction is based on this.
#
#         Returns:
#             dict of metrics metric_name -> metric_value.
#         """
#         logs = {}
#         for metric in self.metrics:
#             res = metric(y_pred, embedded_x, obf_vars)
#             logs[self.prefix + metric.name] = res
#         return logs


# @dataclass
# class MetricContainer:
#     """ Container holding a list of metrics.
#     """
#
#     metric_names: list[str]
#     prefix: str = ""
#
#     def __post_init__(self):
#         self.metrics = Metric.get_metrics_by_names(self.metric_names)
#         self.names = [self.prefix + name for name in self.metric_names]
#
#     def __call__(self, y_true: numpy.ndarray, y_pred: numpy.ndarray):
#         """ Compute all metrics and store into a dict.
#
#         Args:
#         y_true: Target matrix or vector
#         y_pred: Score matrix or vector
#
#         Returns:
#             dict of metrics metric_name -> metric_value.
#         """
#         logs = {}
#         for metric in self.metrics:
#             if isinstance(y_pred, list):
#                 res = numpy.mean(
#                     [metric(y_true[:, i], y_pred[i]) for i in range(len(y_pred))]
#                 )
#             else:
#                 res = metric(y_true, y_pred)
#             logs[self.prefix + metric.name] = res
#         return logs
