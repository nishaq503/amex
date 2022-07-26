import logging
import random
import time

import datatable
import pandas
import tensorflow
import torch

from amex.utils import amex_metric
from amex.utils import helpers
from amex.utils import paths

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = helpers.make_logger(__name__)


def main():

    logger.info(f'Reading dataframes ...')

    train_labels: pandas.DataFrame = pandas.read_csv(paths.TRAIN_LABELS_PATH, index_col='customer_ID')
    # noinspection PyTypeChecker
    predictions = pandas.DataFrame({'prediction': (train_labels['target'] / 2 + [random.random() for _ in range(train_labels.shape[0])]) / 2})

    time_official = 0.
    time_datatable = 0.
    time_numpy = 0.
    time_tensorflow = 0.
    time_torch = 0.

    num_runs = 10

    for i in range(num_runs):
        logger.info(f'Starting run {i + 1}/{num_runs} ...')

        start_official = time.perf_counter()
        _ = amex_metric.amex_metric_official(train_labels, predictions)
        time_official += time.perf_counter() - start_official

        start_datatable = time.perf_counter()
        _ = amex_metric.amex_metric_datatable(datatable.Frame(train_labels), datatable.Frame(predictions))
        time_datatable += time.perf_counter() - start_datatable

        start_numpy = time.perf_counter()
        _ = amex_metric.amex_metric_numpy(train_labels.to_numpy().ravel(), predictions.to_numpy().ravel())
        time_numpy = time.perf_counter() - start_numpy

        start_tensorflow = time.perf_counter()
        _ = amex_metric.amex_metric_tensorflow(tensorflow.convert_to_tensor(train_labels), tensorflow.convert_to_tensor(predictions))
        time_tensorflow = time.perf_counter() - start_tensorflow

        start_torch = time.perf_counter()
        _ = amex_metric.amex_metric_pytorch(torch.tensor(train_labels.values).ravel(), torch.tensor(predictions.values).ravel())
        time_torch = time.perf_counter() - start_torch

    time_official /= num_runs
    time_datatable /= num_runs
    time_numpy /= num_runs
    time_tensorflow /= num_runs
    time_torch /= num_runs

    logger.info(f'Pandas (Official): {time_official:02.12f} seconds.')
    logger.info(f'Datatable:         {time_datatable:02.12f} seconds.')
    logger.info(f'Numpy:             {time_numpy:02.12f} seconds.')
    logger.info(f'TensorFlow:        {time_tensorflow:02.12f} seconds.')
    logger.info(f'Torch:             {time_torch:02.12f} seconds.')

    logger.info(f'Done!')

    return


if __name__ == '__main__':
    main()
    #             Mac,            Linux
    # Official,   0.420962229000,
    # Datatable,  0.059550395900,
    # Numpy,      0.004801500000,
    # TensorFlow, 0.005497016600,
    # Torch,      0.005006804100,
