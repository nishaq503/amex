import logging
import random
import time

import datatable
import pandas
import tensorflow

from amex.utils import helpers
from amex.utils import metrics
from amex.utils import paths

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = helpers.make_logger(__name__)

logger.info(f'Reading dataframes ...')

train_data: pandas.DataFrame = pandas.read_csv(
    paths.TRAIN_DATA_PATH,
    index_col='customer_ID',
    usecols=['customer_ID', 'P_2'],
)

train_labels: pandas.DataFrame = pandas.read_csv(paths.TRAIN_LABELS_PATH, index_col='customer_ID')
# noinspection PyTypeChecker
predictions = pandas.DataFrame({'prediction': (train_labels['target'] / 2 + [random.random() for i in range(train_labels.shape[0])]) / 2})

time_official = 0.
time_datatable = 0.
time_numpy = 0.
time_tensorflow = 0.

for i in range(10):
    start_official = time.perf_counter()
    metric_official = metrics.amex_metric_official(train_labels, predictions)
    time_official += time.perf_counter() - start_official

    start_datatable = time.perf_counter()
    metric_datatable = metrics.amex_metric_datatable(datatable.Frame(train_labels), datatable.Frame(predictions))
    time_datatable += time.perf_counter() - start_datatable

    start_numpy = time.perf_counter()
    metric_numpy = metrics.amex_metric_numpy(train_labels.to_numpy().ravel(), predictions.to_numpy().ravel())
    time_numpy = time.perf_counter() - start_numpy

    start_tensorflow = time.perf_counter()
    metric_tensorflow = metrics.amex_metric_tensorflow(tensorflow.convert_to_tensor(train_labels), tensorflow.convert_to_tensor(predictions))
    time_tensorflow = time.perf_counter() - start_tensorflow

logger.info(f'Pandas (Official): {time_official:02.12f} seconds.')
logger.info(f'Datatable:         {time_datatable:02.12f} seconds.')
logger.info(f'Numpy:             {time_numpy:02.12f} seconds.')
logger.info(f'TensorFlow:        {time_tensorflow:02.12f} seconds.')

logger.info(f'Done!')
