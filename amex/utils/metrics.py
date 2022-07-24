""" https://www.kaggle.com/code/rohanrao/amex-competition-metric-implementations
"""

import datatable
import numpy
import pandas
import tensorflow


def _top_four_percent_captured(y_true: pandas.DataFrame, y_pred: pandas.DataFrame) -> float:
    df = (pandas.concat([y_true, y_pred], axis='columns')
          .sort_values('prediction', ascending=False))
    df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
    four_pct_cutoff = int(0.04 * df['weight'].sum())
    df['weight_cumsum'] = df['weight'].cumsum()
    df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
    return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()


def _weighted_gini(y_true: pandas.DataFrame, y_pred: pandas.DataFrame) -> float:
    df = (pandas.concat([y_true, y_pred], axis='columns')
          .sort_values('prediction', ascending=False))
    df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
    df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
    total_pos = (df['target'] * df['weight']).sum()
    df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
    df['lorentz'] = df['cum_pos_found'] / total_pos
    df['gini'] = (df['lorentz'] - df['random']) * df['weight']
    return df['gini'].sum()


def _normalized_weighted_gini(y_true: pandas.DataFrame, y_pred: pandas.DataFrame) -> float:
    y_true_pred = y_true.rename(columns={'target': 'prediction'})
    return _weighted_gini(y_true, y_pred) / _weighted_gini(y_true, y_true_pred)


def amex_metric_official(y_true: pandas.DataFrame, y_pred: pandas.DataFrame) -> float:

    g = _normalized_weighted_gini(y_true, y_pred)
    d = _top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


def amex_metric_datatable(y_true: datatable.Frame, y_pred: datatable.Frame) -> float:

    # create datatable frame
    # noinspection PyArgumentList
    df = datatable.Frame(target=y_true, prediction=y_pred)

    # sort by descending prediction values
    # noinspection PyArgumentList
    df = df[:, :, datatable.sort(-datatable.f.prediction)]

    # create row weights, its percentage & cumulative sum
    df['weight'] = 20 - (datatable.f.target * 19)
    df['weight_perc'] = datatable.f.weight / df['weight'].sum1()
    df['weight_perc_cumsum'] = df['weight_perc'].to_numpy().cumsum()  # use native datatable cumsum when v1.1.0 is released

    # filter the top 4%
    four_pct_filter = datatable.f.weight_perc_cumsum <= 0.04

    # default rate captured at 4%
    d = df[four_pct_filter, 'target'].sum1() / df['target'].sum1()

    # weighted Gini coefficient
    df['weighted_target'] = datatable.f.target * datatable.f.weight
    df['weighted_target_perc'] = datatable.f.weighted_target / df['weighted_target'].sum1()
    df['lorentz'] = df['weighted_target_perc'].to_numpy().cumsum()  # use native datatable cumsum when v1.1.0 is released
    df['gini'] = (datatable.f.lorentz - datatable.f.weight_perc_cumsum) * datatable.f.weight
    gini = df['gini'].sum1()

    # max weighted Gini coefficient
    total_pos = df['target'].sum1()
    total_neg = df.nrows - total_pos
    gini_max = 10 * total_neg * (total_pos + 20 * total_neg - 19) / (total_pos + 20 * total_neg)

    # normalized weighted Gini coefficient
    g = gini / gini_max

    # amex metric
    m = 0.5 * (g + d)

    return m


def amex_metric_numpy(y_true: numpy.array, y_pred: numpy.array) -> float:

    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting by decreasing prediction values
    indices = numpy.argsort(y_pred)[::-1]
    preds, target = y_pred[indices], y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = target[four_pct_filter].sum() / n_pos

    # weighted Gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted Gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted Gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)


def amex_metric_tensorflow(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor) -> float:

    # convert dtypes to float64
    y_true = tensorflow.cast(y_true, dtype=tensorflow.float64)
    y_pred = tensorflow.cast(y_pred, dtype=tensorflow.float64)

    # count of positives and negatives
    n_pos = tensorflow.math.reduce_sum(y_true)
    n_neg = tensorflow.cast(tensorflow.shape(y_true)[0], dtype=tensorflow.float64) - n_pos

    # sorting by decreasing prediction values
    indices = tensorflow.argsort(y_pred, axis=0, direction='DESCENDING')
    preds, target = tensorflow.gather(y_pred, indices), tensorflow.gather(y_true, indices)

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = tensorflow.cumsum(weight / tensorflow.reduce_sum(weight))
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = tensorflow.reduce_sum(target[four_pct_filter]) / n_pos

    # weighted Gini coefficient
    lorentz = tensorflow.cumsum(target / n_pos)
    gini = tensorflow.reduce_sum((lorentz - cum_norm_weight) * weight)

    # max weighted Gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted Gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)


__all__ = [
    'amex_metric_official',
    'amex_metric_datatable',
    'amex_metric_numpy',
    'amex_metric_tensorflow',
]
