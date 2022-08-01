""" https://www.kaggle.com/code/medali1992/amex-tabnetclassifier-feature-eng-0-791
"""

import functools
import gc
import typing

import datatable
import numpy
import pandas

from amex.utils import helpers
from amex.utils import paths

logger = helpers.make_logger(__name__)

NON_FEATURES = ['customer_ID', 'S_2', 'target']
CAT_FEATURES = [
    'B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64',
    'D_66', 'D_68',
]
FEATURES_AVG = [
    'B_11', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_2',
    'B_20', 'B_28', 'B_29', 'B_3', 'B_33', 'B_36', 'B_37', 'B_4', 'B_42',
    'B_5', 'B_8', 'B_9', 'D_102', 'D_103', 'D_105', 'D_111', 'D_112', 'D_113',
    'D_115', 'D_118', 'D_119', 'D_121', 'D_124', 'D_128', 'D_129', 'D_131',
    'D_132', 'D_133', 'D_139', 'D_140', 'D_141', 'D_143', 'D_144', 'D_145',
    'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48',
    'D_49', 'D_50', 'D_51', 'D_52', 'D_56', 'D_58', 'D_62', 'D_70', 'D_71',
    'D_72', 'D_74', 'D_75', 'D_79', 'D_81', 'D_83', 'D_84', 'D_88', 'D_91',
    'P_2', 'P_3', 'R_1', 'R_10', 'R_11', 'R_13', 'R_18', 'R_19', 'R_2', 'R_26',
    'R_27', 'R_28', 'R_3', 'S_11', 'S_12', 'S_22', 'S_23', 'S_24', 'S_26',
    'S_27', 'S_5', 'S_7', 'S_8',
]
FEATURES_MIN = [
    'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_19', 'B_2', 'B_20', 'B_22',
    'B_24', 'B_27', 'B_28', 'B_29', 'B_3', 'B_33', 'B_36', 'B_4', 'B_42',
    'B_5', 'B_9', 'D_102', 'D_103', 'D_107', 'D_109', 'D_110', 'D_111',
    'D_112', 'D_113', 'D_115', 'D_118', 'D_119', 'D_121', 'D_122', 'D_128',
    'D_129', 'D_132', 'D_133', 'D_139', 'D_140', 'D_141', 'D_143', 'D_144',
    'D_145', 'D_39', 'D_41', 'D_42', 'D_45', 'D_46', 'D_48', 'D_50', 'D_51',
    'D_53', 'D_54', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_62', 'D_70',
    'D_71', 'D_74', 'D_75', 'D_78', 'D_79', 'D_81', 'D_83', 'D_84', 'D_86',
    'D_88', 'D_96', 'P_2', 'P_3', 'P_4', 'R_1', 'R_11', 'R_13', 'R_17', 'R_19',
    'R_2', 'R_27', 'R_28', 'R_4', 'R_5', 'R_8', 'S_11', 'S_12', 'S_23', 'S_25',
    'S_3', 'S_5', 'S_7', 'S_9',
]
FEATURES_MAX = [
    'B_1', 'B_11', 'B_13', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_2',
    'B_22', 'B_24', 'B_27', 'B_28', 'B_29', 'B_3', 'B_31', 'B_33', 'B_36',
    'B_4', 'B_42', 'B_5', 'B_7', 'B_9', 'D_102', 'D_103', 'D_105', 'D_109',
    'D_110', 'D_112', 'D_113', 'D_115', 'D_121', 'D_124', 'D_128', 'D_129',
    'D_131', 'D_139', 'D_141', 'D_144', 'D_145', 'D_39', 'D_41', 'D_42',
    'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_50', 'D_51', 'D_52',
    'D_53', 'D_56', 'D_58', 'D_59', 'D_60', 'D_62', 'D_70', 'D_72', 'D_74',
    'D_75', 'D_79', 'D_81', 'D_83', 'D_84', 'D_88', 'D_89', 'P_2', 'P_3',
    'R_1', 'R_10', 'R_11', 'R_26', 'R_28', 'R_3', 'R_4', 'R_5', 'R_7', 'R_8',
    'S_11', 'S_12', 'S_23', 'S_25', 'S_26', 'S_27', 'S_3', 'S_5', 'S_7', 'S_8',
]
FEATURES_LAST = [
    'B_1', 'B_11', 'B_12', 'B_13', 'B_14', 'B_16', 'B_18', 'B_19', 'B_2',
    'B_20', 'B_21', 'B_24', 'B_27', 'B_28', 'B_29', 'B_3', 'B_30', 'B_31',
    'B_33', 'B_36', 'B_37', 'B_38', 'B_39', 'B_4', 'B_40', 'B_42', 'B_5',
    'B_8', 'B_9', 'D_102', 'D_105', 'D_106', 'D_107', 'D_108', 'D_110',
    'D_111', 'D_112', 'D_113', 'D_114', 'D_115', 'D_116', 'D_117', 'D_118',
    'D_119', 'D_120', 'D_121', 'D_124', 'D_126', 'D_128', 'D_129', 'D_131',
    'D_132', 'D_133', 'D_137', 'D_138', 'D_139', 'D_140', 'D_141', 'D_142',
    'D_143', 'D_144', 'D_145', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45',
    'D_46', 'D_47', 'D_48', 'D_49', 'D_50', 'D_51', 'D_52', 'D_53', 'D_55',
    'D_56', 'D_59', 'D_60', 'D_62', 'D_63', 'D_64', 'D_66', 'D_68', 'D_70',
    'D_71', 'D_72', 'D_73', 'D_74', 'D_75', 'D_77', 'D_78', 'D_81', 'D_82',
    'D_83', 'D_84', 'D_88', 'D_89', 'D_91', 'D_94', 'D_96', 'P_2', 'P_3',
    'P_4', 'R_1', 'R_10', 'R_11', 'R_12', 'R_13', 'R_16', 'R_17', 'R_18',
    'R_19', 'R_25', 'R_28', 'R_3', 'R_4', 'R_5', 'R_8', 'S_11', 'S_12',
    'S_23', 'S_25', 'S_26', 'S_27', 'S_3', 'S_5', 'S_7', 'S_8', 'S_9',
]


def preprocess(name: typing.Literal['train', 'test']):
    if name == 'train':
        input_path = paths.TRAIN_DATA_PATH
        output_path = paths.TRAIN_FTR_PATH
    else:
        input_path = paths.TEST_DATA_PATH
        output_path = paths.TEST_FTR_PATH

    logger.info(f'Reading raw {name} dataset ...')
    raw_df: pandas.DataFrame = datatable.fread(input_path).to_pandas()
    raw_df.fillna(0, inplace=True)
    cust_id = pandas.Categorical(raw_df.pop('customer_ID'), ordered=True)
    # last_id = (cust_id != numpy.roll(cust_id, -1))  # mask for last statement of every customer

    logger.info(f'Encoding Categorical features in {name} dataset ...')
    cat_df: pandas.DataFrame = (
        pandas
        .get_dummies(
            raw_df[CAT_FEATURES],
            dummy_na=True,
            dtype=numpy.float16,
        )
        .groupby(cust_id)
        .last()
    )
    cat_df = cat_df.rename(columns={f: f'{f}_cat' for f in cat_df.columns.tolist()})

    if name == 'test':
        train = pandas.read_feather(paths.TRAIN_FTR_PATH)
        cat_cols: set[str] = set(train.columns.tolist() + cat_df.columns.tolist())
        cat_cols = {col for col in cat_cols if col.endswith('_cat')}

        missing_in_test = [col for col in cat_cols if col not in cat_df.columns.tolist()]
        for col in missing_in_test:
            cat_df[col] = [0] * len(cat_df.index)

        missing_in_train = [col for col in cat_cols if col not in train.columns.tolist()]
        for col in missing_in_train:
            train[col] = [0] * len(cat_df.index)

        train = train.astype(numpy.float16).reindex(sorted(train.columns), axis=1)
        train.reset_index(drop=True, inplace=True)
        train.to_feather(output_path)
        del train
        gc.collect()

    num_features = [f for f in raw_df.columns.tolist() if f not in CAT_FEATURES + NON_FEATURES]
    raw_num_df = raw_df[num_features].astype(numpy.float16).groupby(cust_id)
    del raw_df
    gc.collect()

    logger.info(f'Computing aggregate numerical features in {name} dataset ...')
    # alpha = 2 / 11
    # ema_df = pandas.DataFrame()
    # for i, col in enumerate(num_features, start=1):
    #     logger.info(f'Computing ema of {col} {i}/{len(num_features)} in {name} dataset ...')
    #     ema_df[f'{col}_ema'] = raw_num_df[col].transform(
    #         lambda vals: functools.reduce(lambda x, y: alpha * x + (1 - alpha) * y, vals, 0)
    #     )
    # logger.info(f'Computed ema features in {name} dataset ...')
    mean_df = raw_num_df.mean().rename(columns={f: f'{f}_mean' for f in num_features})
    logger.info(f'Computed mean features in {name} dataset ...')
    std_df = raw_num_df.std().rename(columns={f: f'{f}_std' for f in num_features}).fillna(0)
    logger.info(f'Computed std features in {name} dataset ...')
    min_df = raw_num_df.min().rename(columns={f: f'{f}_min' for f in num_features})
    logger.info(f'Computed min features in {name} dataset ...')
    max_df = raw_num_df.max().rename(columns={f: f'{f}_max' for f in num_features})
    logger.info(f'Computed max features in {name} dataset ...')
    last_df = raw_num_df.last().rename(columns={f: f'{f}_last' for f in num_features})
    logger.info(f'Computed last features in {name} dataset ...')

    del raw_num_df
    gc.collect()

    logger.info(f'Concatenating engineered features in {name} dataset ...')
    df = pandas.concat([cat_df, mean_df, std_df, min_df, max_df, last_df], axis=1)
    del cat_df, mean_df, std_df, min_df, max_df, last_df
    gc.collect()

    logger.info(f'Saving {name} dataset ...')
    df = df.reindex(sorted(df.columns), axis=1).astype(numpy.float16)
    df.reset_index(drop=True, inplace=True)
    df.to_feather(output_path)
    del df
    gc.collect()

    return


def feature_engineering(force: bool = False):
    if force or (not paths.TRAIN_FTR_PATH.exists()):
        preprocess('train')

    if force or (not paths.TEST_FTR_PATH.exists()):
        preprocess('test')

    train = pandas.read_feather(paths.TRAIN_FTR_PATH).astype(numpy.float16)
    test = pandas.read_feather(paths.TEST_FTR_PATH).astype(numpy.float16)

    # print(train[train.columns[train.isna().any()]].isna().sum())
    # print(test[test.columns[test.isna().any()]].isna().sum())
    print(train.columns[train.isna().any()])
    print(test.columns[test.isna().any()])

    if not paths.TARGET_FTR_PATH.exists():
        target: pandas.DataFrame = datatable.fread(paths.TRAIN_LABELS_PATH).to_pandas()
        # target = target.loc[last, 'target']
        target.pop('customer_ID')
        target.reset_index(drop=True, inplace=True)
        target.to_feather(paths.TARGET_FTR_PATH)
        del target
        gc.collect()
    target = pandas.read_feather(paths.TARGET_FTR_PATH)

    logger.info(f'{train.shape = }, {target.shape = }, {test.shape = }')
    return train, target, test


# def feature_engineering():
#     for i in [0, 1]:
#         if (i == 0 and paths.TRAIN_FTR_PATH.exists()) or (i == 1 and paths.TEST_FTR_PATH.exists()):
#             logger.info(f'Skipping {"train" if i == 0 else "test"} data because the processed ifle already exists ...')
#             continue
#         # i == 0 -> process the train data
#         # i == 1 -> process the test data
#         df: pandas.DataFrame = datatable.fread([paths.TRAIN_DATA_PATH, paths.TEST_DATA_PATH][i]).to_pandas()
#         df = df.astype({col: numpy.float16 for col in FEATURES_AVG + FEATURES_MIN + FEATURES_MAX})
#         cid = pandas.Categorical(df.pop('customer_ID'), ordered=True)
#         last = (cid != numpy.roll(cid, -1))  # mask for last statement of every customer
#
#         logger.info(f'Read {"train" if i == 0 else "test"} data ...')
#         gc.collect()
#
#         df_avg = (
#             df
#             .groupby(cid)
#             .mean()[FEATURES_AVG]
#             .rename(columns={f: f"{f}_avg" for f in FEATURES_AVG})
#         )
#         logger.info(f'Computed avg features for {"train" if i == 0 else "test"} data ...')
#         gc.collect()
#
#         df_max = (
#             df
#             .groupby(cid)
#             .max()[FEATURES_MAX]
#             .rename(columns={f: f"{f}_max" for f in FEATURES_MAX})
#         )
#         logger.info(f'Computed max features for {"train" if i == 0 else "test"} data ...')
#         gc.collect()
#
#         df_min = (
#             df
#             .groupby(cid)
#             .min()[FEATURES_MIN]
#             .rename(columns={f: f"{f}_min" for f in FEATURES_MIN})
#         )
#         logger.info(f'Computed min features for {"train" if i == 0 else "test"} data ...')
#         gc.collect()
#
#         df_last = (
#             df
#             .loc[last, FEATURES_LAST]
#             .rename(columns={f: f"{f}_last" for f in FEATURES_LAST})
#             .set_index(numpy.asarray(cid[last]))
#         )
#         logger.info(f'Computed last features for {"train" if i == 0 else "test"} data ...')
#         del df  # we no longer need the original data
#         gc.collect()
#
#         features_cat_last = [f'{f}_last' for f in CAT_FEATURES]
#         df_categorical = df_last[features_cat_last].astype(object)
#         df_categorical.fillna(value=0, inplace=True)
#
#         if paths.OHE_PATH.exists():
#             with open(paths.OHE_PATH, 'rb') as reader:
#                 ohe = pickle.load(reader)
#         else:
#             ohe = dict()
#
#         for col in df_categorical.columns.tolist():
#             categories = list(sorted(set(df_categorical[col].values.tolist())))
#             mapping = {c: i for i, c in enumerate(categories)}
#             if col in ohe:
#                 ohe[col].update(mapping)
#             else:
#                 ohe[col] = mapping
#             df_categorical[col] = df_categorical[col].apply(lambda v: mapping[v])
#
#         with open(paths.OHE_PATH, 'wb') as writer:
#             pickle.dump(ohe, writer)
#
#         features_not_cat = [f for f in df_last.columns if f not in features_cat_last]
#         df_categorical = pandas.DataFrame(
#             df_categorical.astype(numpy.float16),
#             index=df_categorical.index,
#         ).rename(columns=str)
#         logger.info(f'Computed categorical features for {"train" if i == 0 else "test"} data ...')
#
#         df = pandas.concat([df_last[features_not_cat], df_categorical, df_avg, df_min, df_max], axis=1)
#         df.fillna(value=0, inplace=True)  # Impute missing values
#         del df_avg, df_max, df_min, df_last, df_categorical, cid, last, features_not_cat, ohe
#         gc.collect()
#
#         df.reset_index(drop=True, inplace=True)  # frees 0.2 GByte
#         df = df.astype(numpy.float16)
#         df.to_feather(paths.TRAIN_FTR_PATH if i == 0 else paths.TEST_FTR_PATH)
#         logger.info(f'Saved processed {"train" if i == 0 else "test"} data ...')
#
#         del df
#         gc.collect()
#
#     train = pandas.read_feather(paths.TRAIN_FTR_PATH).astype(numpy.float16)
#     test = pandas.read_feather(paths.TEST_FTR_PATH).astype(numpy.float16)
#
#     if not paths.TARGET_FTR_PATH.exists():
#         target: pandas.DataFrame = datatable.fread(paths.TRAIN_LABELS_PATH).to_pandas()
#         # target = target.loc[last, 'target']
#         target.pop('customer_ID')
#         target.reset_index(drop=True, inplace=True)
#         target.to_feather(paths.TARGET_FTR_PATH)
#         del target
#         gc.collect()
#     target = pandas.read_feather(paths.TARGET_FTR_PATH)
#
#     logger.info(f'{train.shape = }, {target.shape = }, {test.shape = }')
#     return train, target, test
