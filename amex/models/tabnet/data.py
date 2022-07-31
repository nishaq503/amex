""" https://www.kaggle.com/code/medali1992/amex-tabnetclassifier-feature-eng-0-791
"""

import gc
import pickle

import datatable
import numpy
import pandas
from sklearn.preprocessing import OneHotEncoder

from amex.utils import helpers
from amex.utils import paths

logger = helpers.make_logger(__name__)

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
FEATURES_CAT = [
    'B_30_last', 'B_38_last', 'D_114_last', 'D_116_last', 'D_117_last', 'D_120_last',
    'D_126_last', 'D_63_last', 'D_64_last', 'D_66_last', 'D_68_last',
]


def feature_engineering():

    for i in [0, 1]:
        if (i == 0 and paths.TRAIN_FTR_PATH.exists()) or (i == 1 and paths.TEST_FTR_PATH.exists()):
            continue
        # i == 0 -> process the train data
        # i == 1 -> process the test data
        df: pandas.DataFrame = datatable.fread([paths.TRAIN_DATA_PATH, paths.TEST_DATA_PATH][i]).to_pandas()
        cid = pandas.Categorical(df.pop('customer_ID'), ordered=True)
        last = (cid != numpy.roll(cid, -1))  # mask for last statement of every customer
        if i == 0:  # train
            target = df.loc[last, 'target']
            target = target.reset_index(drop=True)
            target.to_feather(paths.TARGET_FTR_PATH)

        logger.info(f'Read {"train" if i == 0 else "test"} data ...')
        gc.collect()

        df_avg = (df
                  .groupby(cid)
                  .mean()[FEATURES_AVG]
                  .rename(columns={f: f"{f}_avg" for f in FEATURES_AVG}))
        logger.info(f'Computed avg features for {"train" if i == 0 else "test"} data ...')
        gc.collect()

        df_max = (df
                  .groupby(cid)
                  .max()[FEATURES_MAX]
                  .rename(columns={f: f"{f}_max" for f in FEATURES_MAX}))
        logger.info(f'Computed max features for {"train" if i == 0 else "test"} data ...')
        gc.collect()

        df_min = (df
                  .groupby(cid)
                  .min()[FEATURES_MIN]
                  .rename(columns={f: f"{f}_min" for f in FEATURES_MIN}))
        logger.info(f'Computed min features for {"train" if i == 0 else "test"} data ...')
        gc.collect()

        df_last = (df.loc[last, FEATURES_LAST]
                   .rename(columns={f: f"{f}_last" for f in FEATURES_LAST})
                   .set_index(numpy.asarray(cid[last])))
        logger.info(f'Computed last features for {"train" if i == 0 else "test"} data ...')
        del df  # we no longer need the original data
        gc.collect()

        df_categorical = df_last[FEATURES_CAT].astype(object)
        features_not_cat = [f for f in df_last.columns if f not in FEATURES_CAT]
        if i == 0:  # train
            ohe = OneHotEncoder(drop='first', sparse=False, dtype=numpy.float32, handle_unknown='ignore')
            ohe.fit(df_categorical)
            with open(paths.WORKING_DIR.joinpath('ohe.pickle'), 'wb') as writer:
                pickle.dump(ohe, writer)
        else:
            with open(paths.WORKING_DIR.joinpath('ohe.pickle'), 'rb') as reader:
                ohe = pickle.load(reader)
        df_categorical = pandas.DataFrame(
            ohe.transform(df_categorical).astype(numpy.float16),
            index=df_categorical.index,
        ).rename(columns=str)
        logger.info(f'Computed categorical features for {"train" if i == 0 else "test"} data ...')

        df = pandas.concat([df_last[features_not_cat], df_categorical, df_avg, df_min, df_max], axis=1)
        df.fillna(value=0, inplace=True)  # Impute missing values
        del df_avg, df_max, df_min, df_last, df_categorical, cid, last, features_not_cat, ohe
        gc.collect()

        if i == 0:  # train
            # Free the memory
            df.reset_index(drop=True, inplace=True)  # frees 0.2 GByte
            df.to_feather(paths.TRAIN_FTR_PATH)
        else:
            df.to_feather(paths.TEST_FTR_PATH)

        del df
        gc.collect()

    train = pandas.read_feather(paths.TRAIN_FTR_PATH)
    target = pandas.read_feather(paths.TARGET_FTR_PATH)
    test = pandas.read_feather(paths.TEST_FTR_PATH)

    logger.info(f'{train.shape = }, {target.shape = }, {test.shape = }')
    return train, target, test
