import os
import random
import time
# import psutil
import gc

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, StratifiedKFold

import tensorflow
from tabnet.tabnet import TabNetClassifier
# from tabnet.metrics import Metric

from amex.utils import paths
from amex.utils import helpers
from amex.utils import amex_metric

from . import metrics

logger = helpers.make_logger(__name__)


class CFG:
    DEBUG = True
    model = 'tabnet'
    N_folds = 5
    seed = 42


helpers.seed_everything(CFG.seed)


def read_data() -> pd.DataFrame:
    train: pd.DataFrame = pd.read_csv(paths.TRAIN_DATA_PATH)
    train['S_2'] = pd.to_datetime(train['S_2']).astype('datetime64[ns]')

    train = train.groupby('customer_ID').tail(1).reset_index(drop=True)
    train = train.fillna(-1)
    logger.info(f'{train.shape = }')

    target = pd.read_csv(paths.TRAIN_LABELS_PATH)
    logger.info(f'{target.shape = }')

    train = train.merge(target, on='customer_ID')
    logger.info(f'{train.shape = }')

    return train


class AmexTabnet(metrics.Metric):

    def __init__(self):
        super().__init__('amex_tabnet', True)

    def __call__(self, y_true, y_pred, _=None):
        amex = amex_metric.amex_metric_numpy(y_true, y_pred[:, 1])
        return max(amex, 0.)


def run_training(X, y, n_folds: int):
    # X = train[all_features],
    # y = train['target'],
    # nfolds = CFG.N_folds,
    print('\n ', '-' * 50)
    print('\nTraining: ', CFG.model)
    print('\n ', '-' * 50)

    print('\nSeed: ', CFG.seed)
    print('N folds: ', CFG.N_folds)
    print('train shape: ', X.shape)
    print('targets shape: ', y.shape)

    print('\nN features: ', len(all_features))
    print('\n')

    models = list()

    kfold = StratifiedKFold(n_splits=CFG.N_folds, shuffle=True, random_state=CFG.seed)

    for k, (train_idx, valid_idx) in enumerate(kfold.split(X, y)):

        # DEBUG MODE
        if CFG.DEBUG is True:
            if k > 0:
                print('\nDEBUG mode activated: Will train only one fold...\n')
                break

        start = time.perf_counter()

        model = TabNetClassifier(
            feature_columns=all_features,
            num_classes=2,
            feature_dim=64,
            output_dim=32,
            num_decision_steps=3,
            relaxation_factor=1.3,
            sparsity_coefficient=1e-3,
            batch_momentum=0.98,
            # cat_idxs=cat_index,
            # n_independent=2,
            # n_shared=2,
            # clip_value=None,
            # optimizer_fn=torch.optim.Adam,
            # scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
            # scheduler_params={"T_max": 6},
            # mask_type='sparsemax',
            # seed=CFG.seed
        )

        # train
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]
        model.fit(np.array(X_train),
                  np.array(y_train.values.ravel()),
                  eval_set=[(np.array(X_valid), np.array(y_valid.values.ravel()))],
                  max_epochs=50,
                  patience=10,
                  batch_size=2048,
                  eval_metric=['auc', 'accuracy', AmexTabnet])

        models.append(model)

        end = time.perf_counter()
        time_delta = np.round((end - start) / 60, 2)

        print(f'\nFold {k + 1}/{CFG.N_folds} | {time_delta:.2f} min')

        # free memory
        del X_train, y_train
        del X_valid, y_valid
        gc.collect()

    return models


def run():
    train = read_data()
    all_features = [col for col in train.columns if col not in ['target', 'customer_ID', 'S_2']]
    n_features = len(all_features)
    logger.info(f'{n_features = }')

    cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    cat_index = []
    for cat in range(len(cat_features)):
        cat_index.append(train.columns.get_loc(cat_features[cat]))

    logger.info(f'{len(cat_features) = }')

    return
