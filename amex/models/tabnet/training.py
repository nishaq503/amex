""" https://www.kaggle.com/code/medali1992/amex-tabnetclassifier-feature-eng-0-791
"""

import gc
import time

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold

from amex.utils import amex_metric
from amex.utils import helpers
from . import data


class AmexTabnet(Metric):

    def __init__(self):
        self._name = 'amex_tabnet'
        self._maximize = True

    def __call__(self, y_true, y_pred):
        amex = amex_metric.amex_metric_numpy(y_true, y_pred[:, 1])
        return max(amex, 0.)


def run_training(cfg):
    train_df, target_df, test_df = data.feature_engineering(inference=True)

    helpers.seed_everything(seed=cfg.seed)
    print('\n ', '-' * 50)
    print('\nTraining: ', cfg.model)
    print('\n ', '-' * 50)

    print('\nSeed: ', cfg.seed)
    print('N folds: ', cfg.N_folds)
    print('train shape: ', train_df.shape)
    print('targets shape: ', target_df.shape)

    print('\nN features: ', len(train_df.columns.values.tolist()))
    print('\n')

    # Create out of folds array
    oof_predictions = np.zeros((train_df.shape[0]))
    test_predictions = np.zeros(test_df.shape[0])
    feature_importances = pd.DataFrame()
    feature_importances["feature"] = train_df.columns.tolist()
    stats = pd.DataFrame()
    explain_matrices = []
    masks_ = []

    kfold = StratifiedKFold(n_splits=cfg.N_folds, shuffle=True, random_state=cfg.seed)

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_df, target_df)):

        # DEBUG MODE
        if cfg.DEBUG is True:
            if fold > 0:
                print('\nDEBUG mode activated: Will train only one fold...\n')
                break

        start = time.time()

        train_x, train_y = train_df.loc[train_idx], target_df.loc[train_idx]
        valid_x, valid_y = train_df.loc[valid_idx], target_df.loc[valid_idx]

        model = TabNetClassifier(
            n_d=32,
            n_a=32,
            n_steps=3,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            momentum=0.02,
            lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=1e-3, weight_decay=1e-3),
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            scheduler_params={
                'T_0': 5,
                'eta_min': 1e-4,
                'T_mult': 1,
                'last_epoch': -1,
            },
            mask_type='entmax',
            seed=cfg.seed,
        )

        # train
        model.fit(
            np.array(train_x),
            np.array(train_y.values.ravel()),
            eval_set=[(np.array(valid_x), np.array(valid_y.values.ravel()))],
            max_epochs=cfg.max_epochs,
            patience=50,
            batch_size=cfg.batch_size,
            eval_metric=['auc', 'accuracy', AmexTabnet],
        )  # Last metric is used for early stopping

        # Saving best model
        saving_path_name = f"./fold{fold}"
        saved_filepath = model.save_model(saving_path_name)

        # model explain-ability
        explain_matrix, masks = model.explain(valid_x.values)
        explain_matrices.append(explain_matrix)
        masks_.append(masks[0])
        masks_.append(masks[1])

        # Inference
        oof_predictions[valid_idx] = model.predict_proba(valid_x.values)[:, 1]

        # if CFG
        # logodds function

        test_predictions += model.predict_proba(test_df.values)[:, 1] / 5
        feature_importances[f"importance_fold{fold}+1"] = model.feature_importances_

        # Loss , metric tracking
        stats[f'fold{fold + 1}_train_loss'] = model.history['loss']
        stats[f'fold{fold + 1}_val_metric'] = model.history['val_0_amex_tabnet']

        end = time.time()
        time_delta = np.round((end - start) / 60, 2)

        print(f'\nFold {fold + 1}/{cfg.N_folds} | {time_delta:.2f} min')

        # free memory
        del train_x, train_y
        del valid_x, valid_y
        gc.collect()

    print(f'OOF score across folds: {amex_metric.amex_metric_numpy(target_df, oof_predictions.flatten())}')

    feature_importances['mean_importance'] = feature_importances[['importance_fold0+1', 'importance_fold1+1']].mean(axis=1)
    feature_importances.sort_values(by='mean_importance', ascending=False, inplace=True)

    return stats, feature_importances, masks_, explain_matrices, test_df, test_predictions, saved_filepath
