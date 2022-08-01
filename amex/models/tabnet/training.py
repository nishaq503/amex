""" https://www.kaggle.com/code/medali1992/amex-tabnetclassifier-feature-eng-0-791
"""

import gc
import pathlib
import time

import numpy
import pandas
import torch
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold

from amex.utils import amex_metric
from amex.utils import helpers
from amex.utils import paths
from . import data

logger = helpers.make_logger(__name__)


class AmexMetric(Metric):

    def __init__(self):
        self._name = 'amex_metric'
        self._maximize = True

    def __call__(self, y_true, y_pred):
        return max(0, amex_metric.amex_metric_numpy(y_true, y_pred[:, 1]))


def run_training(cfg):
    train_df, target_df, test_df = data.feature_engineering()

    helpers.seed_everything(seed=cfg.seed)

    logger.info(f'Training {cfg.model}')
    logger.info(f'Seed {cfg.seed}')
    logger.info(f'Num Folds: {cfg.n_folds}')
    logger.info(f'{train_df.shape = }, {target_df.shape = }, {test_df.shape = }')
    logger.info(f'Num Features: {len(train_df.columns.values.tolist())}')

    logger.info(f'Starting pre-training unsupervised model ...')
    unsupervised_model = TabNetPretrainer(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax'  # "sparsemax"
    )
    unsupervised_model.fit(
        X_train=numpy.concatenate([numpy.array(train_df), numpy.array(test_df)]),
        pretraining_ratio=0.8,
        max_epochs=128,
    )
    logger.info(f'Finished pre-training unsupervised model ...')

    # Create out of folds array
    oof_predictions = numpy.zeros((train_df.shape[0]))
    test_predictions = numpy.zeros(test_df.shape[0])

    feature_importances = pandas.DataFrame()
    feature_importances['feature'] = train_df.columns.tolist()

    stats = pandas.DataFrame()
    explain_matrices = list()
    masks_ = list()
    saved_model_paths = list()

    fold_splitter = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    for fold, (train_idx, valid_idx) in enumerate(fold_splitter.split(train_df, target_df)):

        # DEBUG MODE
        if cfg.debug is True:
            if fold > 0:
                logger.info(f'DEBUG mode activated: Will train only one fold ...')
                break

        start = time.perf_counter()

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
            numpy.array(train_x),
            numpy.array(train_y.values.ravel()),
            eval_set=[(numpy.array(valid_x), numpy.array(valid_y.values.ravel()))],
            max_epochs=cfg.max_epochs,
            patience=50,
            batch_size=cfg.batch_size,
            eval_metric=['auc', 'accuracy', AmexMetric],  # Last metric is used for early stopping
            from_unsupervised=unsupervised_model,
        )

        # Saving best model
        saved_filepath = model.save_model(str(paths.WORKING_DIR.joinpath(f'fold_{fold + 1}')))
        saved_model_paths.append(pathlib.Path(saved_filepath).resolve())

        # model explain-ability
        explain_matrix, masks = model.explain(valid_x.values)
        explain_matrices.append(explain_matrix)
        masks_.append(masks[0])
        masks_.append(masks[1])

        # Inference
        oof_predictions[valid_idx] = model.predict_proba(valid_x.values)[:, 1]

        # log-odds function
        test_predictions += model.predict_proba(test_df.values)[:, 1] / 5
        feature_importances[f'importance_fold_{fold}+1'] = model.feature_importances_

        # Loss and metric tracking
        stats[f'fold_{fold + 1}_train_loss'] = model.history['loss']
        stats[f'fold_{fold + 1}_val_metric'] = model.history['val_0_amex_metric']

        time_taken = (time.perf_counter() - start) / 60
        logger.info(f'Fold {fold + 1}/{cfg.n_folds} | {time_taken:.2f} minutes')

        # free memory
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    oof_score = amex_metric.amex_metric_numpy(target_df, oof_predictions.flatten())
    logger.info(f'OOF score across folds: {oof_score:.6f}')

    feature_importances['mean_importance'] = feature_importances[['importance_fold_0+1', 'importance_fold_1+1']].mean(axis=1)
    feature_importances.sort_values(by='mean_importance', ascending=False, inplace=True)

    return stats, feature_importances, masks_, explain_matrices, test_df, test_predictions, saved_model_paths
