import gc
import time

import datatable
import numpy
import pandas
from sklearn.model_selection import StratifiedKFold
from tabnet.tabnet import TabNetClassifier
from tensorflow.python.keras import callbacks

from amex.utils import helpers
from amex.utils import amex_metric
from amex.utils import paths

logger = helpers.make_logger(__name__)


def amex_loss(y_true, y_pred):
    m = amex_metric.amex_metric_tensorflow(y_true, y_pred)
    return 1 - m


def read_data() -> pandas.DataFrame:
    train_df: pandas.DataFrame = datatable.fread(paths.TRAIN_DATA_PATH).to_pandas()
    train_df['S_2'] = pandas.to_datetime(train_df['S_2']).astype('datetime64[ns]')

    train_df = train_df.groupby('customer_ID').tail(1).reset_index(drop=True)
    train_df = train_df.fillna(-1)
    logger.info(f'{train_df.shape = }')

    target_df: pandas.DataFrame = datatable.fread(paths.TRAIN_LABELS_PATH).to_pandas()
    logger.info(f'{target_df.shape = }')

    train_df = train_df.merge(target_df, on='customer_ID')
    logger.info(f'{train_df.shape = }')

    return train_df


def run_training(train_df: pandas.DataFrame, all_features: list[str], cfg):
    helpers.seed_everything(cfg.seed)

    train_features = train_df[all_features].to_numpy(dtype=numpy.float32)
    train_targets = train_df['target'].to_numpy(dtype=numpy.float32)

    logger.info(f'Training {cfg.model} ...')
    logger.info(f'{cfg.seed = } ...')
    logger.info(f'{cfg.N_folds = } ...')
    logger.info(f'{train_features.shape = } ...')
    logger.info(f'{train_targets.shape = } ...')
    logger.info(f'{len(all_features) = } ...')

    models = list()

    splitter = StratifiedKFold(n_splits=cfg.N_folds, shuffle=True, random_state=cfg.seed)

    for k, (train_idx, valid_idx) in enumerate(splitter.split(train_features, train_targets)):

        # DEBUG MODE
        if cfg.DEBUG is True:
            if k > 0:
                print('\nDEBUG mode activated: Will train only one fold...\n')
                break

        start = time.perf_counter()

        model = TabNetClassifier(
            feature_columns=None,
            num_classes=2,
            output_dim=32,
            num_features=len(all_features),
            # scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
            # scheduler_params={"T_max": 6},
            # mask_type='sparsemax',
            # seed=CFG.seed
        )

        model.compile(
            optimizer='adam',
            loss='hinge',
            metrics=[amex_metric.amex_metric_tensorflow],
        )

        # train
        train_x, train_y = train_features[train_idx], train_targets[train_idx]
        valid_x, valid_y = train_features[valid_idx], train_targets[valid_idx]

        model.fit(
            x=train_x,
            y=train_y,
            batch_size=2048,
            epochs=50,
            validation_data=(valid_x, valid_y),
            callbacks=[
                # callbacks.EarlyStopping(min_delta=1e-4, patience=10),
            ],
        )

        models.append(model)

        end = time.perf_counter()
        time_delta = (end - start) / 60

        logger.info(f'Fold {k + 1}/{cfg.N_folds} | {time_delta:.2f} minutes ...')

        # free memory
        del train_x, train_y
        del valid_x, valid_y
        gc.collect()

    return models
