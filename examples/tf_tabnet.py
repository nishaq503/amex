import logging

from amex.models import tf_tabnet
from amex.utils import helpers

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = helpers.make_logger(__name__)


class CFG:
    DEBUG = False
    model = 'tabnet'
    N_folds = 5
    seed = 42


cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
non_features = ['target', 'customer_ID', 'S_2'] + cat_features

train_df = tf_tabnet.read_data()
all_features = [col for col in train_df.columns if col not in non_features]
logger.info(f'{len(all_features) = }')

cat_index = [train_df.columns.get_loc(cat_features[cat]) for cat in range(len(cat_features))]
logger.info(f'{len(cat_features) = }')

models = tf_tabnet.run_training(train_df, all_features, CFG())

# features_importances = models[-1].feature_importances_
# sorting_indices = numpy.argsort(features_importances)
# features_importances_sorted = features_importances[sorting_indices]
#
# feature_names = train_df[all_features].columns
# features_sorted = feature_names[sorting_indices]
#
# # plot feature importances
# pyplot.figure(figsize=(12, 16))
#
# # n features to plot
# n = 50
#
# pyplot.barh(features_sorted[-n:], features_importances_sorted[-n:])
# pyplot.title(f"Feature Importances: {tf_tabnet.CFG.model}")
