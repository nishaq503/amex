""" https://www.kaggle.com/code/medali1992/amex-tabnetclassifier-feature-eng-0-791
"""
import logging
import warnings

import pandas
import seaborn
from matplotlib import pyplot

from amex.models import tabnet
from amex.utils import helpers
from amex.utils import paths

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = helpers.make_logger(__name__)

warnings.filterwarnings("ignore")

pyplot.style.use('ggplot')
orange_black = [
    '#fdc029', '#df861d', '#FF6347', '#aa3d01', '#a30e15', '#800000', '#171820'
]
pyplot.rcParams['figure.figsize'] = (16, 9)
pyplot.rcParams["figure.facecolor"] = '#FFFACD'
pyplot.rcParams["axes.facecolor"] = '#FFFFE0'
pyplot.rcParams["axes.grid"] = True
pyplot.rcParams["grid.color"] = orange_black[3]
pyplot.rcParams["grid.alpha"] = 0.5
pyplot.rcParams["grid.linestyle"] = '--'


class CFG:
    debug = False
    model = 'tabnet'
    n_folds = 5
    seed = 42
    batch_size = 512
    max_epochs = 60


def main():

    stats, feature_importances, masks_, explain_matrices, test_df, test_predictions, _ = tabnet.run_training(CFG)

    figure = pyplot.figure(figsize=(6, 6), dpi=300)
    for i in stats.filter(like='train', axis=1).columns.tolist():
        pyplot.plot(stats[i], label=str(i))
    pyplot.title('Train')
    pyplot.legend()
    pyplot.show()
    pyplot.close(figure)

    figure = pyplot.figure(figsize=(6, 6), dpi=300)
    for i in stats.filter(like='val', axis=1).columns.tolist():
        pyplot.plot(stats[i], label=str(i))
    pyplot.title('Valid')
    pyplot.legend()
    pyplot.show()
    pyplot.close(figure)

    figure = pyplot.figure(figsize=(6, 6), dpi=300)
    seaborn.barplot(y=feature_importances['feature'][:50], x=feature_importances['mean_importance'][:50], palette='inferno')
    pyplot.title('Mean Feature Importance by Folds')
    pyplot.show()
    pyplot.close(figure)

    figure, axes = pyplot.subplots(5, 2, figsize=(16, 16), dpi=300)
    axes = axes.flatten()

    k = -1
    for i, (mask, j) in enumerate(zip(masks_, axes)):
        seaborn.heatmap(mask[:150], ax=j)
        if i % 2 == 0:
            k += 1
        j.set_title(f"Fold {k} Mask for First 150 Instances")
    pyplot.tight_layout()
    pyplot.show()
    pyplot.close(figure)

    figure, axes = pyplot.subplots(len(explain_matrices), 1, figsize=(20, 8))
    for i, matrix in enumerate(explain_matrices):
        axes[i].set_title(f'Fold {i} Explain Matrix for First 150 Instances')
        seaborn.heatmap(matrix[:150], ax=axes[i])
    pyplot.tight_layout()
    pyplot.show()
    pyplot.close(figure)

    submission = pandas.DataFrame({'customer_ID': test_df.index, 'prediction': test_predictions})
    submission.to_csv(paths.SUBMISSION_PATH, index=False)

    return


if __name__ == '__main__':
    main()
