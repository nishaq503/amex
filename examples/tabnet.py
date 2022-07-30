""" https://www.kaggle.com/code/medali1992/amex-tabnetclassifier-feature-eng-0-791
"""

import warnings

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from amex.models import tabnet

warnings.filterwarnings("ignore")

plt.style.use('ggplot')
orange_black = [
    '#fdc029', '#df861d', '#FF6347', '#aa3d01', '#a30e15', '#800000', '#171820'
]
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams["figure.facecolor"] = '#FFFACD'
plt.rcParams["axes.facecolor"] = '#FFFFE0'
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = orange_black[3]
plt.rcParams["grid.alpha"] = 0.5
plt.rcParams["grid.linestyle"] = '--'


class CFG:
    DEBUG = False
    model = 'tabnet'
    N_folds = 5
    seed = 42
    batch_size = 512
    max_epochs = 60


def main():

    stats, feature_importances, masks_, explain_matrices, test_df, test_predictions, saved_filepath = tabnet.run_training(CFG)

    for i in stats.filter(like='train', axis=1).columns.tolist():
        plt.plot(stats[i], label=str(i))
    plt.title('Train loss')
    plt.legend()

    for i in stats.filter(like='val', axis=1).columns.tolist():
        plt.plot(stats[i], label=str(i))
    plt.title('Train RMSPE')
    plt.legend()

    sns.barplot(y=feature_importances['feature'][:50], x=feature_importances['mean_importance'][:50], palette='inferno')
    plt.title('Mean Feature Importance by Folds')
    plt.show()

    fig, axs = plt.subplots(5, 2, figsize=(16, 16))
    axs = axs.flatten()

    k = -1
    for i, (mask, j) in enumerate(zip(masks_, axs)):
        sns.heatmap(mask[:150], ax=j)
        if i % 2 == 0:
            k += 1
        j.set_title(f"Fold{k} Mask for First 150 Instances")
    plt.tight_layout()

    fig, axs = plt.subplots(len(explain_matrices), 1, figsize=(20, 8))
    for i, matrix in enumerate(explain_matrices):
        axs[i].set_title(f'Fold{i} Explain Matrix for First 150 Instances')
        sns.heatmap(matrix[:150], ax=axs[i])
    plt.tight_layout()

    sub = pd.DataFrame({'customer_ID': test_df.index,
                       'prediction': test_predictions})
    sub.to_csv('submission_tabnet.csv', index=False)

    return


if __name__ == '__main__':
    main()
