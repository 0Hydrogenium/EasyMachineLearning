import numpy as np
from matplotlib import pyplot as plt

from coding.llh.static.config import Config


def draw_learning_curve(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std):
    plt.figure(figsize=(10, 6))

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color=Config.COLORS[0]
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color=Config.COLORS[1]
    )

    plt.plot(
        train_sizes,
        train_scores_mean,
        "o-",
        color=Config.COLORS[0],
        label="Training score"
    )
    plt.plot(
        train_sizes,
        test_scores_mean,
        "o-",
        color=Config.COLORS[1],
        label="Cross-validation score"
    )

    plt.title("Learning curve")
    plt.xlabel("Sizes")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.show()
