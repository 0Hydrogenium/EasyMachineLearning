from matplotlib import pyplot as plt

from classes.static_custom_class import *


def draw_learning_curve(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std):
    plt.figure(figsize=(10, 6))

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color=StaticValue.COLORS[0]
    )
    plt.plot(
        train_sizes,
        train_scores_mean,
        "o-",
        color=StaticValue.COLORS[0],
        label="Training score"
    )

    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color=StaticValue.COLORS[1]
    )
    plt.plot(
        train_sizes,
        test_scores_mean,
        "o-",
        color=StaticValue.COLORS[1],
        label="Cross-validation score"
    )

    plt.title("Learning curve")
    plt.xlabel("Sizes")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.show()
