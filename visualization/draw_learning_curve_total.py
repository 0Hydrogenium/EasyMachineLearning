import numpy as np
from matplotlib import pyplot as plt

from static.config import Config


def draw_learning_curve_total(input_dict, type):
    plt.figure(figsize=(10, 6), dpi=300)

    if type == "train":
        i = 0
        for label_name, values in input_dict.items():
            train_sizes = values[0]
            train_scores_mean = values[1]
            train_scores_std = values[2]
            test_scores_mean = values[3]
            test_scores_std = values[4]

            plt.fill_between(
                train_sizes,
                train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std,
                alpha=0.1,
                color=Config.COLORS[i]
            )

            plt.plot(
                train_sizes,
                train_scores_mean,
                "o-",
                color=Config.COLORS[i],
                label=label_name
            )

            i += 1

        title = "Training Learning curve"
        # plt.title(title)

    else:
        i = 0
        for label_name, values in input_dict.items():
            train_sizes = values[0]
            train_scores_mean = values[1]
            train_scores_std = values[2]
            test_scores_mean = values[3]
            test_scores_std = values[4]

            plt.fill_between(
                train_sizes,
                test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std,
                alpha=0.1,
                color=Config.COLORS[i]
            )
            plt.plot(
                train_sizes,
                test_scores_mean,
                "o-",
                color=Config.COLORS[i],
                label=label_name
            )

            i += 1

        title = "Cross-validation Learning curve"
        # plt.title(title)

    plt.xlabel("Sizes")
    plt.ylabel("Adjusted R-square")
    plt.legend()

    # plt.savefig("./diagram/{}.png".format(title), dpi=300)
    # plt.show()
    return plt

