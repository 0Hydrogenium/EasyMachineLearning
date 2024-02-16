import numpy as np
from matplotlib import pyplot as plt

from static.paint import PaintObject
from static.config import Config


def draw_learning_curve_total(input_dict, type, paint_object: PaintObject):
    plt.figure(figsize=(10, 6), dpi=300)

    if type == "train":
        for i, values in enumerate(input_dict.values()):
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
                color=paint_object.get_color_cur_list()[i]
            )

            plt.plot(
                train_sizes,
                train_scores_mean,
                "o-",
                color=paint_object.get_color_cur_list()[i],
                label=paint_object.get_label_cur_list()[i]
            )

    else:
        for i, values in enumerate(input_dict.values()):
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
                color=paint_object.get_color_cur_list()[i]
            )
            plt.plot(
                train_sizes,
                test_scores_mean,
                "o-",
                color=paint_object.get_color_cur_list()[i],
                label=paint_object.get_label_cur_list()[i]
            )

    plt.title(paint_object.get_name())

    plt.xlabel(paint_object.get_x_cur_label())
    plt.ylabel(paint_object.get_y_cur_label())
    plt.legend()

    # plt.savefig("./diagram/{}.png".format(title), dpi=300)
    # plt.show()

    paint_object.set_color_cur_num(len(input_dict.keys()))
    paint_object.set_label_cur_num(len(input_dict.keys()))

    return plt, paint_object

